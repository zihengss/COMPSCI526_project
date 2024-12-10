import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


def check_missing(df):
    years = range(1960, 2024)
    num_non_nulls = []

    for year in years:
        num_non_nulls.append(len(df) - df[str(year)].isnull().sum())

    num_non_nulls = np.array(num_non_nulls)
    year = np.array(years)

    plt.plot(year, num_non_nulls, 'ro')
    plt.xlabel('Year')
    plt.ylabel('Number of Non-Null Values')
    plt.title('Number of Non-Null Values per Year')
    plt.show()


# a tool that find full indicator name(in the table) with part of name
def search_indicator(name, indicator_names_unique):
    search_term = name
    l = [indicator for indicator in indicator_names_unique if search_term.lower() in indicator.lower()]
    print(len(l), l)


# a tool that find full country name(in the table) with part of name
def search_country(name, countries_unique):
    search_term = name
    l = [country for country in countries_unique if search_term.lower() in country.lower()]
    print(len(l), l)


def calculate_non_null_coverage(data, countries):
    """
    Calculate the percentage of non-null coverage for each country.

    Args:
        data (pd.DataFrame): DataFrame containing the data with 'Country' and numerical columns.
        countries (list): List of countries to calculate coverage for.

    Returns:
        pd.DataFrame: DataFrame with countries and their corresponding non-null coverage percentage.
    """
    # Filter the dataset for selected countries
    filtered_data = data[data['Country Name'].isin(countries)]

    # Group by country and calculate non-null percentages
    coverage = (
        filtered_data.set_index('Country Name')
        .iloc[:, 4:]  # Select only the numerical columns (skipping metadata columns)
        .notnull()
        .mean(axis=1)
        .groupby('Country Name')
        .mean()
        .sort_values(ascending=False)
    )

    # Convert to a DataFrame for better visualization
    coverage_df = pd.DataFrame({
        'Country': coverage.index,
        'Coverage (%)': (coverage.values * 100).round(2)
    })

    return coverage_df


def check_coverage():
    df_indicator_names = pd.read_csv('./data/WDISeries.csv')
    code_map = pd.Series(df_indicator_names['Indicator Name'].values, index=df_indicator_names['Series Code']).to_dict()

    years = [str(i) for i in range(1970, 2022)]

    # add column for each row representing the percent of years where the value is not NaN in that column
    df['percent_not_nan'] = df[years].notna().mean(axis=1)

    # average along row values where indicator code matches to get percent of the data for that code that is not NaN
    coverage = df.groupby('Indicator Code')['percent_not_nan'].mean()
    coverage = coverage.sort_values(ascending=False)

    PAGE = 1
    print('\033[1mCoverage   Title\033[0m')
    for i in range(10 * (PAGE - 1), 10 * PAGE):
        print(f'{100 * coverage.iloc[i]:.2f}%    ', code_map.get(coverage.index[i], 'unknown'),
              f'({coverage.index[i]})')
    print("----------------------------")
    pairs = [
        (f'{100 * coverage.iloc[i]:.2f}%    ', code_map.get(coverage.index[i], 'unknown'), f'({coverage.index[i]})') for
        i in range(len(coverage))]
    print_cnt = 0
    for i in filter(lambda x: 'Education' in x[1] or 'School' in x[1], pairs):
        print(*i)
        print_cnt += 1
        if print_cnt > 10:
            break


if __name__ == '__main__':
    df = pd.read_csv('./data/WDICSV.csv')
    # filter country related Data
    first_country_index = np.where(np.array(df["Country Name"]) == "Afghanistan")[0][0]
    # Country data: all the data we need
    df_countries = df.iloc[first_country_index:, :]
    # Geopolitical region data
    df_areas = df.iloc[:first_country_index, :]
    # check missing value distribution over years
    check_missing(df_countries)

    # get all unique indicator and country representation
    indicators_unique = np.array(df["Indicator Code"].unique())
    countries_unique = np.array(df["Country Name"].unique())
    country_codes_unique = np.array(df["Country Code"].unique())
    indicator_names_unique = np.array(df["Indicator Name"].unique())
    countries = df_countries["Country Name"].unique()
    areas = df_areas["Country Name"].unique()
    country_codes = df_countries["Country Code"].unique()
    area_codes = df_areas["Country Code"].unique()

    # check number of non_null values per country
    df[df["Country Name"] == "United States"].isnull().sum()
    # checking null values from 1990s until 2022
    df[df["Country Name"] == "United States"].iloc[:, 34:-1].notnull().sum().head()
    # search example
    search_country("China", countries_unique)
    search_indicator("GDP", indicator_names_unique)

    # sort coverage by country by Coverage ascending
    coverage_by_country = calculate_non_null_coverage(df, countries_unique)
    plt.figure(figsize=(30, 8))
    plt.bar(x=coverage_by_country["Country"], height=coverage_by_country["Coverage (%)"])
    plt.xlabel("Country")
    plt.ylabel("Coverage (%)")
    plt.title("Overall Coverage of Non-Null Values by Country (all indicators)", fontsize=16)
    # decrease fontsize
    plt.xticks(rotation=90, fontsize=9)
    plt.show()

    # school indicators
    df_school = df[df['Indicator Name'].str.contains('school', case=False, na=False)]
    # sort coverage by country by Coverage ascending
    coverage_by_country_school = calculate_non_null_coverage(df_school, countries_unique)
    plt.figure(figsize=(30, 6))
    plt.bar(x=coverage_by_country["Country"], height=coverage_by_country_school["Coverage (%)"])
    plt.xlabel("Country")
    plt.ylabel("Coverage (%)")
    plt.title("Coverage of Non-Null Values in School Indicators by Country")
    # decrease fontsize
    plt.xticks(rotation=90, fontsize=8)
    plt.show()

    # check Coverage
    check_coverage()