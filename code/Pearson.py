import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# create a dictionary with key of indicator_names_unique
# and value of a dataframe with only that indicator for countries
# create a dictionary with key of indicator_names_unique
# and value of a dataframe with only that indicator for countries
def df_to_dic():
    indicator_dict = {}
    for indicator_name in indicator_names_unique:
        indicator_dict[indicator_name] = df_countries[df_countries["Indicator Name"] == indicator_name]
    country_dict = {}
    for country_name in countries_unique:
        country_dict[country_name] = df_countries[df_countries["Country Name"] == country_name]
    return indicator_dict, country_dict


if __name__ == '__main__':
    df = pd.read_csv('./data/WDICSV.csv')
    # filter country related Data
    first_country_index = np.where(np.array(df["Country Name"]) == "Afghanistan")[0][0]
    # Country data: all the data we need
    df_countries = df.iloc[first_country_index:, :]
    # Geopolitical region data
    df_areas = df.iloc[:first_country_index, :]
    # get all unique indicator and country representation
    indicators_unique = np.array(df["Indicator Code"].unique())
    countries_unique = np.array(df["Country Name"].unique())
    country_codes_unique = np.array(df["Country Code"].unique())
    indicator_names_unique = np.array(df["Indicator Name"].unique())
    countries = df_countries["Country Name"].unique()
    areas = df_areas["Country Name"].unique()
    country_codes = df_countries["Country Code"].unique()
    area_codes = df_areas["Country Code"].unique()
    indicator_dict, country_dict = df_to_dic()

    # No lagging
    # Define the countries of interest and the years for analysis
    countries_of_interest = ['Korea, Rep.', 'Niger']
    gdp_indicator_name = 'GDP per capita (current US$)'
    gdp_years = [str(year) for year in range(1995, 2010)]
    education_indicator_names = [
        'Primary completion rate, total (% of relevant age group)',
        'Primary education, teachers',
        'Children out of school (% of primary school age)',
        'School enrollment, secondary (% gross)',
        'School enrollment, tertiary (% gross)'
    ]

    # Extract the GDP per capita data
    gdp_df = indicator_dict.get(gdp_indicator_name)

    # Filter for countries of interest and set index
    gdp_df_filtered = gdp_df[gdp_df['Country Name'].isin(countries_of_interest)]
    gdp_df_filtered.set_index('Country Name', inplace=True)
    gdp_df_filtered_years = gdp_df_filtered[gdp_years]

    # Prepare to store correlation results
    correlation_results = {}

    # Iterate through each education indicator and calculate correlation with GDP
    for indicator_name in education_indicator_names:
        # Get the education data for the current indicator
        education_df = indicator_dict.get(indicator_name)
        if education_df is not None:
            education_df = education_df[education_df['Country Name'].isin(countries_of_interest)]
            education_df.set_index('Country Name', inplace=True)
            education_df_years = education_df[gdp_years]

            # Combine the data for correlation analysis by transposing and joining
            combined = education_df_years.T.join(gdp_df_filtered_years.T, lsuffix='_edu', rsuffix='_gdp').dropna()

            # Calculate correlation for Korea, Rep.
            korea_data = combined[['Korea, Rep._edu', 'Korea, Rep._gdp']].dropna()
            korea_corr = korea_data.corr().iloc[0, 1]  # Extract the correlation coefficient between the two columns
            correlation_results[f"Korea, Rep. - {indicator_name}"] = korea_corr

            # Calculate correlation for Niger
            niger_data = combined[['Niger_edu', 'Niger_gdp']].dropna()
            niger_corr = niger_data.corr(method='pearson').iloc[
                0, 1]  # Extract the correlation coefficient between the two columns
            correlation_results[f"Niger - {indicator_name}"] = niger_corr

    # Print the correlation results for each indicator and each country
    print("Correlation Results:")
    for indicator, corr_value in correlation_results.items():
        print(f"{indicator}: {corr_value}")

    # Optionally, you can visualize the results using a bar plot for better readability
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(correlation_results.keys()), y=list(correlation_results.values()))
    plt.xticks(rotation=90, ha='right')
    plt.title('Correlation between Education Indicators and GDP per capita')
    plt.ylabel('Correlation Coefficient')
    plt.tight_layout()
    plt.show()

    # Create a DataFrame for better visualization
    corr_df = pd.DataFrame(
        list(correlation_results.items()),
        columns=["Indicator and Country", "Correlation Coefficient"]
    )

    # Sort values for better presentation
    corr_df.sort_values(by="Correlation Coefficient", ascending=False, inplace=True)

    # Create the bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=corr_df,
        x="Correlation Coefficient",
        y="Indicator and Country",
        legend=False
    )
    plt.title('Correlation between Education Indicators and GDP per capita')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Education Indicator and Country')
    plt.tight_layout()
    plt.grid(True, axis='x', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.show()
