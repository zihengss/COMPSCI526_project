import numpy as np
import pandas as pd
import statsmodels.api as sm


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

    x_indicator_names = [
        'School enrollment, primary (% gross)',
        'School enrollment, secondary (% gross)',
        'School enrollment, tertiary (% gross)',
        'Government expenditure on education, total (% of GDP)',
        'Pupil-teacher ratio, primary',
        'Pupil-teacher ratio, secondary',
        'Pupil-teacher ratio, tertiary',
        'Current health expenditure (% of GDP)',
        'Control of Corruption: Estimate',
        'Inflation, consumer prices (annual %)'
    ]
    y_indicator_name = 'GDP per capita (current US$)'

    df1 = indicator_dict[y_indicator_name]

    df1_melted = df1.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                          var_name='Year', value_name=y_indicator_name)
    df1_melted = df1_melted[['Country Name', 'Year', y_indicator_name]]
    df1_melted['Year'] = df1_melted['Year'].astype(int)

    combined_df = df1_melted.copy()

    # loop through education expenditure indicator
    for x_indicator_name in x_indicator_names:
        df2 = indicator_dict[x_indicator_name]
        df2_melted = df2.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                              var_name='Year', value_name=x_indicator_name)
        df2_melted = df2_melted[['Country Name', 'Year', x_indicator_name]]
        df2_melted['Year'] = df2_melted['Year'].astype(int)

        combined_df = pd.merge(combined_df, df2_melted, on=['Country Name', 'Year'], how='inner')

    cleaned_combined_df = combined_df.dropna()

    # build model
    y = cleaned_combined_df[y_indicator_name]
    X = cleaned_combined_df[x_indicator_names]
    X = sm.add_constant(X)

    # Fit the OLS regression model
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())