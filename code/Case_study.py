import numpy as np
import pandas as pd
import plotly.graph_objects as go


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


def case_study(indicator_dict):
    # Select countries of interest
    countries_of_interest = ['Korea, Rep.', 'Niger']
    indicator_names = ['School enrollment, tertiary (% gross)', 'GDP per capita (current US$)']

    for indicator_name in indicator_names:
        # Filter the dataframe for those countries
        indicator_df_filtered = indicator_dict[indicator_name]
        indicator_df_filtered = indicator_df_filtered[indicator_df_filtered['Country Name'].isin(countries_of_interest)]

        # Set 'Country Name' as index and filter out necessary columns (years)
        indicator_df_filtered.set_index('Country Name', inplace=True)
        years = [str(year) for year in range(1960, 2024)]  # Modify this range as needed
        indicator_df_filtered_years = indicator_df_filtered[years]

        # Create the plot using Plotly
        fig = go.Figure()

        for country in countries_of_interest:
            fig.add_trace(go.Scatter(
                x=years,
                y=indicator_df_filtered_years.loc[country],
                mode='lines+markers',
                name=country
            ))

        # Update layout
        fig.update_layout(
            title=f'{indicator_name}',
            xaxis_title='Year',
            yaxis_title=f'{indicator_name}',
            legend_title='Country',
            template='plotly_white',
            hovermode='x unified'
        )

        # Show the plot
        fig.show()


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

    # case study
    indicator_dict, country_dict = df_to_dic()
    case_study(indicator_dict)
