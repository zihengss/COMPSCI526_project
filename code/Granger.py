import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
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


def perform_adf_test(df, country_name, indicator_code):
    """
    Perform the Augmented Dickey-Fuller (ADF) test for stationarity on a specified indicator for a given country.

    Args:
    - dataset
    - country_name (str): Name of the country to filter the data.
    - indicator_code (str): Code of the indicator to analyze (e.g., "NY.GDP.MKTP.KD.ZG" for GDP growth).

    Returns:
    - dict: Summary of the ADF test results, including the test statistic, p-value, critical values, and stationarity status.
    """
    try:
        # Filter data for the specified country and indicator
        filtered_data = df[(df["Country Name"] == country_name) &
                           (df["Indicator Code"] == indicator_code)]

        # Extract the relevant years as a time series
        time_series = filtered_data.iloc[:, 4:].transpose()  # Years start from column index 4
        time_series.columns = ["Value"]
        time_series.reset_index(drop=True, inplace=True)
        time_series["Value"] = pd.to_numeric(time_series["Value"], errors="coerce")
        time_series = time_series.dropna()

        plt.plot(time_series.index, time_series["Value"])
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.show()

        # Perform the ADF test
        adf_result = adfuller(time_series["Value"])
        result_summary = {
            "Country": country_name,
            "Indicator": indicator_code,
            "Indicator Name": filtered_data["Indicator Name"].iloc[0],
            "ADF Statistic": adf_result[0],
            "p-value": adf_result[1],
            "Critical Values": adf_result[4],
            "Stationarity": "Stationary" if adf_result[1] <= 0.05 else "Non-stationary"
        }
        return result_summary

    except Exception as e:
        return {"Error": str(e)}


def check_stationarity(series, name):
    result = adfuller(series)
    print(f"{name} ADF Statistic: {result[0]}, p-value: {result[1]}")
    if result[1] > 0.05:
        print(f"{name} is non-stationary. We must transform it.")
    else:
        print(f"{name} is stationary.")


def run_granger_causality_on_indicators(data, country_name, gdp_indicator, indicators_dict, years, max_lag=10):
    """
    Perform Granger causality analysis between GDP and all other indicators for a specific country.

    Args:
    - data (dict): A dictionary containing indicator dataframes.
    - country_name (str): The country to analyze.
    - gdp_indicator (str): The GDP indicator name (e.g., "GDP per capita (current US$)").
    - indicators (dict): A dictionary of indicator names and their respective dataframes.
    - years (list): List of years to include in the analysis.
    - max_lag (int): Maximum number of lags for Granger causality testing.

    Returns:
    - dict: Results of Granger causality tests for each indicator.
    """

    def check_stationarity(series):
        result = adfuller(series)
        return result[1] <= 0.05  # True if stationary, False otherwise

    results = {}
    gdp_df = data.get(gdp_indicator)
    gdp_data = gdp_df[gdp_df["Country Name"] == country_name][years].T.dropna()
    gdp_data.columns = ["GDP"]

    # Check and transform GDP if needed
    if not check_stationarity(gdp_data.squeeze()):
        gdp_data = gdp_data.diff().dropna()
        gdp_data.columns = ["GDP"]

    for indicator_name, indicator_df in indicators_dict.items():
        try:
            # Extract data for the indicator
            indicator_data = indicator_df[indicator_df["Country Name"] == country_name][years].T.dropna()
            indicator_data.columns = ["Indicator"]

            # Check and transform indicator if needed
            if not check_stationarity(indicator_data.squeeze()):
                indicator_data = indicator_data.diff().dropna()
                indicator_data.columns = ["Indicator"]

            # Combine GDP and indicator data
            combined_data = pd.concat([indicator_data, gdp_data], axis=1).dropna()

            # Apply Granger causality test
            granger_test_result = grangercausalitytests(combined_data, max_lag, verbose=False)

            # Extract p-values for each lag
            p_values = {lag: granger_test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag + 1)}

            # Determine if Granger causality exists
            causality_found = any(p <= 0.05 for p in p_values.values())

            # Store results
            results[indicator_name] = {
                "P-Values": p_values,
                "Causality Found": causality_found
            }

        except Exception as e:
            results[indicator_name] = {"Error": str(e)}

    return results


def granger_causality_analysis(countries, gdp_indicator, indicators, countries_dict, max_lag=15):
    """
    Perform Granger causality analysis for countries using panel data in countries_dict.

    Args:
        countries (list): List of countries to analyze.
        gdp_indicator (str): Name of the GDP indicator.
        education_indicators (list): List of education-related indicators to analyze.
        countries_dict (dict): Dictionary where keys are country names, and values are DataFrames with panel data.
        max_lag (int): Maximum lag to consider for Granger causality.

    Returns:
        dict: Results of Granger causality tests for each country and indicator.
    """
    # Dictionary to store results
    results = {}

    for country in countries:
        print(f"\n=== Granger Causality Analysis for {country} ===")
        country_results = {}

        # Ensure the country exists in the countries_dict
        if country not in countries_dict:
            print(f"Data for '{country}' not found in countries_dict. Skipping.")
            continue

        # Get the panel data for the country
        panel_data = countries_dict[country]

        # Ensure the GDP indicator exists in the panel data
        gdp_data = panel_data[panel_data['Indicator Name'] == gdp_indicator]
        if gdp_data.empty:
            print(f"GDP indicator '{gdp_indicator}' not found for {country}. Skipping.")
            continue

        # Transform GDP data
        gdp_series = gdp_data.iloc[:, 4:].T  # Extract years as columns, transpose
        gdp_series.columns = ['GDP Growth']  # Rename column
        gdp_series.index = gdp_series.index.astype(int)  # Ensure years are integers

        for edu_indicator in indicators:
            try:
                print(f"\nAnalyzing indicator: {edu_indicator}")

                # Ensure the education indicator exists in the panel data
                edu_data = panel_data[panel_data['Indicator Name'] == edu_indicator]
                if edu_data.empty:
                    print(f"Indicator '{edu_indicator}' not found for {country}. Skipping.")
                    continue

                # Transform education data
                edu_series = edu_data.iloc[:, 4:].T  # Extract years as columns, transpose
                edu_series.columns = ['Indicator']  # Rename column
                edu_series.index = edu_series.index.astype(int)  # Ensure years are integers

                # Combine GDP and education indicator into a single DataFrame
                data = pd.concat([gdp_series, edu_series], axis=1).dropna()  # Drop rows with NaN values

                if data.shape[0] < 10:
                    continue
                # Perform Augmented Dickey-Fuller (ADF) test for stationarity
                adf_gdp = adfuller(data['GDP Growth'])
                adf_ind = adfuller(data['Indicator'])

                # Differencing if non-stationary
                data_diff = pd.DataFrame(index=data.index)
                if adf_gdp[1] > 0.05:
                    data_diff['GDP Growth Diff'] = data['GDP Growth'].diff()
                else:
                    data_diff['GDP Growth Diff'] = data['GDP Growth']
                if adf_ind[1] > 0.05:
                    data_diff['Indicator Diff'] = data['Indicator'].diff()
                else:
                    data_diff['Indicator Diff'] = data['Indicator']

                # Drop NaN values after differencing
                data_diff = data_diff.dropna()

                # Check asgain for stationarity, if not, difference again
                adf_gdp_2 = adfuller(data_diff['GDP Growth Diff'])
                adf_ind_2 = adfuller(data_diff['Indicator Diff'])

                if adf_gdp_2[1] > 0.05:
                    data_diff['GDP Growth Diff'] = data_diff['GDP Growth Diff'].diff()
                else:
                    data_diff['GDP Growth Diff'] = data_diff['GDP Growth Diff']
                if adf_ind_2[1] > 0.05:
                    data_diff['Indicator Diff'] = data_diff['Indicator Diff'].diff()
                else:
                    data_diff['Indicator Diff'] = data_diff['Indicator Diff']

                data_diff = data_diff.dropna()

                # Check for constant series
                if data_diff['GDP Growth Diff'].nunique() <= 1 or data_diff['Indicator Diff'].nunique() <= 1:
                    print(
                        f"One of the series is constant after differencing for {country} with indicator '{edu_indicator}'. Skipping.")
                    continue

                try:
                    # Try Granger causality test
                    # print(data_diff)
                    max_lag = int(data_diff.shape[0] / 3) - 1

                    granger_results = grangercausalitytests(data_diff[['GDP Growth Diff', 'Indicator Diff']],
                                                            maxlag=max_lag, verbose=False)

                    # Collect p-values for each lag
                    p_values = [granger_results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                    min_p_value = min(p_values)
                    optimal_lag = p_values.index(min_p_value) + 1

                    # Check if any p-value is less than 0.05
                    if (min_p_value < 0.05):
                        # print(f"Indicator '{edu_indicator}' Granger-causes GDP Growth at lag {optimal_lag} (p-value: {min_p_value:.4f})")
                        country_results[edu_indicator] = {
                            'Granger_causality': True,
                            'Optimal_lag': optimal_lag,
                            'p_value': min_p_value
                        }
                        print(
                            f"Granger causality detected for indicator '{edu_indicator}' in {country} with lag {optimal_lag}")
                    else:
                        # print(f"No Granger causality detected for indicator '{edu_indicator}' in {country}.")
                        country_results[edu_indicator] = {
                            'Granger_causality': False,
                            'Optimal_lag': optimal_lag,
                            'p_value': min_p_value
                        }
                except Exception as e:
                    print(f"An error occurred while testing Granger causality for '{edu_indicator}' in {country}: {e}")
                    continue
            except Exception as e:
                print(f"An error occurred while processing indicator '{edu_indicator}' in {country}: {e}")
                continue

        # Store results for the country
        results[country] = country_results
    return results


def flatten_granger_results(results):
    """
    Flatten the nested Granger causality results into a DataFrame.

    Args:
        results (dict): Nested dictionary from granger_causality_analysis.

    Returns:
        pd.DataFrame: Flattened DataFrame with columns: Country, Indicator, Lag, P-Value.
    """
    rows = []
    for country, indicators in results.items():
        for indicator, result in indicators.items():
            if result['Granger_causality']:  # Only include cases with Granger causality
                rows.append({
                    "Country": country,
                    "Indicator": indicator,
                    "Lag": result['Optimal_lag'],
                    "P-Value": result['p_value']
                })
    return pd.DataFrame(rows)


def plot_heatmap_by_lag(results_df):
    """
    Plot a heatmap showing the number of lags as the color intensity instead of p-values.

    Args:
        results_df (pd.DataFrame): Flattened results DataFrame with columns:
                                   Country, Indicator, Lag, P-Value.
    """
    # Specify the desired country order
    #     country_order = ["India", "China", "Niger", "Korea, Rep.",  "United Kingdom", "United States", "Germany", "Japan"]

    # Pivot the data for heatmap visualization
    pivot_data = results_df.pivot(index="Indicator", columns="Country", values="Lag")

    # Plot the heatmap of Lags
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pivot_data,
        fmt=".0f",
        cmap="viridis",  # Use a diverging colormap for lags
        cbar_kws={'label': 'Number of Lags'},
        linewidths=0.5,
    )
    plt.title("Heatmap of Granger Causality Lags Across Countries")
    plt.xlabel("Country")
    plt.ylabel("Indicator")
    plt.tight_layout()
    plt.show()


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
    warnings.filterwarnings("ignore")
    # Time Lag Case test
    # -----------------------------------------------------------------------------
    # Define the target country and indicators
    country_name = "Korea, Rep."
    gdp_indicator_name = "GDP per capita (current US$)"
    education_indicator_name = "School enrollment, tertiary (% gross)"

    # Select the desired years for analysis
    years = [str(year) for year in range(1985, 2024)]

    # Extract GDP and education data for the country
    gdp_df = indicator_dict.get(gdp_indicator_name)
    education_df = indicator_dict.get(education_indicator_name)

    # Filter the data for the selected country and years
    gdp_data = gdp_df[gdp_df["Country Name"] == country_name][years].T.dropna()
    education_data = education_df[education_df["Country Name"] == country_name][years].T.dropna()

    # Ensure both time series have the same index and length
    gdp_data.columns = ["GDP"]
    education_data.columns = ["Education"]
    combined_data = pd.concat([education_data, gdp_data], axis=1).dropna()

    # Apply Granger causality test
    granger_test_result = grangercausalitytests(combined_data, maxlag=10, verbose=False)

    # Extract p-values from the Granger test results for each lag
    p_values = {
        6: granger_test_result[6][0]['ssr_chi2test'][1],
        7: granger_test_result[7][0]['ssr_chi2test'][1],
        8: granger_test_result[8][0]['ssr_chi2test'][1],
        9: granger_test_result[9][0]['ssr_chi2test'][1],
        10: granger_test_result[10][0]['ssr_chi2test'][1]
    }

    # Plot the p-values for each lag
    lags = list(p_values.keys())
    p_vals = list(p_values.values())

    plt.figure(figsize=(8, 6))
    sns.barplot(x=lags, y=p_vals, hue=lags, palette="Blues_d", legend=False)
    plt.axhline(0.05, color='r', linestyle='--', label='Significance Level (0.05)')
    plt.title('Granger Causality Test P-values for Different Lags')
    plt.xlabel('Lag')
    plt.ylabel('P-value')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # -----------------------------------------------------------------------------
    # Check for stationary
    check_stationarity(gdp_data.squeeze(), "GDP")
    check_stationarity(education_data.squeeze(), "Education")

    # If non-stationary, difference the series
    gdp_data_diff = gdp_data.diff().dropna()
    education_data_diff = education_data.diff().dropna()

    optimal_lag = 9

    # Extract p-values dynamically for all tested lags
    p_values = {lag: granger_test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, optimal_lag + 1)}

    # Plot dynamically adjusted p-values
    lags = list(p_values.keys())
    p_vals = list(p_values.values())

    plt.figure(figsize=(8, 6))
    sns.barplot(x=lags, y=p_vals, legend=False)
    plt.axhline(0.05, color='r', linestyle='--', label='Significance Level (0.05)')
    plt.title(
        'Granger Causality for School enrollment, tertiary (% gross) on  GDP per capita. P-values for Different Lags')
    plt.xlabel('Lag')
    plt.ylabel('P-value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # List of selected countries (you can modify this list)
    developed_countries = ['United States', 'Germany', 'Japan']
    emerging_countries = ['India', 'China']
    case_study = ["Niger", "Korea, Rep."]
    selected_countries = developed_countries + emerging_countries + case_study

    # check if string contains word "school"
    contains_school = np.array([True if 'school' in item.lower() else False for item in indicator_names_unique])
    education_indicators = indicator_names_unique[contains_school]

    # Countries and indicators

    gdp_indicator = "GDP per capita (current US$)"
    # check if string contains word "school"
    contains_school = np.array([True if 'school' in item.lower() else False for item in indicator_names_unique])
    education_indicators = indicator_names_unique[contains_school]
    # indicator_names_unique

    # Perform analysis
    results_education = granger_causality_analysis(
        countries=selected_countries,
        gdp_indicator=gdp_indicator,
        indicators=education_indicators,
        countries_dict=country_dict
    )

    # Plot the heatmap of P-Values
    results_df = flatten_granger_results(results_education)
    pivot_data = results_df.pivot(index="Indicator", columns="Country", values="P-Value")

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        cbar_kws={'label': 'P-Value'},
        linewidths=0.5,
    )
    plt.title("Heatmap of Granger Causality P-Values Across Countries")
    plt.xlabel("Country")
    plt.ylabel("Indicator")
    plt.tight_layout()
    plt.show()

    print("\n=== Granger Causality Summary for all Indicators ===")
    for country, indicators in results_education.items():
        print(f"\nCountry: {country}")
        for indicator, result in indicators.items():
            if result['Granger_causality']:
                print(
                    f" - Indicator '{indicator}' Granger-causes GDP Growth at lag {result['Optimal_lag']} (p-value: {result['p_value']:.4f})")
            else:
                continue

    plot_heatmap_by_lag(results_df)
    results_df.head()
