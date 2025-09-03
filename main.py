import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re


def clean_qaly_data(qaly_string):
    """Extracts the upper bound of QALYs from a string."""
    if isinstance(qaly_string, str):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", qaly_string)
        if numbers:
            return max(map(float, numbers))
    return np.nan


def model_drug_pricing(drug_name, anchor_country, anchor_price, cap_price_flag, cap_percentage):
    """
    Models and contrasts pharmaceutical pricing structures.
    """
    # Load data
    drug_costs_df = pd.read_csv('drug_costs.csv')
    country_data_df = pd.read_csv('country_data.csv')
    health_impact_df = pd.read_csv('health_impact.csv')

    # Add ISO-3 country codes for robust map plotting
    gapminder_df = px.data.gapminder().query("year==2007")
    country_iso_map = gapminder_df.set_index('country')['iso_alpha'].to_dict()

    # Manual mapping for names that differ between datasets
    manual_name_map = {
        "Congo, Dem. Rep.": "Democratic Republic of Congo",
        "Congo, Rep.": "Congo, Rep.",
        "Czechia": "Czech Republic",
        "Egypt, Arab Rep.": "Egypt",
        "Hong Kong SAR, China": "Hong Kong, China",
        "Iran, Islamic Rep.": "Iran",
        "Korea, Rep.": "Korea, Rep.",
        "Russian Federation": "Russia",
        "Slovak Republic": "Slovak Republic",
        "Taiwan, China": "Taiwan",
        "United Kingdom": "United Kingdom",
        "United States": "United States",
        "Venezuela, RB": "Venezuela",
        "Yemen, Rep.": "Yemen, Rep."
    }

    country_data_df['country_for_iso'] = country_data_df['Country'].replace(manual_name_map)
    country_data_df['ISO-3'] = country_data_df['country_for_iso'].map(country_iso_map)

    # Data Cleaning and Preparation
    # For simplicity, we'll take the lower end of the patient population range
    for col in country_data_df.columns:
        if 'Patient Population Range' in col:
            country_data_df[col.replace(' Range', '')] = country_data_df[col].str.split(' - ').str[0].str.replace(',',
                                                                                                                  '').str.replace(
                '< ', '').astype(float)

    # Convert 'B', 'T', 'M' to numeric values
    for col in ['GDP (Current US$)', 'Total Health Expenditure (Current US$)']:
        country_data_df[col] = country_data_df[col].replace({'B': '*1e9', 'T': '*1e12', 'M': '*1e6'}, regex=True).map(
            pd.eval).astype(float)

    country_data_df['GDP per Capita'] = country_data_df['GDP (Current US$)'] / (
                country_data_df['Insulin - Est. Patient Population'] * 10)  # Simplified population estimate

    # --- Calculations ---
    # Get anchor country's GDP per capita
    anchor_gdp_per_capita = country_data_df[country_data_df['Country'] == anchor_country]['GDP per Capita'].iloc[0]

    # Calculate GDP-adjusted price
    country_data_df['GDP Adjusted Price'] = (country_data_df['GDP per Capita'] / anchor_gdp_per_capita) * anchor_price

    # Price Capping Logic
    if cap_price_flag:
        cap_value = country_data_df['Total Health Expenditure (Current US$)'] * cap_percentage
        patient_col = f'{drug_name} - Est. Annual Patient Population'
        total_cost_gdp = country_data_df['GDP Adjusted Price'] * country_data_df[patient_col]

        # Avoid division by zero for countries with no patients
        country_data_df['Capped Price'] = np.where(
            (total_cost_gdp > cap_value) & (country_data_df[patient_col] > 0),
            cap_value / country_data_df[patient_col],
            country_data_df['GDP Adjusted Price']
        )
        country_data_df['Final Price'] = country_data_df['Capped Price']
    else:
        country_data_df['Final Price'] = country_data_df['GDP Adjusted Price']

    # --- Financial Analysis ---
    drug_cost = float(
        drug_costs_df[drug_costs_df['Drug'] == drug_name]['Base Cost'].iloc[0].split('/')[0].replace('$', ''))

    # Current System
    country_data_df['Current Revenue'] = country_data_df[f'Current {drug_name} Price'] * country_data_df[
        f'{drug_name} - Est. Annual Patient Population']
    country_data_df['Current Profit'] = (country_data_df[f'Current {drug_name} Price'] - drug_cost) * country_data_df[
        f'{drug_name} - Est. Annual Patient Population']

    # New System
    country_data_df['New Revenue'] = country_data_df['Final Price'] * country_data_df[
        f'{drug_name} - Est. Annual Patient Population']
    country_data_df['New Profit'] = (country_data_df['Final Price'] - drug_cost) * country_data_df[
        f'{drug_name} - Est. Annual Patient Population']

    # --- Visualization ---
    # Revenue and Profit Comparison
    summary_df = country_data_df[
        ['Country', 'Current Revenue', 'Current Profit', 'New Revenue', 'New Profit']].dropna().groupby(
        'Country').sum().reset_index()

    fig_revenue = go.Figure(data=[
        go.Bar(name='Current Revenue', x=summary_df['Country'], y=summary_df['Current Revenue']),
        go.Bar(name='New Revenue (GDP Adjusted)', x=summary_df['Country'], y=summary_df['New Revenue'])
    ])
    fig_revenue.update_layout(barmode='group',
                              title_text=f'Revenue Comparison for {drug_name} (Anchor: {anchor_country})')
    fig_revenue.write_html('revenue_comparison.html')

    fig_profit = go.Figure(data=[
        go.Bar(name='Current Profit', x=summary_df['Country'], y=summary_df['Current Profit']),
        go.Bar(name='New Profit (GDP Adjusted)', x=summary_df['Country'], y=summary_df['New Profit'])
    ])
    fig_profit.update_layout(barmode='group',
                             title_text=f'Profit Comparison for {drug_name} (Anchor: {anchor_country})')
    fig_profit.write_html('profit_comparison.html')

    # World Map Visualizations
    country_data_df['Price Change (%)'] = ((country_data_df['Final Price'] - country_data_df[
        f'Current {drug_name} Price']) / country_data_df[f'Current {drug_name} Price']) * 100

    fig_map_price = px.choropleth(country_data_df, locations="ISO-3", locationmode='ISO-3', color="Price Change (%)",
                                  hover_name="Country",
                                  title=f"Relative Price Change for {drug_name} (Anchor: {anchor_country})",
                                  color_continuous_scale=px.colors.sequential.Plasma)
    fig_map_price.write_html('price_change_map.html')

    fig_map_profit = px.choropleth(country_data_df, locations="ISO-3", locationmode='ISO-3', color="New Profit",
                                   hover_name="Country",
                                   title=f"Net Profit by Country for {drug_name} (Anchor: {anchor_country})",
                                   color_continuous_scale=px.colors.sequential.Viridis)
    fig_map_profit.write_html('net_profit_map.html')

    # --- Health Impact Visualization ---
    health_impact_df['QALYs (Upper Bound)'] = health_impact_df['Healthy Years Gained (QALYs)'].apply(clean_qaly_data)

    fig_qaly = px.bar(health_impact_df, x='Drug', y='QALYs (Upper Bound)',
                      title='Potential Healthy Years Gained (QALYs) per Patient')
    fig_qaly.write_html('qaly_comparison.html')

    # Create a composite health impact score
    drug_impact_score = health_impact_df.set_index('Drug')['QALYs (Upper Bound)'].to_dict()
    country_data_df['Health Impact Score'] = country_data_df[
                                                 f'{drug_name} - Est. Annual Patient Population'] * drug_impact_score.get(
        drug_name, 0)

    fig_map_health_impact = px.choropleth(country_data_df, locations="ISO-3", locationmode='ISO-3',
                                          color="Health Impact Score",
                                          hover_name="Country",
                                          title=f"Potential Health Impact (QALYs) for {drug_name} by Country",
                                          color_continuous_scale=px.colors.sequential.Magma)
    fig_map_health_impact.write_html('health_impact_map.html')

    # --- CSV Output ---
    country_data_df.to_csv('analysis_results.csv', index=False)
    health_impact_df.to_csv('health_impact_summary.csv', index=False)

    print("Analysis complete. Results saved to CSV files and interactive charts saved to HTML files.")


if __name__ == '__main__':
    # --- Control Section ---
    # Change these parameters to run different scenarios
    DRUG_TO_ANALYZE = 'Keytruda'  # Options: 'Keytruda', 'Ozempic', 'Insulin', 'Sovaldi'
    ANCHOR_COUNTRY = 'China'  # Options: 'United States', 'United Kingdom', 'China'

    # Use the anchor country's current price as the anchor price
    country_data_for_price = pd.read_csv('country_data.csv')
    anchor_price_val = country_data_for_price[country_data_for_price['Country'] == ANCHOR_COUNTRY][
        f'Current {DRUG_TO_ANALYZE} Price'].iloc[0]

    ANCHOR_PRICE = anchor_price_val

    CAP_PRICE = True  # Set to True to cap the price based on healthcare budget
    HEALTHCARE_BUDGET_CAP_PERCENTAGE = 0.10  # 10% of the healthcare budget

    model_drug_pricing(DRUG_TO_ANALYZE, ANCHOR_COUNTRY, ANCHOR_PRICE, CAP_PRICE, HEALTHCARE_BUDGET_CAP_PERCENTAGE)

