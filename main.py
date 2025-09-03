import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def model_drug_pricing(drug_name, anchor_country, anchor_price, cap_price_flag, cap_percentage):
    """
    Models and contrasts pharmaceutical pricing structures.
    """
    # Load data
    drug_costs_df = pd.read_csv('drug_costs.csv')
    country_data_df = pd.read_csv('country_data.csv')
    health_impact_df = pd.read_csv('health_impact.csv')

    # Data Cleaning and Preparation
    # For simplicity, we'll take the lower end of the patient population range
    for col in country_data_df.columns:
        if 'Patient Population Range' in col:
            country_data_df[col.replace(' Range', '')] = country_data_df[col].str.split(' - ').str[0].str.replace(',',
                                                                                                                  '').str.replace(
                '< ', '').astype(float)

    # Convert 'B' and 'T' to numeric values
    for col in ['GDP (Current US$)', 'Total Health Expenditure (Current US$)']:
        country_data_df[col] = country_data_df[col].replace({'B': '*1e9', 'T': '*1e12', 'M': '*1e6'}, regex=True).map(
            pd.eval).astype(float)

    country_data_df['GDP per Capita'] = country_data_df['GDP (Current US$)'] / (
                country_data_df['Insulin - Est. Annual Patient Population'] * 10)  # Simplified population estimate

    # --- Calculations ---
    # Get anchor country's GDP per capita
    anchor_gdp_per_capita = country_data_df[country_data_df['Country'] == anchor_country]['GDP per Capita'].iloc[0]

    # Calculate GDP-adjusted price
    country_data_df['GDP Adjusted Price'] = (country_data_df['GDP per Capita'] / anchor_gdp_per_capita) * anchor_price

    # Price Capping Logic
    if cap_price_flag:
        cap_value = country_data_df['Total Health Expenditure (Current US$)'] * cap_percentage
        total_cost_gdp = country_data_df['GDP Adjusted Price'] * country_data_df[
            f'{drug_name} - Est. Annual Patient Population']

        country_data_df['Capped Price'] = np.where(
            total_cost_gdp > cap_value,
            cap_value / country_data_df[f'{drug_name} - Est. Annual Patient Population'],
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
    fig_revenue.show()

    fig_profit = go.Figure(data=[
        go.Bar(name='Current Profit', x=summary_df['Country'], y=summary_df['Current Profit']),
        go.Bar(name='New Profit (GDP Adjusted)', x=summary_df['Country'], y=summary_df['New Profit'])
    ])
    fig_profit.update_layout(barmode='group',
                             title_text=f'Profit Comparison for {drug_name} (Anchor: {anchor_country})')
    fig_profit.show()

    # World Map Visualizations
    country_data_df['Price Change (%)'] = ((country_data_df['Final Price'] - country_data_df[
        f'Current {drug_name} Price']) / country_data_df[f'Current {drug_name} Price']) * 100

    fig_map_price = px.choropleth(country_data_df, locations="Country", locationmode='country names',
                                  color="Price Change (%)",
                                  hover_name="Country",
                                  title=f"Relative Price Change for {drug_name} (Anchor: {anchor_country})",
                                  color_continuous_scale=px.colors.sequential.Plasma)
    fig_map_price.show()

    fig_map_profit = px.choropleth(country_data_df, locations="Country", locationmode='country names',
                                   color="New Profit",
                                   hover_name="Country",
                                   title=f"Net Profit by Country for {drug_name} (Anchor: {anchor_country})",
                                   color_continuous_scale=px.colors.sequential.Viridis)
    fig_map_profit.show()

    # --- Excel Output ---
    with pd.ExcelWriter('pharmaceutical_pricing_analysis.xlsx') as writer:
        country_data_df.to_excel(writer, sheet_name='Analysis_Results', index=False)
        health_impact_df.to_excel(writer, sheet_name='Health_Impact_Data', index=False)

    print("Analysis complete. Results saved to 'pharmaceutical_pricing_analysis.xlsx'")


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
