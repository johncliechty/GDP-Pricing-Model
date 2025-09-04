import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os


def clean_qaly_data(qaly_string):
    """Extracts the upper bound of QALYs from a string."""
    if isinstance(qaly_string, str):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", qaly_string)
        if numbers:
            return max(map(float, numbers))
    return np.nan


def model_drug_pricing(drug_name, anchor_country, anchor_price, cap_price_flag, cap_percentage, country_data_df,
                       drug_costs_df, health_impact_df):
    """
    Models a single scenario for pharmaceutical pricing, using refined patient adoption logic.
    Note: Assumes country_data_df already includes regional data.
    """
    df = country_data_df.copy()

    anchor_gdp_per_capita = df.loc[df['Country'] == anchor_country, 'GDP per Capita'].iloc[0]
    df['GDP Adjusted Price'] = (df['GDP per Capita'] / anchor_gdp_per_capita) * anchor_price

    # --- Refined Patient Adoption Logic ---
    developed_regions = ['North America', 'Europe', 'Japan', 'Australia/NZ']

    # Use current patients for developed regions, potential patients for others
    df['Scenario Patients'] = np.where(
        df['Region'].isin(developed_regions),
        df[f'{drug_name} - Est. Annual Patient Population'],
        df['Potential Patients']
    )

    if cap_price_flag:
        cap_value = df['Total Health Expenditure (Current US$)'] * cap_percentage
        total_cost_gdp = df['GDP Adjusted Price'] * df['Scenario Patients']

        df['Capped Price'] = np.where(
            (total_cost_gdp > cap_value) & (df['Scenario Patients'] > 0),
            cap_value / df['Scenario Patients'],
            df['GDP Adjusted Price']
        )
        df['Final Price'] = df['Capped Price']
    else:
        df['Final Price'] = df['GDP Adjusted Price']

    drug_cost = float(
        drug_costs_df.loc[drug_costs_df['Drug'] == drug_name, 'Base Cost'].iloc[0].split('/')[0].replace('$', ''))

    df['Revenue'] = df['Final Price'] * df['Scenario Patients']
    df['Profit'] = (df['Final Price'] - drug_cost) * df['Scenario Patients']

    drug_qaly = health_impact_df.set_index('Drug')['QALYs (Upper Bound)'].get(drug_name, 0)
    df['QALYs Gained'] = df['Scenario Patients'] * drug_qaly

    mortality_reduction_rate = 0.20
    df['Lives Saved'] = (df['Scenario Patients'] * mortality_reduction_rate).astype(int)
    avg_productivity_gain_per_patient = 50000
    df['Productivity Gained'] = df['Scenario Patients'] * avg_productivity_gain_per_patient

    results = {
        'Total Revenue': df['Revenue'].sum(),
        'Total Profit': df['Profit'].sum(),
        'Patients Treated': df['Scenario Patients'].sum(),
        'Total QALYs Gained': df['QALYs Gained'].sum(),
        'Total Lives Saved': df['Lives Saved'].sum(),
        'Total Productivity Gained': df['Productivity Gained'].sum()
    }

    return results, df


def generate_visualizations(scenario_dfs, drug_name, output_folder):
    """Generates all maps and the regional bar charts, saving them to the output folder."""

    metrics_to_map = ['Revenue', 'Profit', 'QALYs Gained', 'Lives Saved', 'Productivity Gained']
    for scenario_name, df in scenario_dfs.items():
        # Ensure ISO-3 codes are present for mapping
        gapminder_df = px.data.gapminder().query("year==2007")
        country_iso_map = gapminder_df.set_index('country')['iso_alpha'].to_dict()
        manual_name_map = {
            "Bolivia": "BOL", "Brunei Darussalam": "BRN", "Congo, Dem. Rep.": "COD",
            "Congo, Rep.": "COG", "Czechia": "CZE", "Egypt, Arab Rep.": "EGY",
            "Hong Kong SAR, China": "HKG", "Iran, Islamic Rep.": "IRN", "Korea, Rep.": "KOR",
            "Lao PDR": "LAO", "Macao SAR, China": "MAC", "Moldova": "MDA",
            "Russian Federation": "RUS", "Slovak Republic": "SVK", "Tanzania": "TZA",
            "Taiwan, China": "TWN", "United Kingdom": "GBR", "United States": "USA",
            "Venezuela, RB": "VEN", "Vietnam": "VNM", "Yemen, Rep.": "YEM"
        }
        df['country_for_iso'] = df['Country'].replace(manual_name_map)
        df['ISO-3'] = df['country_for_iso'].map(country_iso_map)

        for metric in metrics_to_map:
            if metric in df.columns:
                fig = px.choropleth(df, locations="ISO-3", locationmode='ISO-3', color=metric,
                                    hover_name="Country",
                                    title=f"{metric.replace('_', ' ')} for {drug_name} ({scenario_name.replace('_', ' ')})",
                                    color_continuous_scale=px.colors.sequential.Plasma)
                fig.write_html(os.path.join(output_folder, f"map_{scenario_name.replace(' ', '_')}_{metric}.html"))

    # --- Regional Bar Charts ---
    regional_revenue_summary = pd.DataFrame({
        scenario: df.groupby('Region')['Revenue'].sum() for scenario, df in scenario_dfs.items()
    }).reset_index()

    fig_reg_rev = px.bar(regional_revenue_summary, x='Region', y=list(scenario_dfs.keys()),
                         title=f'Regional Revenue Comparison for {drug_name}', barmode='group',
                         labels={'value': 'Total Revenue (USD)', 'variable': 'Scenario'})
    fig_reg_rev.write_html(os.path.join(output_folder, 'regional_revenue_comparison_chart.html'))

    # --- Regional Price Comparison Chart ---
    price_col = f'Current {drug_name} Price'
    regional_price_data = []
    for scenario_name, df in scenario_dfs.items():
        if scenario_name == 'Current_System':
            price_column = price_col
            patient_column = f'{drug_name} - Est. Annual Patient Population'
        else:
            price_column = 'Final Price'
            patient_column = 'Scenario Patients'

        def weighted_avg(group, avg_name, weight_name):
            d = group[avg_name]
            w = group[weight_name]
            if w.sum() == 0:
                return 0
            return (d * w).sum() / w.sum()

        regional_prices = df.groupby('Region').apply(weighted_avg, price_column, patient_column,
                                                     include_groups=False).reset_index(name='Avg Price')
        regional_prices['Scenario'] = scenario_name.replace('_', ' ')
        regional_price_data.append(regional_prices)

    regional_price_summary = pd.concat(regional_price_data)

    fig_reg_price = px.bar(regional_price_summary, x='Region', y='Avg Price', color='Scenario',
                           title=f'Regional Price Comparison for {drug_name}', barmode='group',
                           labels={'Avg Price': 'Average Price per Patient (USD)'})
    fig_reg_price.write_html(os.path.join(output_folder, 'regional_price_comparison_chart.html'))


def generate_pdf_report(report_data, drug_name, assumptions_df, output_folder):
    """Generates a PDF report summarizing the findings, saving it to the output folder."""
    labels = list(report_data.keys())
    revenues = [report_data[label]['Total Revenue'] / 1e9 for label in labels]
    profits = [report_data[label]['Total Profit'] / 1e9 for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, revenues, width, label='Revenue', color='royalblue')
    rects2 = ax.bar(x + width / 2, profits, width, label='Profit', color='lightgreen')

    ax.set_ylabel('Amount (in Billions USD)')
    ax.set_title(f'Comparative Revenue and Profit for {drug_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='${:,.1f}B')
    ax.bar_label(rects2, padding=3, fmt='${:,.1f}B')
    fig.tight_layout()
    chart_filename = os.path.join(output_folder, 'revenue_profit_comparison.png')
    plt.savefig(chart_filename)
    plt.close()

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, f'Pharmaceutical Pricing Model: Comparative Analysis for {drug_name}', new_x=XPos.LMARGIN,
             new_y=YPos.NEXT, align='C')
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Executive Summary', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    summary_text = (
        "This report contrasts the established global pharmaceutical pricing structure with three alternative "
        "GDP-adjusted pricing models. The analysis reveals significant potential for expanding patient access while "
        "maintaining or increasing manufacturer revenue. GDP-adjusted models demonstrate a pathway to a more equitable "
        "global distribution of life-saving medicines, substantially increasing total health benefits (QALYs) and "
        "boosting global productivity."
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(5)

    pdf.image(chart_filename, x=10, y=None, w=190)
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Key Assumptions and Model Logic', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    assumptions_text = (
        "The model is based on several key assumptions:\n"
        "- **Current System:** Revenue is calculated from countries with known prices. A 'Medical Tourism' component is added to match Merck's reported global revenue of $29.5B. This component is now smaller due to more countries having listed prices.\n"
        "- **GDP-Adjusted Models:** Price is adjusted based on GDP per capita relative to an anchor country. These models assume expanded access.\n"
        "- **Patient Adoption:** For GDP models, patient numbers for developed nations are held at current levels, while Low- and Middle-Income Countries expand to their 'Potential Patient' forecast, reflecting where new access occurs.\n"
        "- **US Revenue in China-Anchored Model:** The high US revenue (~$20B) in this scenario is a direct result of the formula, highlighting that non-price factors also limit access.\n"
        "- **Pricing Data:** Sourced from provided documents and external research (see country_price_assumptions.csv)."
    )
    pdf.multi_cell(0, 5, assumptions_text)
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Scenario Comparison', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font('Helvetica', 'B', 10)

    num_cols = len(labels) + 1
    col_width = 190 / num_cols

    pdf.cell(col_width, 10, 'Metric', 1)
    for label in labels:
        pdf.cell(col_width, 10, label, 1)
    pdf.ln()

    pdf.set_font('Helvetica', '', 9)
    metrics_for_report = ['Total Revenue', 'Total Profit', 'Patients Treated', 'Total QALYs Gained',
                          'Total Lives Saved', 'Total Productivity Gained']
    for metric in metrics_for_report:
        pdf.cell(col_width, 10, metric.replace('Total ', '').replace(' Gained', ''), 1)

        for label in labels:
            val = report_data[label][metric]

            if 'Revenue' in metric or 'Profit' in metric or 'Productivity' in metric:
                formatted_val = f"${val / 1e9:,.2f}B"
                pdf.cell(col_width, 10, formatted_val, 1)
            else:
                formatted_val = f"{int(val):,}"
                pdf.cell(col_width, 10, formatted_val, 1)
        pdf.ln()

    pdf.output(os.path.join(output_folder, 'summary_report.pdf'))


def parse_financial_string(value):
    value = str(value).strip().lower()
    if pd.isna(value) or value in ['nan', '']:
        return 0.0

    multiplier = 1
    if 't' in value:
        multiplier = 1e12
        value = value.replace('t', '')
    elif 'b' in value:
        multiplier = 1e9
        value = value.replace('b', '')
    elif 'm' in value:
        multiplier = 1e6
        value = value.replace('m', '')

    try:
        return float(value.replace(',', '')) * multiplier
    except (ValueError, TypeError):
        return 0.0


def main():
    DRUG_TO_ANALYZE = 'Keytruda'
    CAP_PRICE = True
    HEALTHCARE_BUDGET_CAP_PERCENTAGE = 0.10
    ACTUAL_KEYTRUDA_REVENUE = 29.5 * 1e9
    OUTPUT_FOLDER = 'output'

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    drug_costs_df = pd.read_csv('drug_costs.csv')
    country_data_df = pd.read_csv('country_data.csv')
    health_impact_df = pd.read_csv('health_impact.csv')
    potential_patients_df = pd.read_csv('potential_patient_forecast.csv')
    region_df = pd.read_csv('country_regional_mapping.csv')
    assumptions_df = pd.read_csv('country_price_assumptions.csv')

    health_impact_df['QALYs (Upper Bound)'] = health_impact_df['Healthy Years Gained (QALYs)'].apply(clean_qaly_data)

    # --- Merge all dataframes early ---
    country_data_df = pd.merge(country_data_df, potential_patients_df, on='Country', how='left')
    country_data_df = pd.merge(country_data_df, region_df, on='Country', how='left')

    for col in country_data_df.columns:
        if 'Patient Population Range' in col:
            cleaned_series_lower = country_data_df[col].astype(str).str.split(' - ').str[0].str.replace(',',
                                                                                                        '').str.replace(
                '< ', '')
            country_data_df[col.replace(' Range', '')] = pd.to_numeric(cleaned_series_lower, errors='coerce')

    for col in ['GDP (Current US$)', 'Total Health Expenditure (Current US$)']:
        country_data_df[col] = country_data_df[col].apply(parse_financial_string)

    country_data_df['Population'] = (country_data_df['Insulin - Est. Patient Population'] * 10).replace(0, np.nan)
    country_data_df['GDP per Capita'] = country_data_df['GDP (Current US$)'] / country_data_df['Population']

    country_data_df = country_data_df.fillna(0)
    country_data_df['Potential Patients'] = country_data_df['Potential Patients'].astype(int)

    current_df = country_data_df.copy()
    drug_cost = float(
        drug_costs_df[drug_costs_df['Drug'] == DRUG_TO_ANALYZE]['Base Cost'].iloc[0].split('/')[0].replace('$', ''))

    price_col = f'Current {DRUG_TO_ANALYZE} Price'
    patient_col = f'{DRUG_TO_ANALYZE} - Est. Annual Patient Population'
    current_df.loc[current_df['Country'] == 'China', patient_col] = 0

    base_revenue = (current_df[price_col] * current_df[patient_col]).sum()
    revenue_shortfall = max(0, ACTUAL_KEYTRUDA_REVENUE - base_revenue)

    high_income_countries_for_tourism = ['United States', 'Germany', 'United Kingdom', 'France', 'Canada', 'Japan',
                                         'Switzerland']
    avg_tourist_price = current_df[current_df['Country'].isin(high_income_countries_for_tourism)][price_col].mean()

    medical_tourists = 0
    if revenue_shortfall > 0 and avg_tourist_price > 0:
        medical_tourists = revenue_shortfall / avg_tourist_price

    total_patients_current = current_df[patient_col].sum() + medical_tourists
    current_revenue_adjusted = base_revenue + (medical_tourists * avg_tourist_price)
    current_profit_adjusted = (current_revenue_adjusted - (total_patients_current * drug_cost))

    drug_qaly = health_impact_df.set_index('Drug')['QALYs (Upper Bound)'].get(DRUG_TO_ANALYZE, 0)
    current_qaly = total_patients_current * drug_qaly

    mortality_reduction_rate = 0.20
    current_lives_saved = total_patients_current * mortality_reduction_rate
    avg_productivity_gain_per_patient = 50000
    current_productivity_gained = total_patients_current * avg_productivity_gain_per_patient

    report_data = {
        'Current System': {
            'Total Revenue': current_revenue_adjusted,
            'Total Profit': current_profit_adjusted,
            'Patients Treated': total_patients_current,
            'Total QALYs Gained': current_qaly,
            'Total Lives Saved': current_lives_saved,
            'Total Productivity Gained': current_productivity_gained
        }
    }

    anchor_countries = ['United States', 'United Kingdom', 'China']
    scenario_dfs_for_maps = {}

    current_system_map_df = current_df.copy()
    current_system_map_df['Revenue'] = current_system_map_df[price_col] * current_system_map_df[patient_col]
    current_system_map_df['Profit'] = (current_system_map_df[price_col] - drug_cost) * current_system_map_df[
        patient_col]
    current_system_map_df['QALYs Gained'] = current_system_map_df[patient_col] * drug_qaly
    current_system_map_df['Lives Saved'] = (current_system_map_df[patient_col] * mortality_reduction_rate).astype(int)
    current_system_map_df['Productivity Gained'] = current_system_map_df[
                                                       patient_col] * avg_productivity_gain_per_patient
    scenario_dfs_for_maps['Current_System'] = current_system_map_df

    for anchor in anchor_countries:
        anchor_data = country_data_df[country_data_df['Country'] == anchor]

        if anchor_data.empty or pd.isna(anchor_data[price_col].iloc[0]) or anchor_data[price_col].iloc[0] == 0:
            print(f"Warning: Anchor '{anchor}' not found or has invalid price. Skipping.")
            continue
        anchor_price = anchor_data[price_col].iloc[0]

        scenario_results, scenario_df = model_drug_pricing(
            DRUG_TO_ANALYZE, anchor, anchor_price, CAP_PRICE, HEALTHCARE_BUDGET_CAP_PERCENTAGE,
            country_data_df, drug_costs_df, health_impact_df
        )
        report_data[f'{anchor} Anchor'] = scenario_results
        scenario_dfs_for_maps[f'{anchor}_Anchor'] = scenario_df

    generate_pdf_report(report_data, DRUG_TO_ANALYZE, assumptions_df, OUTPUT_FOLDER)
    generate_visualizations(scenario_dfs_for_maps, DRUG_TO_ANALYZE, OUTPUT_FOLDER)

    print(f"Analysis complete. All reports, maps, and charts generated in '{OUTPUT_FOLDER}' folder.")


if __name__ == '__main__':
    main()

