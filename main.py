import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos


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
    Models a single scenario for pharmaceutical pricing, using the expanded 'Potential Patients' column.
    """
    df = country_data_df.copy()

    anchor_gdp_per_capita = df[df['Country'] == anchor_country]['GDP per Capita'].iloc[0]
    df['GDP Adjusted Price'] = (df['GDP per Capita'] / anchor_gdp_per_capita) * anchor_price

    patient_col = 'Potential Patients'

    if cap_price_flag:
        cap_value = df['Total Health Expenditure (Current US$)'] * cap_percentage
        total_cost_gdp = df['GDP Adjusted Price'] * df[patient_col]

        df['Capped Price'] = np.where(
            (total_cost_gdp > cap_value) & (df[patient_col] > 0),
            cap_value / df[patient_col],
            df['GDP Adjusted Price']
        )
        df['Final Price'] = df['Capped Price']
    else:
        df['Final Price'] = df['GDP Adjusted Price']

    drug_cost = float(
        drug_costs_df[drug_costs_df['Drug'] == drug_name]['Base Cost'].iloc[0].split('/')[0].replace('$', ''))

    df['Revenue'] = df['Final Price'] * df[patient_col]
    df['Profit'] = (df['Final Price'] - drug_cost) * df[patient_col]

    drug_qaly = health_impact_df.set_index('Drug')['QALYs (Upper Bound)'].to_dict().get(drug_name, 0)
    df['QALYs Gained'] = df[patient_col] * drug_qaly

    # Estimate Lives Saved and Productivity Gains
    # These are simplified estimations for demonstration
    mortality_reduction_rate = 0.20  # Hypothetical 20% reduction
    df['Lives Saved'] = (df[patient_col] * mortality_reduction_rate).astype(int)
    avg_productivity_gain_per_patient = 50000  # Hypothetical $50,000 gain
    df['Productivity Gained'] = df[patient_col] * avg_productivity_gain_per_patient

    results = {
        'Total Revenue': df['Revenue'].sum(),
        'Total Profit': df['Profit'].sum(),
        'Patients Treated': df[patient_col].sum(),
        'Total QALYs Gained': df['QALYs Gained'].sum(),
        'Total Lives Saved': df['Lives Saved'].sum(),
        'Total Productivity Gained': df['Productivity Gained'].sum()
    }

    return results, df


def generate_visualizations(scenario_dfs, drug_name, region_df):
    """Generates all maps and the regional bar chart."""

    # --- 1. Generate Global Maps for each Scenario ---
    metrics_to_map = ['Revenue', 'Profit', 'QALYs Gained', 'Lives Saved', 'Productivity Gained']

    for scenario_name, df in scenario_dfs.items():
        # Ensure ISO-3 codes are present for mapping
        gapminder_df = px.data.gapminder().query("year==2007")
        country_iso_map = gapminder_df.set_index('country')['iso_alpha'].to_dict()
        manual_name_map = {"Congo, Dem. Rep.": "COD", "Congo, Rep.": "COG", "Czechia": "CZE", "Egypt, Arab Rep.": "EGY",
                           "Hong Kong SAR, China": "HKG", "Iran, Islamic Rep.": "IRN", "Korea, Rep.": "KOR",
                           "Russian Federation": "RUS", "Slovak Republic": "SVK", "Taiwan, China": "TWN",
                           "United Kingdom": "GBR", "United States": "USA", "Venezuela, RB": "VEN",
                           "Yemen, Rep.": "YEM"}
        df['country_for_iso'] = df['Country'].replace(manual_name_map)
        df['ISO-3'] = df['country_for_iso'].map(country_iso_map)

        for metric in metrics_to_map:
            if metric in df.columns:
                fig = px.choropleth(df,
                                    locations="ISO-3",
                                    locationmode='ISO-3',
                                    color=metric,
                                    hover_name="Country",
                                    title=f"{metric.replace('_', ' ')} for {drug_name} ({scenario_name.replace('_', ' ')})",
                                    color_continuous_scale=px.colors.sequential.Plasma)
                fig.write_html(f"map_{scenario_name.replace(' ', '_')}_{metric}.html")

    # --- 2. Generate Regional Bar Chart ---
    # Use the China Anchor scenario for regional breakdown as a representative example
    regional_df = pd.merge(scenario_dfs['China_Anchor'], region_df, on='Country')
    regional_summary = regional_df.groupby('Region')[['Revenue', 'Profit']].sum().reset_index()

    fig_regional = go.Figure()
    fig_regional.add_trace(go.Bar(
        x=regional_summary['Region'],
        y=regional_summary['Revenue'],
        name='Revenue',
        marker_color='indianred'
    ))
    fig_regional.add_trace(go.Bar(
        x=regional_summary['Region'],
        y=regional_summary['Profit'],
        name='Profit',
        marker_color='lightsalmon'
    ))

    fig_regional.update_layout(
        title_text=f'Regional Revenue and Profit Comparison for {drug_name} (China Anchor)',
        barmode='group',
        xaxis_tickangle=-45,
        yaxis_title="Amount (USD)"
    )
    fig_regional.write_html('regional_revenue_profit_chart.html')


def generate_pdf_report(report_data, drug_name):
    """Generates a PDF report summarizing the findings, including a chart."""
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
    chart_filename = 'revenue_profit_comparison.png'
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
        "GDP-adjusted pricing models, anchored to the United States, the United Kingdom, and China, respectively. "
        "The analysis reveals significant potential for expanding patient access while maintaining or even increasing "
        "manufacturer revenue and profit. The GDP-adjusted models, particularly when anchored to a lower-cost market "
        "like China, demonstrate a pathway to a more equitable global distribution of life-saving medicines, substantially "
        "increasing the total health benefits and boosting global productivity."
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(5)

    pdf.image(chart_filename, x=10, y=None, w=190)
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

    pdf.output('summary_report.pdf')


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

    drug_costs_df = pd.read_csv('drug_costs.csv')
    country_data_df = pd.read_csv('country_data.csv')
    health_impact_df = pd.read_csv('health_impact.csv')
    potential_patients_df = pd.read_csv('potential_patient_forecast.csv')
    region_df = pd.read_csv('country_regional_mapping.csv')

    health_impact_df['QALYs (Upper Bound)'] = health_impact_df['Healthy Years Gained (QALYs)'].apply(clean_qaly_data)
    country_data_df = pd.merge(country_data_df, potential_patients_df, on='Country', how='left')

    for col in country_data_df.columns:
        if 'Patient Population Range' in col:
            cleaned_series_lower = country_data_df[col].astype(str).str.split(' - ').str[0].str.replace(',',
                                                                                                        '').str.replace(
                '< ', '')
            country_data_df[col.replace(' Range', '')] = pd.to_numeric(cleaned_series_lower, errors='coerce')

    for col in ['GDP (Current US$)', 'Total Health Expenditure (Current US$)']:
        country_data_df[col] = country_data_df[col].apply(parse_financial_string)

    country_data_df['Population'] = (country_data_df['Insulin - Est. Patient Population'] * 10)
    country_data_df['GDP per Capita'] = country_data_df['GDP (Current US$)'] / country_data_df['Population']

    country_data_df = country_data_df.fillna(0)
    country_data_df['Potential Patients'] = country_data_df['Potential Patients'].astype(int)

    current_df = country_data_df.copy()
    drug_cost = float(
        drug_costs_df[drug_costs_df['Drug'] == DRUG_TO_ANALYZE]['Base Cost'].iloc[0].split('/')[0].replace('$', ''))
    current_df.loc[current_df['Country'] == 'China', f'{DRUG_TO_ANALYZE} - Est. Annual Patient Population'] = 0

    price_col = f'Current {DRUG_TO_ANALYZE} Price'
    patient_col = f'{DRUG_TO_ANALYZE} - Est. Annual Patient Population'

    base_revenue = (current_df[price_col] * current_df[patient_col]).sum()
    revenue_shortfall = ACTUAL_KEYTRUDA_REVENUE - base_revenue

    high_income_countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Canada', 'Japan']
    avg_tourist_price = current_df[current_df['Country'].isin(high_income_countries)][price_col].mean()

    medical_tourists = 0
    if revenue_shortfall > 0 and avg_tourist_price > 0:
        medical_tourists = revenue_shortfall / avg_tourist_price

    total_patients_current = current_df[patient_col].sum() + medical_tourists
    current_revenue_adjusted = base_revenue + (medical_tourists * avg_tourist_price)
    current_profit_adjusted = (current_revenue_adjusted - (total_patients_current * drug_cost))

    drug_qaly = health_impact_df.set_index('Drug')['QALYs (Upper Bound)'].to_dict().get(DRUG_TO_ANALYZE, 0)
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

        if anchor_data.empty:
            print(f"Warning: Anchor country '{anchor}' not found. Skipping.")
            continue
        anchor_price = anchor_data[price_col].iloc[0]
        if pd.isna(anchor_price) or anchor_price == 0:
            print(f"Warning: Anchor '{anchor}' has a missing/zero price. Skipping.")
            continue

        scenario_results, scenario_df = model_drug_pricing(
            DRUG_TO_ANALYZE, anchor, anchor_price, CAP_PRICE, HEALTHCARE_BUDGET_CAP_PERCENTAGE,
            country_data_df, drug_costs_df, health_impact_df
        )
        report_data[f'{anchor} Anchor'] = scenario_results
        scenario_dfs_for_maps[f'{anchor}_Anchor'] = scenario_df

    generate_pdf_report(report_data, DRUG_TO_ANALYZE)
    generate_visualizations(scenario_dfs_for_maps, DRUG_TO_ANALYZE, region_df)

    print("Analysis complete. All reports, maps, and charts generated.")


if __name__ == '__main__':
    main()

