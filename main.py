import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os
import warnings

# Suppress general FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration Constants ---
DRUG_TO_ANALYZE = 'Keytruda'
CAP_PRICE = True
HEALTHCARE_BUDGET_CAP_PERCENTAGE = 0.10
ACTUAL_KEYTRUDA_REVENUE = 29.5 * 1e9
OUTPUT_FOLDER = 'output_final_verified_analysis'

# Define countries to exclude from the Current System baseline (based on user premise)
COUNTRIES_TO_EXCLUDE_FROM_CURRENT_SYSTEM = ['China']

# World Bank Income Thresholds (FY25 Proxy)
INCOME_THRESHOLDS = {
    'Low income': (0, 1135),
    'Lower middle income': (1136, 4465),
    'Upper middle income': (4466, 13845),
    'High income': (13846, float('inf'))
}

# Countries for visualization labeling
KEY_COUNTRIES_TO_LABEL = ['United States', 'China', 'United Kingdom', 'Germany', 'India', 'Brazil', 'South Africa',
                          'Japan', 'Nigeria']
KEY_ECONOMIES_PRICE_CHART = ['United States', 'United Kingdom', 'France', 'Germany', 'China', 'Saudi Arabia',
                             'United Arab Emirates']

# Comprehensive ISO-3 Mapping (Ensures all countries are mapped)
COMPREHENSIVE_ISO_MAP = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "American Samoa": "ASM", "Andorra": "AND",
    "Angola": "AGO", "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM", "Aruba": "ABW",
    "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE", "Bahamas, The": "BHS", "Bahrain": "BHR",
    "Bangladesh": "BGD", "Barbados": "BRB", "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ",
    "Benin": "BEN", "Bermuda": "BMU", "Bhutan": "BTN", "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH",
    "Botswana": "BWA", "Brazil": "BRA", "British Virgin Islands": "VGB", "Brunei Darussalam": "BRN",
    "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI", "Cabo Verde": "CPV", "Cambodia": "KHM",
    "Cameroon": "CMR", "Canada": "CAN", "Cayman Islands": "CYM", "Central African Republic": "CAF",
    "Chad": "TCD", "Chile": "CHL", "China": "CHN", "Colombia": "COL", "Comoros": "COM",
    "Congo, Dem. Rep.": "COD", "Congo, Rep.": "COG", "Costa Rica": "CRI", "Cote d'Ivoire": "CIV",
    "Côte d'Ivoire": "CIV",
    "Croatia": "HRV", "Cuba": "CUB", "Curaçao": "CUW", "Cyprus": "CYP", "Czechia": "CZE", "Denmark": "DNK",
    "Djibouti": "DJI", "Dominica": "DMA", "Dominican Republic": "DOM", "Ecuador": "ECU", "Egypt, Arab Rep.": "EGY",
    "El Salvador": "SLV", "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Estonia": "EST", "Eswatini (Swaziland)": "SWZ",
    "Ethiopia": "ETH", "Faroe Islands": "FRO", "Fiji": "FJI", "Finland": "FIN", "France": "FRA",
    "French Polynesia": "PYF", "Gabon": "GAB", "Gambia, The": "GMB", "Georgia": "GEO", "Germany": "DEU",
    "Ghana": "GHA", "Gibraltar": "GIB", "Greece": "GRC", "Greenland": "GRL", "Grenada": "GRD", "Guam": "GUM",
    "Guatemala": "GTM", "Guinea": "GIN", "Guinea-Bissau": "GNB", "Guyana": "GUY", "Haiti": "HTI",
    "Honduras": "HND", "Hong Kong SAR, China": "HKG", "Hungary": "HUN", "Iceland": "ISL", "India": "IND",
    "Indonesia": "IDN", "Iran, Islamic Rep.": "IRN", "Iraq": "IRQ", "Ireland": "IRL", "Isle of Man": "IMN",
    "Israel": "ISR", "Italy": "ITA", "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Kiribati": "KIR", "Korea, Dem. People's Rep.": "PRK", "Korea, Rep.": "KOR", "Kosovo": "XKX",
    "Kuwait": "KWT", "Kyrgyz Republic": "KGZ", "Lao PDR": "LAO", "Latvia": "LVA", "Lebanon": "LBN",
    "Lesotho": "LSO", "Liberia": "LBR", "Libya": "LBY", "Liechtenstein": "LIE", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Macao SAR, China": "MAC", "Madagascar": "MDG", "Malawi": "MWI", "Malaysia": "MYS",
    "Maldives": "MDV", "Mali": "MLI", "Malta": "MLT", "Marshall Islands": "MHL", "Mauritania": "MRT",
    "Mauritius": "MUS", "Mexico": "MEX", "Micronesia, Fed. Sts.": "FSM", "Moldova": "MDA", "Monaco": "MCO",
    "Mongolia": "MNG", "Montenegro": "MNE", "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR",
    "Namibia": "NAM", "Nauru": "NRU", "Nepal": "NPL", "Netherlands": "NLD", "New Caledonia": "NCL",
    "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER", "Nigeria": "NGA", "North Macedonia": "MKD",
    "Northern Mariana Islands": "MNP", "Norway": "NOR", "Oman": "OMN", "Pakistan": "PAK", "Palau": "PLW",
    "Panama": "PAN", "Papua New Guinea": "PNG", "Paraguay": "PRY", "Peru": "PER", "Philippines": "PHL",
    "Poland": "POL", "Portugal": "PRT", "Puerto Rico": "PRI", "Qatar": "QAT", "Romania": "ROU",
    "Russian Federation": "RUS", "Rwanda": "RWA", "Samoa": "WSM", "San Marino": "SMR", "Sao Tome and Principe": "STP",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB", "Seychelles": "SYC", "Sierra Leone": "SLE",
    "Singapore": "SGP", "Sint Maarten (Dutch part)": "SXM", "Slovak Republic": "SVK", "Slovenia": "SVN",
    "Solomon Islands": "SLB", "Somalia": "SOM", "South Africa": "ZAF", "South Sudan": "SSD", "Spain": "ESP",
    "Sri Lanka": "LKA", "St. Kitts and Nevis": "KNA", "St. Lucia": "LCA", "St. Martin (French part)": "MAF",
    "St. Vincent and the Grenadines": "VCT", "Sudan": "SDN", "Suriname": "SUR", "Sweden": "SWE",
    "Switzerland": "CHE", "Syrian Arab Republic": "SYR", "Taiwan, China": "TWN", "Tajikistan": "TJK",
    "Tanzania": "TZA", "Thailand": "THA", "Timor-Leste": "TLS", "Togo": "TGO", "Tonga": "TON",
    "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Türkiye": "TUR", "Turkmenistan": "TKM",
    "Turks and Caicos Islands": "TCA", "Tuvalu": "TUV", "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "United Kingdom": "GBR", "United States": "USA", "Uruguay": "URY",
    "Uzbekistan": "UZB", "Vanuatu": "VUT", "Venezuela, RB": "VEN", "Vietnam": "VNM", "Virgin Islands (U.S.)": "VIR",
    "West Bank and Gaza": "PSE", "Yemen, Rep.": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE"
}


# --- Helper Functions ---

def get_drug_cost(drug_costs_df, drug_name):
    """Robustly extracts the Manufacturing Cost (COGS) for a given drug."""
    try:
        # Filter for the specific drug and the COGS metric
        cost_entry = drug_costs_df[(drug_costs_df['Drug'] == drug_name) &
                                   (drug_costs_df['Metric'] == 'Manufacturing Cost (COGS)')]
        if not cost_entry.empty:
            # Get the value and ensure it's numeric, handling potential non-numeric entries
            drug_cost = pd.to_numeric(cost_entry['Value (USD)'], errors='coerce').iloc[0]
            if pd.isna(drug_cost):
                return 0.0
            return float(drug_cost)
        else:
            return 0.0
    except Exception as e:
        print(f"Warning: Error retrieving cost for {drug_name}. Defaulting to 0. Error: {e}")
        return 0.0


def clean_qaly_data(qaly_string):
    if isinstance(qaly_string, str):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", qaly_string)
        if numbers:
            return max(map(float, numbers))
    return np.nan


def parse_financial_string(value):
    value = str(value).strip().lower()
    if pd.isna(value) or value in ['nan', '']:
        return 0.0
    multiplier = 1
    if 't' in value:
        multiplier = 1e12
    elif 'b' in value:
        multiplier = 1e9
    elif 'm' in value:
        multiplier = 1e6
    value = value.replace('t', '').replace('b', '').replace('m', '')
    try:
        return float(value.replace('$', '').replace(',', '')) * multiplier
    except (ValueError, TypeError):
        return 0.0


def classify_income(gdp_pc):
    for level, (low, high) in INCOME_THRESHOLDS.items():
        if low <= gdp_pc <= high:
            return level
    return 'Unknown'


# --- Symlog Transformation Functions ---
# (No changes needed here)

def symlog_transform(data, C):
    data = np.array(data, dtype=float)
    C_calc = max(C, 1e-9)
    return np.sign(data) * np.log10(1 + np.abs(data) / C_calc)


def format_tick_label(val, use_suffixes=True, is_percentage=False):
    """Formats tick labels cleanly for axes and color bars."""
    if abs(val) < 1e-9:
        return '0'

    # Handle percentage formatting
    if is_percentage:
        if abs(val) >= 10:
            return f'{val:+.0f}%'
        elif abs(val) >= 0.01:
            label = f'{val:+.2f}'
            return label.rstrip('0').rstrip('.') + '%'
        else:
            return f'{val:+.2g}%'

    # Standard formatting for numbers
    sign = '+' if val > 0 else ('-' if val < 0 else '')
    abs_val = abs(val)

    if use_suffixes:
        if abs_val >= 1e9:
            return f'{sign}{abs_val / 1e9:.1f}B'.replace('.0B', 'B')
        elif abs_val >= 1e6:
            return f'{sign}{abs_val / 1e6:.1f}M'.replace('.0M', 'M')
        elif abs_val >= 1e3:
            return f'{sign}{abs_val / 1e3:.1f}K'.replace('.0K', 'K')

    if abs_val >= 100:
        return f'{sign}{abs_val:.0f}'
    elif abs_val >= 1:
        return f'{sign}{abs_val:.1f}'.replace('.0', '')
    else:
        return f'{sign}{abs_val:.2g}'


def generate_symlog_ticks(min_val, max_val, C, use_suffixes=True, is_percentage=False):
    abs_max = max(abs(min_val), abs(max_val))
    if abs_max == 0:
        return [0], ['0']

    C_calc = max(C, 1e-9)

    # Ensure input to log10 is positive
    input_to_log = max(abs_max, C_calc)
    if input_to_log <= 0:
        return [0], ['0']

    max_magnitude = int(np.ceil(np.log10(input_to_log)))

    tick_vals_actual = [0]

    # Ensure C_calc is positive for log10
    if C_calc > 0:
        start_magnitude = int(np.floor(np.log10(C_calc)))
    else:
        start_magnitude = 0  # Should not happen due to C_calc = max(C, 1e-9)

    # Generate power-of-10 ticks
    for magnitude in range(start_magnitude, max_magnitude + 1):
        val = 10 ** magnitude
        if val <= max_val * 1.1:
            tick_vals_actual.append(val)
        if -val >= min_val * 1.1:
            tick_vals_actual.append(-val)

    transformed_ticks = symlog_transform(tick_vals_actual, C)
    tick_labels = [format_tick_label(val, use_suffixes, is_percentage) for val in tick_vals_actual]

    # Sort by the transformed tick value
    combined = sorted(zip(transformed_ticks, tick_labels), key=lambda x: x[0])
    transformed_ticks_sorted = [x[0] for x in combined]
    tick_labels_sorted = [x[1] for x in combined]

    return transformed_ticks_sorted, tick_labels_sorted


# --- Standardized Regional Mapping ---
# (No changes needed here)
def get_standardized_region(country):
    """Maps countries to standardized geographical regions."""
    mapping = {
        'East Asia & Pacific': ["Australia", "Brunei Darussalam", "Cambodia", "China", "Fiji", "Hong Kong SAR, China",
                                "Indonesia", "Japan", "Kiribati", "Korea, Dem. People's Rep.", "Korea, Rep.", "Lao PDR",
                                "Macao SAR, China", "Malaysia", "Marshall Islands", "Micronesia, Fed. Sts.", "Mongolia",
                                "Myanmar", "Nauru", "New Caledonia", "New Zealand", "Palau", "Papua New Guinea",
                                "Philippines", "Samoa", "Singapore", "Solomon Islands", "Taiwan, China", "Thailand",
                                "Timor-Leste", "Tonga", "Tuvalu", "Vanuatu", "Vietnam", "American Samoa",
                                "French Polynesia", "Guam", "Northern Mariana Islands"],
        'Europe & Central Asia': ["Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
                                  "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark",
                                  "Estonia", "Faroe Islands", "Finland", "France", "Georgia", "Germany", "Gibraltar",
                                  "Greece", "Hungary", "Iceland", "Ireland", "Isle of Man", "Italy", "Kazakhstan",
                                  "Kosovo", "Kyrgyz Republic", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg",
                                  "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
                                  "Norway", "Poland", "Portugal", "Romania", "Russian Federation", "San Marino",
                                  "Serbia", "Slovak Republic", "Slovenia", "Spain", "Sweden", "Switzerland",
                                  "Tajikistan", "Türkiye", "Turkmenistan", "Ukraine", "United Kingdom", "Uzbekistan"],
        'Latin America & Caribbean': ["Antigua and Barbuda", "Argentina", "Aruba", "Bahamas, The", "Barbados", "Belize",
                                      "Bolivia", "Brazil", "British Virgin Islands", "Cayman Islands", "Chile",
                                      "Colombia", "Costa Rica", "Cuba", "Curaçao", "Dominica", "Dominican Republic",
                                      "Ecuador", "El Salvador", "Grenada", "Guatemala", "Guyana", "Haiti", "Honduras",
                                      "Jamaica", "Mexico", "Nicaragua", "Panama", "Paraguay", "Peru", "Puerto Rico",
                                      "Sint Maarten (Dutch part)", "St. Kitts and Nevis", "St. Lucia",
                                      "St. Martin (French part)", "St. Vincent and the Grenadines", "Suriname",
                                      "Trinidad and Tobago", "Turks and Caicos Islands", "Uruguay", "Venezuela, RB",
                                      "Virgin Islands (U.S.)"],
        'Middle East & North Africa': ["Algeria", "Bahrain", "Djibouti", "Egypt, Arab Rep.", "Iran, Islamic Rep.",
                                       "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Libya", "Morocco", "Oman",
                                       "Qatar", "Saudi Arabia", "Syrian Arab Republic", "Tunisia",
                                       "United Arab Emirates", "West Bank and Gaza", "Yemen, Rep."],
        'North America': ["Bermuda", "Canada", "Greenland", "United States"],
        'South Asia': ["Afghanistan", "Bangladesh", "Bhutan", "India", "Maldives", "Nepal", "Pakistan", "Sri Lanka"],
        'Sub-Saharan Africa': ["Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon",
                               "Central African Republic", "Chad", "Comoros", "Congo, Dem. Rep.", "Congo, Rep.",
                               "Côte d'Ivoire", "Equatorial Guinea", "Eritrea", "Eswatini (Swaziland)", "Ethiopia",
                               "Gabon", "Gambia, The", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho",
                               "Liberia", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Mozambique",
                               "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal",
                               "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
                               "Tanzania", "Togo", "Uganda", "Zambia", "Zimbabwe"]
    }

    for region, countries in mapping.items():
        if country in countries:
            return region
    return 'Other'


# --- Core Modeling Function ---

def model_drug_pricing(drug_name, anchor_country, anchor_price, cap_price_flag, cap_percentage, country_data_df,
                       drug_costs_df, health_impact_df):
    """Models a single scenario for pharmaceutical pricing (Universal Acceptance)."""
    df = country_data_df.copy()
    price_col = f'Current {drug_name} Price'
    # Use the standardized column name
    current_patient_col = f'{drug_name} - Est. Annual Patient Population'

    # 1. Calculate GDP Adjusted Price
    try:
        anchor_gdp_per_capita = df.loc[df['Country'] == anchor_country, 'GDP per Capita'].iloc[0]
    except IndexError:
        # Handle case where anchor country might be missing from the dataset
        anchor_gdp_per_capita = 0

    if anchor_gdp_per_capita > 0:
        # Use the anchor_price passed into the function (the original price), not the price in the baseline DF.
        df['GDP Adjusted Price'] = (df['GDP per Capita'] / anchor_gdp_per_capita) * anchor_price
    else:
        df['GDP Adjusted Price'] = 0

    # 2. Patient Adoption Logic
    # 'Is Established Market' uses the baseline price (which is now 0 for China in country_data_df)
    df['Is Established Market'] = df[price_col] > 0

    # If established, use baseline patients. If not (now including China), use Potential Patients.
    df['Scenario Patients'] = np.where(
        df['Is Established Market'],
        df[current_patient_col],
        df['Potential Patients']
    )

    # 3. Price Determination & Budget Impact Capping
    if cap_price_flag:
        cap_value = df['Total Health Expenditure (Current US$)'] * cap_percentage
        total_cost_gdp = df['GDP Adjusted Price'] * df['Scenario Patients']

        df['Final Price'] = np.where(
            (total_cost_gdp > cap_value) & (df['Scenario Patients'] > 0) & (
                    df['Total Health Expenditure (Current US$)'] > 0),
            cap_value / df['Scenario Patients'],
            df['GDP Adjusted Price']
        )
    else:
        df['Final Price'] = df['GDP Adjusted Price']

    # 4. Revenue, Profit, Budget Impact, and Health Outcomes

    # CORRECTION: Use the helper function to get the drug cost based on COGS
    drug_cost = get_drug_cost(drug_costs_df, drug_name)

    df['Final Price'] = np.maximum(0, df['Final Price'])
    df['Revenue'] = df['Final Price'] * df['Scenario Patients']
    df['Profit'] = (df['Final Price'] - drug_cost) * df['Scenario Patients']

    df['Budget Impact (% THE)'] = df.apply(
        lambda row: (row['Revenue'] / row['Total Health Expenditure (Current US$)']) * 100 if row[
                                                                                                  'Total Health Expenditure (Current US$)'] > 0 else 0,
        axis=1
    )

    drug_qaly = health_impact_df.set_index('Drug')['QALYs (Upper Bound)'].get(drug_name, 0)
    df['QALYs Gained'] = df['Scenario Patients'] * drug_qaly

    mortality_reduction_rate = 0.20
    df['Lives Saved'] = (df['Scenario Patients'] * mortality_reduction_rate).fillna(0).astype(int)

    results = {
        'Total Revenue': df['Revenue'].sum(),
        'Total Profit': df['Profit'].sum(),
        'Patients Treated': df['Scenario Patients'].sum(),
        'Total QALYs Gained': df['QALYs Gained'].sum(),
        'Total Lives Saved': df['Lives Saved'].sum(),
    }

    return results, df


# --- Analysis and Visualization Functions ---
# (These functions are kept as they were in the original script, relying on the corrected data inputs)

def generate_budget_impact_summary(scenario_dfs):
    # (Implementation remains the same as original script)
    summary_data = []
    income_order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']

    for scenario_name, df in scenario_dfs.items():
        df_valid = df[df['Total Health Expenditure (Current US$)'] > 0]

        if 'Income Level' in df_valid.columns and 'Budget Impact (% THE)' in df_valid.columns:
            impact_by_income = df_valid.groupby('Income Level')['Budget Impact (% THE)'].agg(
                ['mean', 'max']).reset_index()

            for index, row in impact_by_income.iterrows():
                summary_data.append({
                    'Scenario': scenario_name.replace('_', ' '),
                    'Income Level': row['Income Level'],
                    'Average Impact (% THE)': row['mean'],
                    'Maximum Impact (% THE)': row['max']
                })

    summary_df = pd.DataFrame(summary_data)

    if not summary_df.empty:
        summary_df['Income Level'] = pd.Categorical(summary_df['Income Level'], categories=income_order, ordered=True)
        summary_df = summary_df.sort_values(by=['Scenario', 'Income Level'])

    return summary_df


def generate_key_country_price_chart(scenario_dfs, drug_name, country_gdp_order, output_folder):
    # (Implementation remains the same as original script)
    """Generates a bar chart comparing prices for key economies, sorted by GDP (Largest on Right)."""

    price_data = []
    current_price_col = f'Current {drug_name} Price'

    for scenario_name, df in scenario_dfs.items():
        scenario_display_name = scenario_name.replace('_', ' ')

        # Determine the price column
        if scenario_name == 'Current_System':
            price_column = current_price_col
        else:
            price_column = 'Final Price'

        if price_column not in df.columns: continue

        # Filter for key countries and extract prices
        key_country_prices = df[df['Country'].isin(KEY_ECONOMIES_PRICE_CHART)]

        for index, row in key_country_prices.iterrows():
            price_data.append({
                'Country': row['Country'],
                'Scenario': scenario_display_name,
                'Price': row[price_column]
            })

    # Create DataFrame and Plot
    price_df = pd.DataFrame(price_data)

    if not price_df.empty:
        fig = px.bar(price_df, x='Country', y='Price', color='Scenario', barmode='group',
                     title=f'Price Comparison for Key Economies: {drug_name}',
                     labels={'Price': 'Price per Patient (USD)'})

        # Update layout: Sort order (Largest GDP on the Right means reversing the descending GDP list)
        # and use logarithmic scale for Y axis.
        fig.update_layout(
            xaxis_title="Country (Sorted by GDP, Largest on Right)",
            yaxis_type="log",
            xaxis={'categoryorder': 'array', 'categoryarray': country_gdp_order[::-1]}
        )

        # Save HTML and PNG files
        html_path = os.path.join(output_folder, 'key_country_price_comparison_chart.html')
        fig.write_html(html_path)
        try:
            png_path = html_path.replace('.html', '.png')
            fig.write_image(png_path)
        except Exception as e:
            print(
                f"Warning: Could not save {png_path}. Ensure 'kaleido' is installed (`pip install kaleido`). Error: {e}")


def generate_scatter_plots(scenario_dfs, drug_name, output_folder):
    # (Implementation remains the same as original script, but relies on standardized column names)
    # Define Symlog C values (Linear thresholds)
    SYMLOG_C_PRICE = 100
    SYMLOG_C_HEALTH_EXP = 100
    SYMLOG_C_BUDGET_SCATTER = 0.1

    for scenario_name, df_raw in scenario_dfs.items():
        df = df_raw.copy()

        # Determine the Price column
        if scenario_name == 'Current_System':
            price_col = f'Current {drug_name} Price'
            if 'Scenario Patients' not in df.columns:
                # Use standardized name
                df['Scenario Patients'] = df[f'{DRUG_TO_ANALYZE} - Est. Annual Patient Population']
        else:
            price_col = 'Final Price'

        if price_col not in df.columns: continue

        plot_df = df.copy()

        # --- Plot 1: Price vs. Health Expenditure per Capita ---
        # (Plotting logic remains the same...)
        # Apply Symlog transformations
        plot_df['Price_symlog'] = symlog_transform(plot_df[price_col], SYMLOG_C_PRICE)
        plot_df['HealthExp_symlog'] = symlog_transform(plot_df['Health Expenditure per Capita'], SYMLOG_C_HEALTH_EXP)

        # Generate Ticks
        x_ticks, x_labels = generate_symlog_ticks(plot_df[price_col].min(), plot_df[price_col].max(), SYMLOG_C_PRICE,
                                                  use_suffixes=True, is_percentage=False)
        y_ticks, y_labels = generate_symlog_ticks(plot_df['Health Expenditure per Capita'].min(),
                                                  plot_df['Health Expenditure per Capita'].max(), SYMLOG_C_HEALTH_EXP,
                                                  use_suffixes=True, is_percentage=False)

        # Create Figure
        fig = px.scatter(plot_df, x='Price_symlog', y='HealthExp_symlog',
                         hover_name='Country',
                         hover_data={price_col: ':,.0f', 'Health Expenditure per Capita': ':,.0f',
                                     'Price_symlog': False, 'HealthExp_symlog': False},
                         title=f"{drug_name} Price vs. Health Expenditure per Capita ({scenario_name.replace('_', ' ')})",
                         color='Income Level',
                         size='Scenario Patients', size_max=40)

        # Update Axes to use custom Symlog ticks
        fig.update_layout(
            xaxis_title="Price (USD)",
            yaxis_title="Health Expenditure per Capita (USD)",
            xaxis=dict(tickmode='array', tickvals=x_ticks, ticktext=x_labels),
            yaxis=dict(tickmode='array', tickvals=y_ticks, ticktext=y_labels)
        )

        # Add Annotations for Key Countries
        for country in KEY_COUNTRIES_TO_LABEL:
            if country in plot_df['Country'].values:
                try:
                    data_point = plot_df[plot_df['Country'] == country].iloc[0]
                    fig.add_annotation(x=data_point['Price_symlog'], y=data_point['HealthExp_symlog'],
                                       text=country, showarrow=True, arrowhead=1, ax=20, ay=-30)
                except IndexError:
                    continue

        # Save HTML and PNG files
        html_path_1 = os.path.join(output_folder, f"scatter_Price_vs_HealthExp_{scenario_name.replace(' ', '_')}.html")
        fig.write_html(html_path_1)
        try:
            png_path_1 = html_path_1.replace('.html', '.png')
            fig.write_image(png_path_1)
        except Exception as e:
            print(
                f"Warning: Could not save {png_path_1}. Ensure 'kaleido' is installed (`pip install kaleido`). Error: {e}")

        # --- Plot 2: Price vs. Budget Impact (% THE) ---
        # (Plotting logic remains the same...)
        plot_df_budget = df.copy()

        # Apply Symlog transformations
        plot_df_budget['Price_symlog'] = symlog_transform(plot_df_budget[price_col], SYMLOG_C_PRICE)
        plot_df_budget['BudgetImpact_symlog'] = symlog_transform(plot_df_budget['Budget Impact (% THE)'],
                                                                 SYMLOG_C_BUDGET_SCATTER)

        # Generate Ticks (Y-axis uses percentage formatting)
        x_ticks_b, x_labels_b = generate_symlog_ticks(plot_df_budget[price_col].min(), plot_df_budget[price_col].max(),
                                                      SYMLOG_C_PRICE, use_suffixes=True, is_percentage=False)
        y_ticks_b, y_labels_b = generate_symlog_ticks(plot_df_budget['Budget Impact (% THE)'].min(),
                                                      plot_df_budget['Budget Impact (% THE)'].max(),
                                                      SYMLOG_C_BUDGET_SCATTER, is_percentage=True)

        # Create Figure
        fig_b = px.scatter(plot_df_budget, x='Price_symlog', y='BudgetImpact_symlog',
                           hover_name='Country',
                           hover_data={price_col: ':,.0f', 'Budget Impact (% THE)': ':.2f%', 'Price_symlog': False,
                                       'BudgetImpact_symlog': False},
                           title=f"{drug_name} Price vs. Budget Impact (% THE) ({scenario_name.replace('_', ' ')})",
                           color='Income Level',
                           size='Scenario Patients', size_max=40)

        # Update Axes
        fig_b.update_layout(
            xaxis_title="Price (USD)",
            yaxis_title="Budget Impact (% of Total Health Expenditure)",
            xaxis=dict(tickmode='array', tickvals=x_ticks_b, ticktext=x_labels_b),
            yaxis=dict(tickmode='array', tickvals=y_ticks_b, ticktext=y_labels_b)
        )

        # Add Annotations
        for country in KEY_COUNTRIES_TO_LABEL:
            if country in plot_df_budget['Country'].values:
                try:
                    data_point = plot_df_budget[plot_df_budget['Country'] == country].iloc[0]
                    fig_b.add_annotation(x=data_point['Price_symlog'], y=data_point['BudgetImpact_symlog'],
                                         text=country, showarrow=True, arrowhead=1, ax=20, ay=-30)
                except IndexError:
                    continue

        # Save HTML and PNG files
        html_path_2 = os.path.join(output_folder,
                                   f"scatter_Price_vs_BudgetImpact_{scenario_name.replace(' ', '_')}.html")
        fig_b.write_html(html_path_2)
        try:
            png_path_2 = html_path_2.replace('.html', '.png')
            fig_b.write_image(png_path_2)
        except Exception as e:
            print(
                f"Warning: Could not save {png_path_2}. Ensure 'kaleido' is installed (`pip install kaleido`). Error: {e}")


def generate_visualizations(scenario_dfs, drug_name, output_folder):
    # (Implementation remains the same as original script, relies on ISO-3 codes being present)
    """Generates Symlog maps, Binary Profit maps, and regional bar charts."""

    # Define metrics and Symlog thresholds (C)
    metrics_to_map = ['Revenue', 'Profit', 'QALYs Gained', 'Lives Saved', 'Budget Impact (% THE)']
    SYMLOG_C_FINANCIAL = 1000
    SYMLOG_C_HEALTH = 10
    SYMLOG_C_BUDGET = 0.1

    for scenario_name, df_raw in scenario_dfs.items():
        df = df_raw.copy()

        # Ensure ISO-3 mapping is present (it should be after the merge correction in main)
        if 'ISO-3' not in df.columns:
            df['ISO-3'] = df['Country'].map(COMPREHENSIVE_ISO_MAP)

        # Filter out rows without ISO codes for plotting
        plot_df_base = df.dropna(subset=['ISO-3']).copy()
        if plot_df_base.empty: continue

        # --- 1. Generate Detailed Choropleth Maps (Symlog) ---
        # (Plotting logic remains the same...)
        for metric in metrics_to_map:
            if metric in plot_df_base.columns:
                plot_df = plot_df_base.copy()

                # Determine Symlog C value and formatting
                is_percentage = '(% THE)' in metric
                if metric in ['Revenue', 'Profit']:
                    C = SYMLOG_C_FINANCIAL
                elif is_percentage:
                    C = SYMLOG_C_BUDGET
                else:
                    C = SYMLOG_C_HEALTH

                # Apply Symlog transformation
                plot_df[f'{metric}_symlog'] = symlog_transform(plot_df[metric], C)
                color_metric = f'{metric}_symlog'

                # Determine color scale
                v_min_actual = plot_df[metric].min();
                v_max_actual = plot_df[metric].max()

                if v_min_actual < -1e-9:  # Check if significantly negative
                    color_scale = px.colors.diverging.RdBu
                    transformed_v_min = plot_df[color_metric].min();
                    transformed_v_max = plot_df[color_metric].max()
                    v_abs_max = max(abs(transformed_v_min), abs(transformed_v_max))
                    color_range = [-v_abs_max, v_abs_max] if v_abs_max > 0 else None
                else:
                    color_scale = px.colors.sequential.Plasma
                    color_range = None

                # Generate Ticks (using standardized formatting)
                tick_vals, tick_text = generate_symlog_ticks(v_min_actual, v_max_actual, C, use_suffixes=True,
                                                             is_percentage=is_percentage)

                # Create the figure
                fig = px.choropleth(plot_df,
                                    locations="ISO-3",
                                    locationmode='ISO-3',
                                    color=color_metric,
                                    hover_name="Country",
                                    hover_data={metric: (':.2f' if is_percentage else ':,.2f'), color_metric: False,
                                                'Income Level': True},
                                    title=f"{metric.replace('_', ' ')} for {drug_name} ({scenario_name.replace('_', ' ')})",
                                    color_continuous_scale=color_scale,
                                    range_color=color_range)

                # Update color bar
                fig.update_layout(
                    coloraxis_colorbar=dict(title=metric, tickvals=tick_vals, ticktext=tick_text)
                )

                if v_min_actual < -1e-9 and color_range:
                    fig.update_layout(coloraxis_cmid=0)

                filename_metric = metric.replace(' (% THE)', '').replace(' ', '_')

                # Save HTML and PNG files
                html_path = os.path.join(output_folder, f"map_{scenario_name.replace(' ', '_')}_{filename_metric}.html")
                fig.write_html(html_path)
                try:
                    png_path = html_path.replace('.html', '.png')
                    fig.write_image(png_path)
                except Exception as e:
                    print(
                        f"Warning: Could not save {png_path}. Ensure 'kaleido' is installed (`pip install kaleido`). Error: {e}")

        # --- 2. Generate Binary Profit Status Map ---
        # (Plotting logic remains the same...)
        if 'Profit' in plot_df_base.columns:
            plot_df_profit = plot_df_base.copy()

            # Create categorical status
            conditions = [
                (plot_df_profit['Profit'] > 0.01),  # Use a small threshold
                (plot_df_profit['Profit'] < -0.01)
            ]
            choices = ['Profit (> $0)', 'Loss (< $0)']
            plot_df_profit['Profit Status'] = np.select(conditions, choices, default='Zero/Neutral')

            color_discrete_map = {'Profit (> $0)': 'Green', 'Loss (< $0)': 'Red', 'Zero/Neutral': 'LightGrey'}

            fig_status = px.choropleth(plot_df_profit,
                                       locations="ISO-3",
                                       color='Profit Status',
                                       hover_name="Country",
                                       hover_data={'Profit': ':,.2f', 'Income Level': True},
                                       title=f"Profit Status for {drug_name} ({scenario_name.replace('_', ' ')})",
                                       color_discrete_map=color_discrete_map,
                                       category_orders={
                                           "Profit Status": ["Profit (> $0)", "Loss (< $0)", "Zero/Neutral"]})

            # Save HTML and PNG files
            html_path_status = os.path.join(output_folder, f"map_{scenario_name.replace(' ', '_')}_Profit_Status.html")
            fig_status.write_html(html_path_status)
            try:
                png_path_status = html_path_status.replace('.html', '.png')
                fig_status.write_image(png_path_status)
            except Exception as e:
                print(
                    f"Warning: Could not save {png_path_status}. Ensure 'kaleido' is installed (`pip install kaleido`). Error: {e}")

    # --- 3. Regional Bar Charts (Revenue and Price) ---
    # (Plotting logic remains the same...)

    # Regional Revenue
    regional_revenue_data = {}
    for scenario, df in scenario_dfs.items():
        if 'Region' in df.columns and 'Revenue' in df.columns:
            valid_df = df[df['Region'].notna()]
            regional_revenue_data[scenario] = valid_df.groupby('Region')['Revenue'].sum()

    if regional_revenue_data:
        regional_revenue_summary = pd.DataFrame(regional_revenue_data).reset_index()
        if not regional_revenue_summary.empty:
            fig_reg_rev = px.bar(regional_revenue_summary, x='Region', y=list(scenario_dfs.keys()),
                                 title=f'Regional Revenue Comparison (Standardized Regions) for {drug_name}',
                                 barmode='group',
                                 labels={'value': 'Total Revenue (USD)', 'variable': 'Scenario'})
            fig_reg_rev.update_layout(xaxis_tickangle=-45)

            # Save HTML and PNG files
            html_path_rev = os.path.join(output_folder, 'regional_revenue_comparison_chart.html')
            fig_reg_rev.write_html(html_path_rev)
            try:
                png_path_rev = html_path_rev.replace('.html', '.png')
                fig_reg_rev.write_image(png_path_rev)
            except Exception as e:
                print(
                    f"Warning: Could not save {png_path_rev}. Ensure 'kaleido' is installed (`pip install kaleido`). Error: {e}")

    # Regional Price Comparison
    price_col = f'Current {DRUG_TO_ANALYZE} Price'
    # Use standardized name
    patient_col_base = f'{DRUG_TO_ANALYZE} - Est. Annual Patient Population'
    regional_price_data = []

    for scenario_name, df in scenario_dfs.items():
        if 'Region' not in df.columns: continue
        df_copy = df[df['Region'].notna()].copy()

        if scenario_name == 'Current_System':
            price_column_name = price_col
            patient_column_name = patient_col_base
        else:
            price_column_name = 'Final Price'
            patient_column_name = 'Scenario Patients'

        if price_column_name not in df_copy.columns or patient_column_name not in df_copy.columns:
            continue

        df_copy['price_times_patients'] = df_copy[price_column_name] * df_copy[patient_column_name]
        grouped = df_copy.groupby('Region').agg(
            total_price_patients=('price_times_patients', 'sum'),
            total_patients=(patient_column_name, 'sum')
        ).reset_index()

        grouped['Avg Price'] = grouped.apply(
            lambda row: row['total_price_patients'] / row['total_patients'] if row['total_patients'] > 0 else 0,
            axis=1
        )

        regional_prices = grouped[['Region', 'Avg Price']].copy()
        regional_prices['Scenario'] = scenario_name.replace('_', ' ')
        regional_price_data.append(regional_prices)

    if regional_price_data:
        regional_price_summary = pd.concat(regional_price_data)
        fig_reg_price = px.bar(regional_price_summary, x='Region', y='Avg Price', color='Scenario',
                               title=f'Weighted Average Regional Price Comparison for {drug_name}', barmode='group',
                               labels={'Avg Price': 'Average Price per Patient (USD)'})
        fig_reg_price.update_layout(xaxis_tickangle=-45)

        # Save HTML and PNG files
        html_path_price = os.path.join(output_folder, 'regional_price_comparison_chart.html')
        fig_reg_price.write_html(html_path_price)
        try:
            png_path_price = html_path_price.replace('.html', '.png')
            fig_reg_price.write_image(png_path_price)
        except Exception as e:
            print(
                f"Warning: Could not save {png_path_price}. Ensure 'kaleido' is installed (`pip install kaleido`). Error: {e}")


def generate_pdf_report(report_data, drug_name, budget_impact_summary_df, output_folder):
    # (Implementation remains the same as original script)
    # --- Generate Comparison Chart ---
    labels = list(report_data.keys())

    # Filter out scenarios that might have been skipped (e.g., if anchor price was 0)
    valid_labels = []
    revenues = []
    profits = []
    for label in labels:
        if label in report_data:
            valid_labels.append(label)
            revenues.append(report_data[label]['Total Revenue'] / 1e9)
            profits.append(report_data[label]['Total Profit'] / 1e9)

    if not valid_labels:
        print("No scenarios to report. PDF generation skipped.")
        return

    x = np.arange(len(valid_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, revenues, width, label='Revenue', color='royalblue')
    rects2 = ax.bar(x + width / 2, profits, width, label='Profit', color='lightgreen')

    ax.set_ylabel('Amount (in Billions USD)')
    ax.set_title(f'Comparative Revenue and Profit for {drug_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, rotation=45, ha="right")
    ax.legend()

    # Handle potential formatting issues if values are NaN or inf
    def safe_format(val):
        if np.isfinite(val):
            return f'${val:,.1f}B'
        return 'N/A'

    ax.bar_label(rects1, padding=3, labels=[safe_format(v) for v in revenues])
    ax.bar_label(rects2, padding=3, labels=[safe_format(v) for v in profits])

    fig.tight_layout()
    chart_filename = os.path.join(output_folder, 'revenue_profit_comparison.png')
    plt.savefig(chart_filename)
    plt.close()

    # --- Generate PDF ---
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, f'Final Verified Pricing Model Analysis: {drug_name}', new_x=XPos.LMARGIN,
             new_y=YPos.NEXT, align='C')
    pdf.ln(5)

    # Executive Summary (Updated)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Executive Summary (Verified Baseline)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    summary_text = (
        "This report presents the finalized analysis with a corrected baseline for the 'Current System'. "
        f"Based on the premise that {', '.join(COUNTRIES_TO_EXCLUDE_FROM_CURRENT_SYSTEM)} currently has no access, their participation in the 'Current System' has been explicitly excluded (Price=$0, Patients=0). "
        "This correction ensures the baseline accurately reflects the defined access limitations and highlights the significant expansion of access modeled in the GDP-dependent scenarios."
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(5)

    # Add Chart
    if os.path.exists(chart_filename):
        pdf.image(chart_filename, x=10, y=None, w=190)
        pdf.ln(5)

    # Scenario Comparison Table
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Scenario Comparison: Key Metrics', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', 'B', 9)
    num_scenarios = len(valid_labels);
    metric_width = 55
    value_width = (190 - metric_width) / num_scenarios if num_scenarios > 0 else 0
    pdf.cell(metric_width, 10, 'Metric', 1)
    for label in valid_labels:
        pdf.cell(value_width, 10, label, 1, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    metrics_for_report = ['Total Revenue', 'Total Profit', 'Patients Treated', 'Total QALYs Gained',
                          'Total Lives Saved']
    for metric in metrics_for_report:
        pdf.cell(metric_width, 10, metric.replace('Total ', ''), 1)
        for label in valid_labels:
            val = report_data[label].get(metric, 0)
            if 'Revenue' in metric or 'Profit' in metric:
                formatted_val = f"${val / 1e9:,.2f}B"
            else:
                formatted_val = f"{int(val):,}"
            pdf.cell(value_width, 10, formatted_val, 1, align='R')
        pdf.ln()

    # --- Budget Impact Analysis Section ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Budget Impact Analysis (% of Total Health Expenditure)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    if not budget_impact_summary_df.empty:
        pdf.set_font('Helvetica', 'B', 9)
        col_widths = [40, 40, 40, 40]
        headers = ['Scenario', 'Income Level', 'Avg Impact (% THE)', 'Max Impact (% THE)']
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 10, header, 1)
        pdf.ln()
        pdf.set_font('Helvetica', '', 9)
        for index, row in budget_impact_summary_df.iterrows():
            pdf.cell(col_widths[0], 10, str(row['Scenario']), 1)
            pdf.cell(col_widths[1], 10, str(row['Income Level']), 1)
            pdf.cell(col_widths[2], 10, f"{row['Average Impact (% THE)']:.2f}%", 1, align='R')
            pdf.cell(col_widths[3], 10, f"{row['Maximum Impact (% THE)']:.2f}%", 1, align='R')
            pdf.ln()

    # Key Assumptions (Updated)
    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Key Assumptions and Model Logic (Corrected)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    assumptions_text = (
        f"- **Current System Baseline [CORRECTED]:** If the current price is $0, OR if the country is in the exclusion list ({', '.join(COUNTRIES_TO_EXCLUDE_FROM_CURRENT_SYSTEM)}), the current patient population and price are assumed to be 0.\n"
        "- **Universal Acceptance of GDP Pricing:** ALL countries accept the GDP-dependent price in the alternative scenarios.\n"
        "- **Patient Adoption (GDP Scenarios):** Established markets maintain current volumes; non-established markets (now including China) expand to potential forecasts.\n"
        f"- **Budget Cap:** Prices are capped if the total cost exceeds {HEALTHCARE_BUDGET_CAP_PERCENTAGE * 100:.0f}% of a country's Total Health Expenditure.\n"
        "- **Data Completeness:** All countries in the input data are mapped using standardized ISO-3 codes."
    )
    pdf.multi_cell(0, 5, assumptions_text)

    # Output PDF
    pdf_output_path = os.path.join(output_folder, 'summary_report_final_verified.pdf')
    try:
        pdf.output(pdf_output_path)
    except Exception as e:
        print(f"Error generating PDF: {e}")


# --- Main Execution ---

def main():
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Add a note about the 'kaleido' dependency for PNG export
    print("Note: For PNG image export of charts, please ensure 'kaleido' is installed (`pip install kaleido`).")

    # Set Matplotlib backend
    try:
        plt.switch_backend('Agg')
    except Exception:
        pass

    # --- Data Loading ---
    try:
        drug_costs_df = pd.read_csv('drug_costs.csv')
        # Load the raw country data first
        country_data_raw_df = pd.read_csv('country_data.csv')

        # Load health impact file
        if os.path.exists('health_impact.csv'):
            health_impact_df = pd.read_csv('health_impact.csv')
        elif os.path.exists('health_impact_summary.csv'):
            health_impact_df = pd.read_csv('health_impact_summary.csv')
        else:
            raise FileNotFoundError("Health impact data file not found.")

        potential_patients_df = pd.read_csv('potential_patient_forecast.csv')

    except FileNotFoundError as e:
        print(f"Error loading input files: {e}")
        return

    # --- Data Preprocessing and Cleaning ---

    # Start with a copy of the raw data for processing
    country_data_df = country_data_raw_df.copy()

    # 1. Clean QALY data
    if 'Healthy Years Gained (QALYs)' in health_impact_df.columns:
        health_impact_df['QALYs (Upper Bound)'] = health_impact_df['Healthy Years Gained (QALYs)'].apply(
            clean_qaly_data)
    if 'QALYs (Upper Bound)' not in health_impact_df.columns:
        health_impact_df['QALYs (Upper Bound)'] = 0
    else:
        health_impact_df['QALYs (Upper Bound)'] = pd.to_numeric(health_impact_df['QALYs (Upper Bound)'],
                                                                errors='coerce').fillna(0)

    # 2. Merge dataframes and apply standardized regions

    # CORRECTION 1: Use ISO codes for robust merging instead of inconsistent Country Names
    # Map ISO codes to country_data_df using the comprehensive map
    country_data_df['ISO-3'] = country_data_df['Country'].map(COMPREHENSIVE_ISO_MAP)

    # Merge using ISO codes.
    if 'ISO3_Code' in potential_patients_df.columns and 'Potential_Patient_Forecast' in potential_patients_df.columns:
        # Merge the main dataframe with the forecast dataframe on the ISO codes
        country_data_df = pd.merge(country_data_df, potential_patients_df[['ISO3_Code', 'Potential_Patient_Forecast']],
                                   left_on='ISO-3', right_on='ISO3_Code', how='left')

        # Drop the redundant ISO column used for merging
        country_data_df = country_data_df.drop(columns=['ISO3_Code'])

        # CORRECTION 2: Rename the forecast column to the name expected by the rest of the script
        country_data_df = country_data_df.rename(columns={'Potential_Patient_Forecast': 'Potential Patients'})
    else:
        print(
            "Error: Expected columns 'ISO3_Code' or 'Potential_Patient_Forecast' not found in potential_patient_forecast.csv.")
        # Initialize 'Potential Patients' to 0 if required columns are missing
        country_data_df['Potential Patients'] = 0

    country_data_df['Region'] = country_data_df['Country'].apply(get_standardized_region)

    # 3. Parse financial columns
    for col in ['GDP (Current US$)', 'Total Health Expenditure (Current US$)']:
        country_data_df[col] = country_data_df[col].apply(parse_financial_string)

    # 4. Data Correction (Afghanistan GDP Correction - $14.58B)
    AFG_CORRECTED_GDP = 14.58 * 1e9
    try:
        if 'Afghanistan' in country_data_df['Country'].values:
            current_afg_gdp = country_data_df.loc[country_data_df['Country'] == 'Afghanistan', 'GDP (Current US$)']
            if not current_afg_gdp.empty and current_afg_gdp.iloc[0] < (AFG_CORRECTED_GDP * 0.5):
                country_data_df.loc[
                    country_data_df['Country'] == 'Afghanistan', 'GDP (Current US$)'] = AFG_CORRECTED_GDP
    except IndexError:
        pass

    # 5. Calculate Per Capita Metrics and Income Level
    country_data_df['Population'] = pd.to_numeric(country_data_df['Population'].astype(str).str.replace(',', ''),
                                                  errors='coerce').fillna(0)

    # GDP per Capita
    country_data_df['GDP per Capita'] = country_data_df.apply(
        lambda row: row['GDP (Current US$)'] / row['Population'] if row['Population'] > 0 else 0,
        axis=1
    ).replace([np.inf, -np.inf], 0)

    # Health Expenditure per Capita
    country_data_df['Health Expenditure per Capita'] = country_data_df.apply(
        lambda row: row['Total Health Expenditure (Current US$)'] / row['Population'] if row['Population'] > 0 else 0,
        axis=1
    ).replace([np.inf, -np.inf], 0)

    country_data_df['Income Level'] = country_data_df['GDP per Capita'].apply(classify_income)

    # 6. Parse Patient Population Ranges and Standardize Column Names
    # CORRECTION 4: Standardize the output column names to ensure robustness.
    expected_suffix = ' - Est. Annual Patient Population'

    for col in country_data_df.columns:
        if 'Patient Population Range' in col:
            # Extract the Drug Name (assuming format: Drug - Description Range)
            drug_name_match = col.split(' - ')[0].strip()

            # Define the standardized output column name
            standardized_col_name = f"{drug_name_match}{expected_suffix}"

            # Clean the data (extract lower bound)
            cleaned_series_lower = country_data_df[col].astype(str).str.split(' - ').str[0].str.replace(',',
                                                                                                        '').str.replace(
                '< ', '', regex=False)

            # Assign cleaned data to the standardized column name
            country_data_df[standardized_col_name] = pd.to_numeric(cleaned_series_lower, errors='coerce').fillna(
                0).astype(int)

    # --- Determine GDP Sort Order for Key Economies Chart (Before baseline modifications) ---
    key_economies_df = country_data_df[country_data_df['Country'].isin(KEY_ECONOMIES_PRICE_CHART)]
    # Sort descending by GDP (Largest first)
    country_gdp_order = key_economies_df.sort_values(by='GDP (Current US$)', ascending=False)['Country'].tolist()

    # --- Capture Original Prices for Anchoring (Before baseline modifications) ---
    # Crucial Step: Capture the original prices before applying the China exclusion for the baseline.
    original_prices = {}
    price_col = f'Current {DRUG_TO_ANALYZE} Price'

    if price_col in country_data_df.columns:
        # Ensure the column is numeric before capturing prices (handles 'NA' strings in CSV)
        country_data_df[price_col] = pd.to_numeric(country_data_df[price_col], errors='coerce').fillna(0)

        for country in ['United States', 'United Kingdom', 'China']:
            try:
                original_prices[country] = country_data_df.loc[country_data_df['Country'] == country, price_col].iloc[0]
            except IndexError:
                original_prices[country] = 0
    else:
        print(f"Error: Price column '{price_col}' not found.")
        return

    # 7. CRITICAL CORRECTION: Enforce Baseline Exclusions and Zero Price/Zero Patient Rule
    # Modify the dataframe (country_data_df) to reflect the Current System premise.

    # Use the standardized column name
    patient_col = f'{DRUG_TO_ANALYZE} - Est. Annual Patient Population'

    if patient_col in country_data_df.columns:
        # Explicitly set Price and Patients to 0 for excluded countries (e.g., China)
        country_data_df.loc[country_data_df['Country'].isin(COUNTRIES_TO_EXCLUDE_FROM_CURRENT_SYSTEM), price_col] = 0

        # Set patient population to 0 where price is 0 (this handles exclusions AND natural zeros)
        country_data_df.loc[country_data_df[price_col] == 0, patient_col] = 0
    else:
        print(f"Error: Standardized patient column '{patient_col}' not found.")
        return

    # 8. Final cleanup
    country_data_df['Potential Patients'] = country_data_df['Potential Patients'].fillna(0).astype(int)
    numeric_cols = country_data_df.select_dtypes(include=np.number).columns
    country_data_df[numeric_cols] = country_data_df[numeric_cols].fillna(0)

    # --- Current System Analysis (Using Corrected Data) ---
    current_df = country_data_df.copy()

    # Get drug cost
    # CORRECTION 3: Use the helper function to get the drug cost
    drug_cost = get_drug_cost(drug_costs_df, DRUG_TO_ANALYZE)

    # Calculate base revenue (China is now correctly excluded here)
    base_revenue = (current_df[price_col] * current_df[patient_col]).sum()

    # Calculate revenue shortfall (This will be larger now)
    revenue_shortfall = max(0, ACTUAL_KEYTRUDA_REVENUE - base_revenue)

    # Estimate medical tourism
    high_income_countries = current_df[current_df[price_col] > 50000]
    if not high_income_countries.empty:
        avg_tourist_price = high_income_countries[price_col].mean()
    else:
        avg_tourist_price = current_df[price_col].mean() if current_df[price_col].max() > 0 else 0
    medical_tourists = 0
    if revenue_shortfall > 0 and avg_tourist_price > 0:
        medical_tourists = revenue_shortfall / avg_tourist_price

    # Calculate adjusted totals
    total_patients_current = current_df[patient_col].sum() + medical_tourists
    current_revenue_adjusted = ACTUAL_KEYTRUDA_REVENUE if revenue_shortfall >= 0 else base_revenue
    current_profit_adjusted = (current_revenue_adjusted - (total_patients_current * drug_cost))

    # Calculate current health outcomes
    drug_qaly = health_impact_df.set_index('Drug')['QALYs (Upper Bound)'].get(DRUG_TO_ANALYZE, 0)
    current_qaly = total_patients_current * drug_qaly
    mortality_reduction_rate = 0.20
    current_lives_saved = total_patients_current * mortality_reduction_rate

    # Initialize report data structure
    report_data = {
        'Current System': {
            'Total Revenue': current_revenue_adjusted,
            'Total Profit': current_profit_adjusted,
            'Patients Treated': total_patients_current,
            'Total QALYs Gained': current_qaly,
            'Total Lives Saved': current_lives_saved,
        }
    }

    # --- GDP-Dependent Pricing Scenarios ---
    anchor_countries = ['United States', 'United Kingdom', 'China']
    scenario_dfs = {}

    # Prepare the Current System dataframe
    current_system_map_df = current_df.copy()
    current_system_map_df['Scenario Patients'] = current_system_map_df[patient_col]
    current_system_map_df['Revenue'] = current_system_map_df[price_col] * current_system_map_df[patient_col]
    current_system_map_df['Profit'] = (current_system_map_df[price_col] - drug_cost) * current_system_map_df[
        patient_col]
    # Health outcomes are now correctly 0 for China
    current_system_map_df['QALYs Gained'] = current_system_map_df[patient_col] * drug_qaly
    current_system_map_df['Lives Saved'] = (current_system_map_df[patient_col] * mortality_reduction_rate).fillna(
        0).astype(int)

    current_system_map_df['Budget Impact (% THE)'] = current_system_map_df.apply(
        lambda row: (row['Revenue'] / row['Total Health Expenditure (Current US$)']) * 100 if row[
                                                                                                  'Total Health Expenditure (Current US$)'] > 0 else 0,
        axis=1
    )

    scenario_dfs['Current_System'] = current_system_map_df

    # Run the model for each anchor country
    for anchor in anchor_countries:

        # Use the ORIGINAL price captured before baseline modification
        anchor_price = original_prices.get(anchor, 0)

        if anchor_price == 0:
            # In the provided country_data.csv, China's price is NA (which becomes 0).
            print(f"Note: Anchor price for {anchor} is $0 based on input data. Skipping scenario '{anchor} Anchor'.")
            continue

        # Execute the model (passing the corrected country_data_df)
        scenario_results, scenario_df = model_drug_pricing(
            DRUG_TO_ANALYZE, anchor, anchor_price, CAP_PRICE, HEALTHCARE_BUDGET_CAP_PERCENTAGE,
            country_data_df, drug_costs_df, health_impact_df
        )

        # Store results
        if scenario_results:
            report_data[f'{anchor} Anchor'] = scenario_results
            scenario_dfs[f'{anchor}_Anchor'] = scenario_df

    # --- Reporting and Visualization ---
    if not report_data:
        return

    # Generate Budget Impact Summary
    budget_impact_summary_df = generate_budget_impact_summary(scenario_dfs)

    # Generate PDF Report
    generate_pdf_report(report_data, DRUG_TO_ANALYZE, budget_impact_summary_df, OUTPUT_FOLDER)

    # Generate Visualizations (Maps and Regional Charts)
    generate_visualizations(scenario_dfs, DRUG_TO_ANALYZE, OUTPUT_FOLDER)

    # Generate Scatter Plots
    generate_scatter_plots(scenario_dfs, DRUG_TO_ANALYZE, OUTPUT_FOLDER)

    # Generate Key Country Price Chart
    generate_key_country_price_chart(scenario_dfs, DRUG_TO_ANALYZE, country_gdp_order, OUTPUT_FOLDER)

    print(
        f"\nAnalysis complete. Data ingestion corrected and standardized. All outputs generated in the '{OUTPUT_FOLDER}' folder.")


if __name__ == '__main__':
    main()