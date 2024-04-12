# fix repetitive functions & naming convention needs to be improved (variables and functions)
# clean up comments and old/ unused code
# break down into multiple files/sheets
# main.py, 
# later: pmo2 on same timeline as pmo1/3, npv and final outputs/ summaries

## LEGEND ##
# IW38 = cost
# IP24/18/19 = strategy
# IK17 = counter

#import necessary libraries
import streamlit as st
import psycopg2
from psycopg2 import OperationalError
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from calendar import monthrange
from pandas.tseries.offsets import MonthBegin

#type of fleet
# ***adjust for new site ***
# read in from the fleet table/list
fleet_options = [
    "", "Excavator - Large", "Excavator - Large H1200", "Excavator - Large H2600", "D11R Dozers", "D11 Dozers", "D10 Dozers", "992 Loaders", "993 Loaders", "777F Haul Trucks", "777F Water Trucks", "Floats", "Graders", "Support Dozers", "Scrub Dozer", "Drills", "Road Trains"
]

#initiate connection to sql database
def init_connection():
    try:
        conn = psycopg2.connect(**st.secrets["postgres"])
        return conn
    except OperationalError:
        # Could not connect to the database
        return None

##Streamlit set up
#set page layout in streamlit to be wide
st.set_page_config(layout="wide")
#caches data for 10 minutes (600 secs)
@st.cache_data(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

#function to read in csv or xlsx files and return an error if incorrect format
def read_file(file_data):
    if file_data is not None:
        file_extension = file_data.name.split(".")[-1].lower()
        if file_extension == "csv":
            df = pd.read_csv(file_data)
        elif file_extension == "xlsx":
            df = pd.read_excel(file_data)
        else:
            st.warning("Unsupported file format. Please upload a CSV or XLSX file.")
            return None
        return df
    else:
        return None
    
#convert to fiscal month
def to_financial_month(date):
    if pd.isnull(date):
        return None
    # Financial year starts in July
    month_offset = 6
    financial_month = (date.month - 1 + month_offset) % 12 + 1
    financial_year = date.year if date.month < 7 else date.year + 1
    fin_month_str = f"{financial_year}-{financial_month:02d}"
    # Return as a period object
    return pd.Period(fin_month_str, freq='M')

#function to convert the maintenance interval to hours using the strategy and the average usage per day
# *** adjust for new site??? ***
#100k hour strategies are conditional
def convert_interval_to_hours(interval, average_usage_per_day):
    time_period = interval[-1]
    if time_period == "H":
        return int(interval[:-1])
    elif time_period == "h":
        return int(interval[:-1])
    elif time_period == "M":
        return int(interval[:-1])*average_usage_per_day * 30.75
    elif time_period == "W":
        return int(interval[:-1])*average_usage_per_day * 7
    elif time_period == "D":
        return int(interval[:-1])*average_usage_per_day
    elif time_period == "K":
        return 100000
    elif time_period == "Y":
        return int(interval[:-1])*average_usage_per_day * 365.25
    else:
        raise ValueError(f"Invalid time period: {time_period}. Supported time periods are h, H, M, W, D, K, and Y.")
    
#currently unused *******************************
def calculate_costs(smu, coefficients):
    # takes in coefficients to find cost for given hours
    a, b = coefficients
    return a * smu**2 + b * smu

##Setting up
#function for getting unit numbers for the fleet

def get_unit_numbers_by_fleet(fleet_name, POSTGRES):
    query = f"SELECT unit FROM fleet_list WHERE fleet = '{fleet_name}';"
    result_final = []
    if POSTGRES == 0:
        # Connection failed, request file uploads
        #st.warning("Could not connect to the database. Please upload the extra required files.")
        st.sidebar.subheader("Database file uploads")
        uploaded_fleet_list = st.sidebar.file_uploader("Upload Fleet List", type=["csv", "xlsx"])
        uploaded_linked_strategy = st.sidebar.file_uploader("Upload Linked Strategy", type=["csv", "xlsx"])
        
        if uploaded_fleet_list is not None and uploaded_linked_strategy is not None:
            # Read the uploaded files
            fleet_list = read_file(uploaded_fleet_list)
            result_df = fleet_list[fleet_list['Fleet'] == fleet_name]['Unit']
            result_final = result_df.tolist()
    else:
        uploaded_linked_strategy = None
        result = run_query(query)
        result_final = [row[0] for row in result]
    return result_final, uploaded_linked_strategy
#function to get the user inputs

def get_user_input(POSTGRES):
    #set up sidebar in streamlit
    st.sidebar.header("Input Assumptions")
    #list fleet options and select required fleet
    selected_fleet = st.sidebar.selectbox("Choose Fleet Option", fleet_options, index=0)
    #determine unit numbers based on chosen fleet
    unit_numbers, uploaded_linked_strategy = get_unit_numbers_by_fleet(selected_fleet, POSTGRES)
    if unit_numbers:
        st.sidebar.write(f"Unit Numbers: {unit_numbers}")
    #input end of life date
    eol_date = st.sidebar.date_input("Choose End of Life (EOL) Date", value=pd.to_datetime("2027-06-30"))
    st.sidebar.subheader("Scenarios")
    #choose number of strategies you want to run through
    num_strategies = st.sidebar.selectbox("Number of Strategies", list(range(1, 11)), index=2)
    strategy_hours = {}
    #loop through scenario hours based on the number of strategies you want to run
    for i in range(1, num_strategies + 1):
        strategy_name = f"Scenario {i}"
        strategy_hours[strategy_name] = st.sidebar.number_input(f"{strategy_name} - Replacement Hours", min_value=0, value=60000, step=20000)
    #upload the data files
    st.sidebar.subheader("File Uploads")
    cost_data = st.sidebar.file_uploader("Upload Cost Data", type=["csv", "xlsx"])
    component_counter_data = st.sidebar.file_uploader("Upload Component Counter Data", type=["csv", "xlsx"])
    master_counter_data = st.sidebar.file_uploader("Upload Master Counter Data", type=["csv", "xlsx"])
    unit_scenarios = {}
    #replacement hours for each unit
    for unit_number in unit_numbers:
        unit_scenarios[unit_number] = strategy_hours
    eol_date = pd.Timestamp(eol_date)
    #return the replacement hours for each unit, data files, chosen fleet and end of life date
    return unit_scenarios, cost_data, component_counter_data, master_counter_data, selected_fleet, eol_date, unit_numbers, uploaded_linked_strategy

#####################placeholder for npv
def calculate_npv(scenario_hours):
    return 123456.78

#configuring the output page, print npv for each scenario
def output_page(unit_scenarios):
    st.title("Output Page")
    st.header("Scenario NPV")
    for unit, scenarios in unit_scenarios.items():
        st.subheader(f"Unit {unit}")
        for scenario, replacement_hours in scenarios.items():
            npv_value = calculate_npv(replacement_hours)
            st.write(f"{scenario}: NPV - ${npv_value:.2f}")

#filter IW38 for only PMO2 and calculate average cost for each maint item

def avg_cost_data_pm02(df_cost):
    df_filtered = df_cost[df_cost.iloc[:, 0] == "PM02"]
    #st.write(df_filtered)
    df_avg_cost = df_filtered[df_filtered["TotSum (actual)"] > 10].groupby("MaintItem")["TotSum (actual)"].mean()
    return df_avg_cost

#extract required data for the chosen fleet from master ip24 database

def get_master_strategy_data(selected_units, uploaded_linked_strategy):
    if uploaded_linked_strategy is not None:
        linked_strategy = read_file(uploaded_linked_strategy)
        expected_columns = [
            "Unit", "FunctionalLoc", "MaintenancePlan", "MaintItem", "MaintItemText", 
            "Description", "MaintItemDesc", "MaintItemInterval", "ik17component"
        ]
        for column in expected_columns:
            if column not in linked_strategy.columns:
                linked_strategy[column] = pd.NA

        # Reorder DataFrame to match the expected format
        df_linked_strategy = linked_strategy[expected_columns]
        df_result = df_linked_strategy[df_linked_strategy['Unit'].isin(selected_units)]
    else:
        unit_numbers_str = ", ".join(f"'{unit}'" for unit in selected_units)
        query = f"SELECT * FROM linkedip24 WHERE unit IN ({unit_numbers_str});"
        result = run_query(query)
        df_result = pd.DataFrame(result, columns=["Unit", "FunctionalLoc", "MaintenancePlan", "MaintItem", "MaintItemText", "Description","MaintItemDesc","MaintItemInterval","ik17component"])
    return df_result

#combine required data from iw38 and master ip24

def filter_cost_data_pm02(df_cost_data, selected_fleet, uploaded_linked_strategy):
    df_cost_filtered_PMO2 = avg_cost_data_pm02(df_cost_data)
    fleet_strategy_data = get_master_strategy_data(selected_fleet, uploaded_linked_strategy)
    if fleet_strategy_data is not None:
        fleet_strategy_data = pd.merge(fleet_strategy_data, df_cost_filtered_PMO2, on="MaintItem", how="left")
    return df_cost_filtered_PMO2, fleet_strategy_data

#filter IW38 for only PMO1 and calculate average cost for each maint item

# only need one of the following (pass in PMO1/3)
##########################
def pivot_cost_data(df_cost_data, data_type):
    df_filtered = df_cost_data[df_cost_data["Order Type"] == data_type]
    df_filtered_warranty = df_filtered[~df_filtered['Description'].str.contains('warranty', case=False, na= False)]
    #ensure date is datetime type
    df_filtered_warranty["Actual Finish"] = pd.to_datetime(df_filtered_warranty["Actual Finish"])
    #convert to fiscal month
    df_filtered_warranty["FinMonth"] = df_filtered_warranty.apply(lambda x: to_financial_month(x["Actual Finish"]) if pd.notnull(x["Actual Finish"]) else to_financial_month(x["Basic fin. date"]), axis=1)
    #create pivot table (needs to use maximum counter reading per month per unit)
    pivot_cost = df_filtered_warranty.pivot_table(values="Total act.costs", index="FinMonth", columns = "Sort field", aggfunc = "sum")
    return pivot_cost

#extract counter reading data from ik17 master

def process_master_counter_data(df_master_counter):
    relevant_columns = ["MeasPosition", "Counter reading", "Date"]
    df_master_counter_short = df_master_counter[relevant_columns].copy()
    df_master_counter_short["Date"] = pd.to_datetime(df_master_counter_short["Date"])
    summary_data = df_master_counter_short.groupby("MeasPosition").agg(
        Max_Hour_Reading=("Counter reading", "max"),
        Min_Hour_Reading=("Counter reading", "min"),
        Min_Date=("Date", "min"),
        Max_Date=("Date", "max")
    )
    summary_data["Hours_Used"] = summary_data["Max_Hour_Reading"] - summary_data["Min_Hour_Reading"]
    summary_data["Total_Days"] = (summary_data["Max_Date"] - summary_data["Min_Date"]).dt.days + 1
    summary_data["Average_Hours_Per_Day"] = summary_data["Hours_Used"] / summary_data["Total_Days"]
    summary_data.reset_index(inplace=True)
    return summary_data


def pivot_master_counter_data(df_master_counter):
    #ensure date is datetime type
    df_master_counter["Date"] = pd.to_datetime(df_master_counter["Date"])
    #convert to fiscal month
    df_master_counter["FinMonth"] = df_master_counter["Date"].apply(to_financial_month)
    #create pivot table (needs to use maximum counter reading per month per unit)
    pivot_smu = df_master_counter.pivot_table(values="Counter reading", index="FinMonth", columns = "MeasPosition", aggfunc = "max")
    return pivot_smu


#merge ip24 fleet data with ik17 component data
def merge_counter_strategy_data(fleet_strategy_data, df_component_counter, summary_data):
    df_component_counter.rename(columns={"Measuring point": "ik17component"}, inplace=True)
    merged_data = pd.merge(fleet_strategy_data, df_component_counter, on="ik17component", how="left")
    df_component_counter.rename(columns={"Counter reading": "Counter reading (IK17)"}, inplace=True)
    merged_data = pd.merge(merged_data, summary_data[["MeasPosition", "Average_Hours_Per_Day"]], left_on="Unit", right_on="MeasPosition", how="left")
    merged_data["MaintItemInterval"] = merged_data.apply(lambda row: convert_interval_to_hours(row["MaintItemInterval"], row["Average_Hours_Per_Day"]), axis=1)
    return merged_data

# *** adjust for new site **
#merging smu and cost pivots
def combine_cost_smu_pivots(pivot_smu, pivot_cost_PM01, pivot_cost_PM03, summary_data, unit_numbers):
    pivot_smu_reset = pivot_smu.reset_index()
    pivot_smu_reset.rename(columns={"MeasPosition": "Sort field"}, inplace=True)
    smu_melted = pivot_smu_reset.melt(id_vars=["FinMonth"], var_name="Sort field", value_name="Counter reading")
    
    # Dictionary to hold the melted DataFrames
    melted_dataframes = {}
    # List of tuples containing the DataFrame and its associated key
    df_pivots = [
        (pivot_cost_PM01, 'PMO1'),
        (pivot_cost_PM03, 'PMO3')
    ]
    # Loop over the DataFrames to reset the index and melt them
    for df_pivot, key in df_pivots:
        # Reset the index
        pivot_reset = df_pivot.reset_index()
        # Melt the DataFrame
        df_melted = pivot_reset.melt(id_vars=["FinMonth"], var_name="Sort field", value_name=f"Total act.costs")
        # Store the melted DataFrame in the dictionary
        melted_dataframes[key] = df_melted

    merged_pivots1 = pd.merge(smu_melted, melted_dataframes['PMO1'], on=["FinMonth", "Sort field"], how = "outer")
    merged_pivots = pd.merge(merged_pivots1, melted_dataframes['PMO3'], on=["FinMonth", "Sort field"], how = "outer")
    merged_pivots["Round SMU"] = (merged_pivots["Counter reading"] / 1000).round() * 1000
    #
    cumulative = cumulative_costs(merged_pivots)
    return cumulative


#outline replacement schedule for each unit
def create_PMO2_replacement_schedule(df_complete, current_month, eol_date):
    all_months = pd.date_range(current_month, eol_date, freq='M')
    replacement_schedule = pd.DataFrame(columns=["Interval", "Usual Days Until Replacement", "Unit", "Overdue", "Cost Missing"] + [month.strftime('%b-%y') for month in all_months])
    last_replacement_dates = {}

    for index, row in df_complete.iterrows():
        first_replacement_month = None
        replacement_schedule.loc[index, "MaintItem"] = row["MaintItem"]
        replacement_schedule.loc[index, "Interval"] = row["MaintItemInterval"]
        replacement_schedule.loc[index, "Unit"] = row["Unit"]
        interval = row["MaintItemInterval"]
        usage = row["Average_Hours_Per_Day"]
        current = row["Counter reading"]
        
        usual = interval / usage
        replacement_schedule.loc[index, "Overdue"] = False
        replacement_schedule.loc[index, "Cost Missing"] = pd.isnull(row["TotSum (actual)"])
        if interval < current:
            first_replacement_month = current_month
            replacement_schedule.loc[index, "Overdue"] = True
        else:
            first_replacement_month = current_month + pd.DateOffset(days=(interval - current) / usage)
        replacement_schedule.loc[index, "Usual Days Until Replacement"] = usual

        for month in all_months:
            month_str = month.strftime('%b-%y')
            if first_replacement_month > eol_date:
                break
            if month >= first_replacement_month:
                if row["MaintItem"] in last_replacement_dates and month <= last_replacement_dates[row["MaintItem"]]:
                    continue

                days_difference = (month - first_replacement_month).days
                replacements_this_month = int(days_difference / usual) + 1
                replacement_schedule.loc[index, month_str] = row["TotSum (actual)"] * replacements_this_month

                for _ in range(replacements_this_month):
                    first_replacement_month += pd.DateOffset(days=int(interval / usage))

                last_replacement_dates[row["MaintItem"]] = first_replacement_month - pd.DateOffset(days=int(interval / usage))

    return replacement_schedule

#one function ##########################
def smu_cost_fit(merged_pivots, X):
    pivot_table_max = merged_pivots.pivot_table(values='CumulativeCost%s' %X, 
                                                index='Round SMU', 
                                                columns='Sort field', 
                                                aggfunc='max')
    pivot_table_max['AverageCumulativeCost%s' %X] = pivot_table_max.mean(axis=1)
    # Assuming pivot_table_max is the pivot table created earlier
    cleaned_data = pivot_table_max[['AverageCumulativeCost%s' %X]].dropna()
    x = cleaned_data.index.values  # SMU values
    y = cleaned_data['AverageCumulativeCost%s' %X].values  # Average cumulative cost values
    # Construct the design matrix for quadratic fit (x^2 and x terms)
    X = np.vstack([x**2, x]).T  # .T to transpose, making it two columns
    # Solve the least squares problem
    coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # Rsquared:
    y_pred = coefficients[0] * x**2 + coefficients[1] * x  # Predicted y values based on the fit
    SS_res = np.sum((y - y_pred)**2)  # Sum of squares of residuals
    SS_tot = np.sum((y - np.mean(y))**2)  # Total sum of squares
    R_squared = 1 - (SS_res / SS_tot)
    return coefficients, R_squared

########################
def forecast_monthly_costs(iw39_df):
    # Placeholder for the forecasting method
    # For demonstration purposes, we'll assume a simple forecasting method
    # that multiplies the current cost by a fixed factor for each month.
    # Replace this with your actual forecasting logic.
    return

#displaying and merging data outputs
def merge_all_data(fleet_input, unit_scenarios, df_cost, df_component_counter, df_master_counter, uploaded_linked_strategy):
    selected_fleet = unit_scenarios.keys()
    df_cost_filtered_PM02, fleet_strategy_data = filter_cost_data_pm02(df_cost, selected_fleet, uploaded_linked_strategy)
    if df_cost_filtered_PM02 is None:
        return
    summary_data = process_master_counter_data(df_master_counter) if df_master_counter is not None else None
    merged_data = merge_counter_strategy_data(fleet_strategy_data, df_component_counter, summary_data) if df_component_counter is not None else None
    display_data(fleet_input, df_cost_filtered_PM02, fleet_strategy_data, summary_data, merged_data)
    return merged_data

#displaying all the extracted data
def display_data(fleet_input, df_cost_filtered_PM02, fleet_strategy_data, summary_data, merged_data):
    if df_cost_filtered_PM02 is not None and fleet_strategy_data is not None:
        st.subheader("Filtered Cost Data (PM02) with Average Cost")
        st.write(df_cost_filtered_PM02)
        st.header(f"{fleet_input} Strategy Data (Filtered by Unit Numbers with Average Cost per WO)")
        fleet_strategy_data = st.data_editor(fleet_strategy_data)
    elif summary_data is not None:
        st.write(summary_data)
    elif merged_data is not None:
        st.header("Component Counter Data")
        st.data_editor(merged_data)
        st.header("Component Counter Data with Maintenance Interval in Hours")
        st.data_editor(merged_data)

#overview of financial years
def show_fy_overview(replacement_schedule, start_date, end_date):
    fy_rows = []
    start_year = start_date.year
    end_year = end_date.year
    for unit in replacement_schedule["Unit"].unique():
        unit_schedule = replacement_schedule[replacement_schedule["Unit"] == unit]
        unit_row = {"Unit": unit}
        for year in range(start_year, end_year+1):
            fy_start = pd.Timestamp(f"{year-1}-07-01")
            fy_end = pd.Timestamp(f"{year}-06-30")
            if fy_start < start_date:
                continue
            fy_columns = []
            for col in unit_schedule.columns:
                try:
                    date = pd.to_datetime(col, format='%b-%y')
                    if fy_start <= date <= fy_end:
                        fy_columns.append(col)
                except ValueError:
                    continue

            unit_row[f"FY{str(year)[-2:]}"] = unit_schedule[fy_columns].sum(axis=1).sum()
        unit_row["Total (NOMINAL)"] = sum(value for key, value in unit_row.items() if key.startswith("FY"))
        fy_rows.append(pd.Series(unit_row))
    fy_overview = pd.concat(fy_rows, axis=1).transpose()
    return fy_overview


def cumulative_costs(merged_pivots): 
    # Replace NaN in cost columns with 0
    merged_pivots['Total act.costs_x'] = merged_pivots['Total act.costs_x'].fillna(0)
    merged_pivots['Total act.costs_y'] = merged_pivots['Total act.costs_y'].fillna(0)
    #First, ensure your DataFrame is sorted by "sort field" (machine) and then by "month/year" to correctly calculate cumulative costs.
    merged_pivots.sort_values(by=["Sort field", "FinMonth"], inplace=True)
    # Group by 'sort field' and then apply forward fill within each group
    merged_pivots['Round SMU'] = merged_pivots.groupby('Sort field')['Round SMU'].fillna(method='ffill')
    # Group by 'sort field' and then apply backfill within each group
    merged_pivots['Round SMU'] = merged_pivots.groupby('Sort field')['Round SMU'].fillna(method='bfill')
    #Calculate the total sum of costs, and min and max SMU (hours) for each machine.
    machine_stats = merged_pivots.groupby('Sort field').agg(
        TotalCostsX=("Total act.costs_x", 'sum'),
        TotalCostsY=("Total act.costs_y", 'sum'),
        MinSMU=("Counter reading", 'min'),
        MaxSMU=("Counter reading", 'max'),
        RoundMinSMU=("Round SMU", 'min')
    ).reset_index()
    #Now, calculate the initial cumulative cost factor based on your formula.
    machine_stats['InitialCumulativeFactorX'] = (machine_stats['TotalCostsX'] / (machine_stats['MaxSMU'] - machine_stats['MinSMU'])) * 0.5 * machine_stats['RoundMinSMU']
    machine_stats['InitialCumulativeFactorY'] = (machine_stats['TotalCostsY'] / (machine_stats['MaxSMU'] - machine_stats['MinSMU'])) * 0.5 * machine_stats['RoundMinSMU']
    #Merge this factor back into the original DataFrame on "sort field" to have the initial cumulative cost factor available for each row.
    merged_pivots = merged_pivots.merge(machine_stats[['Sort field', 'InitialCumulativeFactorX']], on='Sort field')
    merged_pivots = merged_pivots.merge(machine_stats[['Sort field', 'InitialCumulativeFactorY']], on='Sort field')
    #For each machine, calculate the cumulative costs, starting with the initial value for the earliest month, then cumulatively summing up total act.costs_x and total act.costs_y respectively.
    #Initialize columns to store the cumulative costs
    merged_pivots['CumulativeCostX'] = 0
    merged_pivots['CumulativeCostY'] = 0

    # Iterate over each machine to calculate cumulative costs#
    for machine in merged_pivots['Sort field'].unique():
        machine_mask = merged_pivots['Sort field'] == machine
        first_index = merged_pivots.loc[machine_mask].index[0]
        merged_pivots.loc[machine_mask, 'CumulativeCostX'] = merged_pivots.loc[machine_mask, 'Total act.costs_x'].cumsum() + merged_pivots.loc[first_index, 'InitialCumulativeFactorX'] - merged_pivots.loc[first_index, 'Total act.costs_x']
        merged_pivots.loc[machine_mask, 'CumulativeCostY'] = merged_pivots.loc[machine_mask, 'Total act.costs_y'].cumsum() + merged_pivots.loc[first_index, 'InitialCumulativeFactorY'] - merged_pivots.loc[first_index, 'Total act.costs_y']

    return merged_pivots


def forecast_unit_costs(start_month, end_of_life, current_hours, avg_hours_per_day, replacement_hours, coefficients_PM01, coefficients_PM03):
    months = pd.date_range(start=start_month, end=end_of_life, freq='MS')
    operating_hours = 0
    cumulative_hours = current_hours
    cumulative_costPM01 = cumulative_costPM03 = 0
    monthly_costPM01 = monthly_costPM03 = 0
    forecast_data = []

    # use first full month
    if start_month.day > 1:
        adjusted_start_month = start_month + MonthBegin(1)
    else:
        adjusted_start_month = start_month

    for month in months:
        # Get the number of days in the month
        days_in_month = pd.Period(month, freq='M').days_in_month

        if cumulative_hours >= replacement_hours:
            # If the unit needs replacement, record the current month and break
            replacement_start_month = month
            break

        operating_hours = avg_hours_per_day * days_in_month
        cumulative_hours += operating_hours

        new_cumulative_costPM01 = coefficients_PM01[0] * cumulative_hours ** 2 + coefficients_PM01[1] * cumulative_hours
        new_cumulative_costPM03 = coefficients_PM03[0] * cumulative_hours ** 2 + coefficients_PM03[1] * cumulative_hours

        if  month == adjusted_start_month:
            monthly_costPM01 = 0
            monthly_costPM03 = 0
        else:
            monthly_costPM01 = new_cumulative_costPM01 - cumulative_costPM01
            monthly_costPM03 = new_cumulative_costPM03 - cumulative_costPM03

        cumulative_costPM01 = new_cumulative_costPM01
        cumulative_costPM03 = new_cumulative_costPM03

        forecast_data.append({
            'Month': month.strftime('%Y-%m'),
            'Operating Hours': operating_hours,
            'Cumulative Hours': cumulative_hours,
            'Cumulative CostX': cumulative_costPM01,
            'Cumulative CostY': cumulative_costPM03,
            'Monthly CostX': monthly_costPM01,
            'Monthly CostY': monthly_costPM03
        })
    else:
        # If loop did not break, all months are forecasted, and no replacement is needed
        replacement_start_month = None

    return forecast_data, replacement_start_month

def calculate_average_daily_hours(units_data):
    total_hours = sum(data['avg_daily_hours'] for data in units_data.values())
    return total_hours / len(units_data)

def forecast_for_scenario(unit, units_data, start_month, scenario_replacement_hours, coefficients_costX, coefficients_costY, end_of_life):
    current_hours = units_data[unit]['current_hours']
    avg_hours_per_day = units_data[unit]['avg_daily_hours']
    
    forecast, replacement_start_month = forecast_unit_costs(
        start_month, end_of_life, current_hours, avg_hours_per_day,
        scenario_replacement_hours, coefficients_costX, coefficients_costY
    )
    
    return forecast, replacement_start_month

def forecast_all_units_scenarios(start_month, unit_numbers, units_data, scenarios, coefficients_costPM01, coefficients_costPM03, end_of_life):
    all_scenarios_forecasts = {}
    average_daily_hours = calculate_average_daily_hours(units_data)
    #start_month = pd.Timestamp('2024-04-01')  # Starting month (adjust if needed)

    for scenario_name, replacement_hours in scenarios.items():
        scenario_forecasts = {}
        
        for unit in unit_numbers:
            forecast, replacement_start_month = forecast_for_scenario(
                unit, units_data, start_month, replacement_hours,
                coefficients_costPM01, coefficients_costPM03, end_of_life
            )
            
            scenario_forecasts[unit] = forecast
            
            if replacement_start_month:
                # Forecast for the replacement unit in this scenario
                replacement_unit = f"{unit}_replacement_{scenario_name}"
                units_data[replacement_unit] = {
                    'current_hours': 0,
                    'avg_daily_hours': average_daily_hours
                }
                scenario_forecasts[replacement_unit], _ = forecast_unit_costs(
                    replacement_start_month, end_of_life, 0, average_daily_hours,
                    replacement_hours, coefficients_costPM01, coefficients_costPM03
                )
        
        all_scenarios_forecasts[scenario_name] = scenario_forecasts

    return all_scenarios_forecasts

def forecast_all_units_scenarios_to_csv(start_month, unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life, output_csv):
    all_scenarios_forecasts = forecast_all_units_scenarios(start_month, unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life)
    
    # Flatten the nested structure into a list of dictionaries for easy export to CSV
    csv_data = []
    for scenario, forecasts in all_scenarios_forecasts.items():
        for unit, unit_forecasts in forecasts.items():
            for month_data in unit_forecasts:
                month_data['Unit'] = unit
                month_data['Scenario'] = scenario
                csv_data.append(month_data)
                
    # Create a DataFrame and write it to a CSV file
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv, index=False)

def format_forecast_outputs(all_scenarios_forecasts):
    data = []
    # Extract data from the nested dictionary structure
    for scenario, units in all_scenarios_forecasts.items():
        for unit, unit_forecasts in units.items():
            for month_forecast in unit_forecasts:
                for data_type, value in month_forecast.items():
                    # Skip the 'Month' data type as we're going to use it as a column header
                    if data_type == 'Month':
                            continue
                    data.append({
                                'Scenario': scenario,
                                'Unit': unit,
                                'Data Type': data_type,
                                'Month': month_forecast['Month'],
                                'Value': value
                            })

    # Convert the list to a DataFrame
    df_long_format = pd.DataFrame(data)

    # Pivot the DataFrame to wide format with months as columns
    df_wide_format = df_long_format.pivot_table(
                index=['Scenario', 'Unit', 'Data Type'], 
                columns='Month', 
                values='Value',
                aggfunc='first'  # Assuming there's only one value per group
            )

    # Reset index to turn multi-index into columns
    df_wide_format.reset_index(inplace=True)

    # Move 'Month' level of column MultiIndex to be column names
    df_wide_format.columns = [col if isinstance(col, tuple) else col for col in df_wide_format.columns]
    df = df_wide_format        
    return df


#main func to call all functions and run the streamlit app 
# needs to be broken down/ cleaned up
def main(POSTGRES):
    st.title("Tundra Resource Analytics - Equipment Strategy Optimization Tool")
    current_month = pd.Timestamp('2024-04-13')

    unit_scenarios, cost_data, component_counter_data, master_counter_data, fleet_input, eol_date, unit_numbers, uploaded_linked_strategy = get_user_input(POSTGRES)
    st.header("User Input")
    pm01_pm03_df = None
    for unit, scenarios in unit_scenarios.items():
        st.write(f"Unit {unit}: {scenarios}")
    replacement_schedule = None
    if st.button("Show"):
        
        selected_fleet = unit_scenarios.keys()
        if not selected_fleet and not cost_data:
            st.info("Please choose a fleet option and upload the cost data to view Master Strategy Data with Average Cost.")
        elif not cost_data:
            st.info("Please upload the cost data.")
        elif not selected_fleet:
            st.info("Please choose a fleet option.")   
        elif not component_counter_data:
            st.info("Please upload the IW17 component data.")
        elif not master_counter_data:
            st.info("Please upload the IW17 master data.")
        else:
            df_cost = read_file(cost_data)
            df_component_counter = read_file(component_counter_data)
            df_master_counter = read_file(master_counter_data)
            pivot_smu = pivot_master_counter_data(df_master_counter)
            st.success("All data uploaded successfully!")
            complete_df = merge_all_data(fleet_input, unit_scenarios,df_cost,df_component_counter,df_master_counter, uploaded_linked_strategy)
            
            if complete_df is not None:
                st.header("Complete Data")
                st.dataframe(complete_df)
                replacement_schedule = create_PMO2_replacement_schedule(complete_df, current_month, eol_date)
                st.session_state['replacement_schedule'] = replacement_schedule
                cost_missing_indices = replacement_schedule[replacement_schedule["Cost Missing"]].index
                for idx in cost_missing_indices:
                    replacement_schedule.loc[idx, "TotSum (actual)"] = st.number_input(f"Enter cost for maintenance item {replacement_schedule.loc[idx, 'MaintItem']}:", value=0.0)
                overdue_indices = replacement_schedule[replacement_schedule["Overdue"]].index
                for idx in overdue_indices:
                    replacement_schedule.loc[idx, "First Replacement Month"] = st.date_input(f"Enter first replacement date for overdue component {replacement_schedule.loc[idx, 'MaintItem']}:", value=pd.to_datetime('today'))
            
            pmo1_cost_pivot = pivot_cost_data(df_cost, "PM01")
            pmo3_cost_pivot = pivot_cost_data(df_cost, "PM03")
            summary_data = process_master_counter_data(df_master_counter)
            merged_pivots = combine_cost_smu_pivots(pivot_smu, pmo1_cost_pivot, pmo3_cost_pivot, summary_data, unit_numbers)
            merged_pivots.to_csv('merged_pivots_test.csv', index=False)

            coeff_PMO1, Rsquare_PMO1 = smu_cost_fit(merged_pivots, "X")
            coeff_PMO3, Rsquare_PMO3 = smu_cost_fit(merged_pivots, "Y")
            
            current_hours = summary_data.set_index('MeasPosition')['Max_Hour_Reading'].to_dict()
            average_daily_hours = summary_data.set_index('MeasPosition')['Average_Hours_Per_Day'].to_dict()
            units_data = {unit: {'current_hours': current_hours[unit],
                     'avg_daily_hours': average_daily_hours[unit]}
              for unit in unit_numbers}
            scenarios = next(iter(unit_scenarios.values()))
            coefficients_costX = coeff_PMO1
            coefficients_costY = coeff_PMO3 
            end_of_life = eol_date

            all_scenarios_forecasts = forecast_all_units_scenarios(current_month, unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life)
            st.session_state['all_scenarios_forecasts'] = all_scenarios_forecasts

            output_csv = 'PMO13forecasts.csv'  # Specify your output CSV file name here
            forecast_all_units_scenarios_to_csv(current_month, unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life, output_csv)

    if st.button("Confirm"):
        if 'replacement_schedule' in st.session_state:
            replacement_schedule = st.session_state['replacement_schedule']
            st.header("Replacement Schedule")
            st.dataframe(replacement_schedule)
            fy_overview = show_fy_overview(replacement_schedule, current_month, eol_date)
            st.header("FY Overview")
            st.write(fy_overview)
        if 'all_scenarios_forecasts' in st.session_state:
            all_scenarios_forecasts = st.session_state['all_scenarios_forecasts']
            
            df = format_forecast_outputs(all_scenarios_forecasts)
            st.header("PMO1 and PMO3 Cost Forecast")
            st.dataframe(df)
            
if __name__ == "__main__":
    conn = init_connection()
    if conn is not None:
        POSTGRES = 1
    else:
        POSTGRES = 0
    main(POSTGRES)
