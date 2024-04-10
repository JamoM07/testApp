#import necessary libraries
import streamlit as st
import psycopg2
from psycopg2 import OperationalError
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from calendar import monthrange

#type of fleet
# *** adujust for new site ***
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
    #print(f"Financial Month: {financial_month}, Type: {type(financial_month)}")
    #print(f"Financial Year: {financial_year}, Type: {type(financial_year)}")
    fin_month_str = f"{financial_year}-{financial_month:02d}"
    # Return as a period object
    return pd.Period(fin_month_str, freq='M')

#function to convert the maintenance interval to hours using the strategy and the average usage per day
# *** adujust for new site??? ***
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
    

def calculate_costs(smu, coefficients):
    """
    Calculate monthly and cumulative costs based on SMU and coefficients.
    Assuming a quadratic relationship: cost = a * smu^2 + b * smu.
    :param smu: Service Meter Units
    :param coefficients: Coefficients [a, b] for the quadratic cost model
    :return: The cost for the given SMU
    """
    a, b = coefficients
    return a * smu**2 + b * smu

##Setting up
#function for getting unit numbers for the fleet
# *** adujust for new site ***
def get_unit_numbers_by_fleet(fleet_name, POSTGRES):
    query = f"SELECT unit FROM fleet_list WHERE fleet = '{fleet_name}';"
    result_final = []
    if POSTGRES == 0:
        # Connection failed, request file uploads
        #st.warning("Could not connect to the database. Please upload the extra required files.")
        st.sidebar.subheader("Database file uploads")
        uploaded_fleet_list = st.sidebar.file_uploader("Upload Fleet List", type=["csv", "xlsx"])
        uploaded_linked_ip24 = st.sidebar.file_uploader("Upload Linked IP24", type=["csv", "xlsx"])
        
        if uploaded_fleet_list is not None and uploaded_linked_ip24 is not None:
            # Read the uploaded files
            fleet_list = read_file(uploaded_fleet_list)
            result_df = fleet_list[fleet_list['Fleet'] == fleet_name]['Unit']
            result_final = result_df.tolist()
            
            #Process data from uploaded files
            #processed_data = process_data(fleet_list, linked_ip24)
    else:
        uploaded_linked_ip24 = None
        result = run_query(query)
        result_final = [row[0] for row in result]
    return result_final, uploaded_linked_ip24
#function to get the user inputs
# *** adujust for new site ***
def get_user_input(POSTGRES):
    #set up sidebar in streamlit
    st.sidebar.header("Input Assumptions")
    #list fleet options and select required fleet
    selected_fleet = st.sidebar.selectbox("Choose Fleet Option", fleet_options, index=0)
    #determine unit numbers based on chosen fleet
    unit_numbers, uploaded_linked_ip24 = get_unit_numbers_by_fleet(selected_fleet, POSTGRES)
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
    iw38_data = st.sidebar.file_uploader("Upload IW38 Data", type=["csv", "xlsx"])
    ik17_component_data = st.sidebar.file_uploader("Upload IK17 Component Data", type=["csv", "xlsx"])
    ik17_master_data = st.sidebar.file_uploader("Upload IK17 Master Data", type=["csv", "xlsx"])
    unit_scenarios = {}
    #replacement hours for each unit
    for unit_number in unit_numbers:
        unit_scenarios[unit_number] = strategy_hours
    eol_date = pd.Timestamp(eol_date)
    #return the replacement hours for each unit, data files, chosen fleet and end of life date
    return unit_scenarios, iw38_data, ik17_component_data, ik17_master_data, selected_fleet, eol_date, unit_numbers, uploaded_linked_ip24

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
# *** adujust for new site ***
def read_iw38_data_pm02(df):
    df_filtered = df[df.iloc[:, 0] == "PM02"]
    st.write(df_filtered)
    avg_cost_df = df_filtered[df_filtered["TotSum (actual)"] > 10].groupby("MaintItem")["TotSum (actual)"].mean()
    return avg_cost_df

#extract required data for the chosen fleet from master ip24 database
# *** adujust for new site ***
def get_master_ip24_data(selected_units, uploaded_linked_ip24):
    if uploaded_linked_ip24 is not None:
        linked_ip24 = read_file(uploaded_linked_ip24)
        expected_columns = [
            "Unit", "FunctionalLoc", "MaintenancePlan", "MaintItem", "MaintItemText", 
            "Description", "MaintItemDesc", "MaintItemInterval", "ik17component"
        ]
        for column in expected_columns:
            if column not in linked_ip24.columns:
                linked_ip24[column] = pd.NA

        # Reorder DataFrame to match the expected format
        linked_ip24_df = linked_ip24[expected_columns]
        result_df = linked_ip24_df[linked_ip24_df['Unit'].isin(selected_units)]
    else:
        unit_numbers_str = ", ".join(f"'{unit}'" for unit in selected_units)
        query = f"SELECT * FROM linkedip24 WHERE unit IN ({unit_numbers_str});"
        result = run_query(query)
        result_df = pd.DataFrame(result, columns=["Unit", "FunctionalLoc", "MaintenancePlan", "MaintItem", "MaintItemText", "Description","MaintItemDesc","MaintItemInterval","ik17component"])
    return result_df

#combine required data from iw38 and master ip24
# *** adujust for new site ***
def filter_iw38_data_pm02(iw38_df, selected_fleet, uploaded_linked_ip24):
    iw38_filtered_df_PMO2 = read_iw38_data_pm02(iw38_df)
    fleet_ip24_data = get_master_ip24_data(selected_fleet, uploaded_linked_ip24)
    if fleet_ip24_data is not None:
        fleet_ip24_data = pd.merge(fleet_ip24_data, iw38_filtered_df_PMO2, on="MaintItem", how="left")
    return iw38_filtered_df_PMO2, fleet_ip24_data

#filter IW38 for only PMO1 and calculate average cost for each maint item
# *** adujust for new site ***
def process_iw38_data_pm01(df):
    df_filtered = df[df["Order Type"] == "PM01"]
    df_filtered1 = df_filtered[~df_filtered['Description'].str.contains('warranty', case=False, na= False)]
    #ensure date is datetime type
    df_filtered1["Actual Finish"] = pd.to_datetime(df_filtered1["Actual Finish"])
    #convert to fiscal month
    df_filtered1["FinMonth"] = df_filtered1.apply(lambda x: to_financial_month(x["Actual Finish"]) if pd.notnull(x["Actual Finish"]) else to_financial_month(x["Basic fin. date"]), axis=1)
    #create pivot table (needs to use maximum counter reading per month per unit)
    pivot_cost_pmo1 = df_filtered1.pivot_table(values="Total act.costs", index="FinMonth", columns = "Sort field", aggfunc = "sum")
    return pivot_cost_pmo1

#filter IW38 for only PMO1 and calculate average cost for each maint item
# *** adujust for new site ***
def process_iw38_data_pm03(df):
    df_filtered = df[df["Order Type"] == "PM03"]
    df_filtered1 = df_filtered[~df_filtered['Description'].str.contains('warranty', case=False, na= False)]
    #ensure date is datetime type
    df_filtered1["Actual Finish"] = pd.to_datetime(df_filtered1["Actual Finish"])
    #convert to fiscal month
    df_filtered1["FinMonth"] = df_filtered1.apply(lambda x: to_financial_month(x["Actual Finish"]) if pd.notnull(x["Actual Finish"]) else to_financial_month(x["Basic fin. date"]), axis=1)
    #create pivot table (needs to use maximum counter reading per month per unit)
    pivot_cost_pmo3 = df_filtered1.pivot_table(values="Total act.costs", index="FinMonth", columns = "Sort field", aggfunc = "sum")
    return pivot_cost_pmo3

#extract counter reading data from ik17 master
# *** adujust for new site ***
def process_ik17_master_data(ik17_master_df):
    relevant_columns = ["MeasPosition", "Counter reading", "Date"]
    ik17_df = ik17_master_df[relevant_columns].copy()
    ik17_df["Date"] = pd.to_datetime(ik17_df["Date"])
    group_data = ik17_df.groupby("MeasPosition").agg(
        Max_Hour_Reading=("Counter reading", "max"),
        Min_Hour_Reading=("Counter reading", "min"),
        Min_Date=("Date", "min"),
        Max_Date=("Date", "max")
    )
    group_data["Hours_Used"] = group_data["Max_Hour_Reading"] - group_data["Min_Hour_Reading"]
    group_data["Total_Days"] = (group_data["Max_Date"] - group_data["Min_Date"]).dt.days + 1
    group_data["Average_Hours_Per_Day"] = group_data["Hours_Used"] / group_data["Total_Days"]
    group_data.reset_index(inplace=True)
    return group_data

# *** adujust for new site ***
def pivot_ik17_master_data(ik17_master_df):
    #ensure date is datetime type
    ik17_master_df["Date"] = pd.to_datetime(ik17_master_df["Date"])
    #convert to fiscal month
    ik17_master_df["FinMonth"] = ik17_master_df["Date"].apply(to_financial_month)
    #extract month from date
    #ik17_master_df["Month"] = ik17_master_df["FinMonth"].dt.to_period("M")
    #create pivot table (needs to use maximum counter reading per month per unit)
    pivot_smu = ik17_master_df.pivot_table(values="Counter reading", index="FinMonth", columns = "MeasPosition", aggfunc = "max")
    return pivot_smu

# *** adujust for new site ***
#merge ip24 fleet data with ik17 component data
def merge_ik17_data(fleet_ip24_data, ik17_component_df, group_data):
    ik17_component_df.rename(columns={"Measuring point": "ik17component"}, inplace=True)
    merged_data = pd.merge(fleet_ip24_data, ik17_component_df, on="ik17component", how="left")
    ik17_component_df.rename(columns={"Counter reading": "Counter reading (IK17)"}, inplace=True)
    merged_data = pd.merge(merged_data, group_data[["MeasPosition", "Average_Hours_Per_Day"]], left_on="Unit", right_on="MeasPosition", how="left")
    merged_data["MaintItemInterval"] = merged_data.apply(lambda row: convert_interval_to_hours(row["MaintItemInterval"], row["Average_Hours_Per_Day"]), axis=1)
    return merged_data

# *** adujust for new site ***
#merging smu and cost pivots
def combine_pivots(pivot_smu, pivot_cost_PMO1, pivot_cost_PMO3, group_data, unit_numbers):
    pivot_smu_reset = pivot_smu.reset_index()
    pivot_smu_reset.rename(columns={"MeasPosition": "Sort field"}, inplace=True)
    pivot_cost_PMO1_reset = pivot_cost_PMO1.reset_index()
    pivot_cost_PMO3_reset = pivot_cost_PMO3.reset_index()
    smu_melted = pivot_smu_reset.melt(id_vars=["FinMonth"], var_name="Sort field", value_name="Counter reading")
    cost_PMO1_melted = pivot_cost_PMO1_reset.melt(id_vars=["FinMonth"], var_name="Sort field", value_name="Total act.costs")
    cost_PMO3_melted = pivot_cost_PMO3_reset.melt(id_vars=["FinMonth"], var_name="Sort field", value_name="Total act.costs")
    merged_pivots1 = pd.merge(smu_melted, cost_PMO1_melted, on=["FinMonth", "Sort field"], how = "outer")
    merged_pivots = pd.merge(merged_pivots1, cost_PMO3_melted, on=["FinMonth", "Sort field"], how = "outer")
    merged_pivots["Round SMU"] = (merged_pivots["Counter reading"] / 1000).round() * 1000
    #
    cumulative, machine_stats = cumulative_costs(merged_pivots, group_data, unit_numbers)
    return cumulative, machine_stats



#outline replacement schedule for each unit
def create_replacement_schedule(complete_df, current_month, eol_date):
    all_months = pd.date_range(current_month, eol_date, freq='M')
    replacement_schedule = pd.DataFrame(columns=["Interval", "Usual Days Until Replacement", "Unit", "Overdue", "Cost Missing"] + [month.strftime('%b-%y') for month in all_months])
    last_replacement_dates = {}

    for index, row in complete_df.iterrows():
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

# *** adujust for new site ***
def PMO1_fit(merged_pivots):
    pivot_table_max = merged_pivots.pivot_table(values='CumulativeCostX', 
                                                index='Round SMU', 
                                                columns='Sort field', 
                                                aggfunc='max')
    pivot_table_max['AverageCumulativeCostX'] = pivot_table_max.mean(axis=1)
    # Assuming pivot_table_max is the pivot table created earlier
    cleaned_data = pivot_table_max[['AverageCumulativeCostX']].dropna()
    x = cleaned_data.index.values  # SMU values
    y = cleaned_data['AverageCumulativeCostX'].values  # Average cumulative cost values
    # Construct the design matrix for quadratic fit (x^2 and x terms)
    X = np.vstack([x**2, x]).T  # .T to transpose, making it two columns
    # Solve the least squares problem
    coefficients_PMO1, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = coefficients_PMO1[0] * x**2 + coefficients_PMO1[1] * x  # Predicted y values based on the fit
    SS_res = np.sum((y - y_pred)**2)  # Sum of squares of residuals
    SS_tot = np.sum((y - np.mean(y))**2)  # Total sum of squares
    R_squared = 1 - (SS_res / SS_tot)
    return coefficients_PMO1, R_squared

# *** adujust for new site ***
def PMO3_fit(merged_pivots):
    pivot_table_max = merged_pivots.pivot_table(values='CumulativeCostY', 
                                                index='Round SMU', 
                                                columns='Sort field', 
                                                aggfunc='max')
    pivot_table_max['AverageCumulativeCostY'] = pivot_table_max.mean(axis=1)
    # Assuming pivot_table_max is the pivot table created earlier
    cleaned_data = pivot_table_max[['AverageCumulativeCostY']].dropna()
    x = cleaned_data.index.values  # SMU values
    y = cleaned_data['AverageCumulativeCostY'].values  # Average cumulative cost values
    # Construct the design matrix for quadratic fit (x^2 and x terms)
    X = np.vstack([x**2, x]).T  # .T to transpose, making it two columns
    # Solve the least squares problem
    coefficients_PMO3, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = coefficients_PMO3[0] * x**2 + coefficients_PMO3[1] * x  # Predicted y values based on the fit
    SS_res = np.sum((y - y_pred)**2)  # Sum of squares of residuals
    SS_tot = np.sum((y - np.mean(y))**2)  # Total sum of squares
    R_squared = 1 - (SS_res / SS_tot)
    return coefficients_PMO3, R_squared

########################
def forecast_monthly_costs(iw39_df):
    # Placeholder for the forecasting method
    # For demonstration purposes, we'll assume a simple forecasting method
    # that multiplies the current cost by a fixed factor for each month.
    # Replace this with your actual forecasting logic.
    return



#displaying and merging data outputs
def process_data(fleet_input, unit_scenarios, iw38_df, ik17_component_df, ik17_master_df, uploaded_linked_ip24):
    selected_fleet = unit_scenarios.keys()
    iw38_filtered_df, fleet_ip24_data = filter_iw38_data_pm02(iw38_df, selected_fleet, uploaded_linked_ip24)
    if iw38_filtered_df is None:
        return
    group_data = process_ik17_master_data(ik17_master_df) if ik17_master_df is not None else None
    merged_data = merge_ik17_data(fleet_ip24_data, ik17_component_df, group_data) if ik17_component_df is not None else None
    display_data(fleet_input, iw38_filtered_df, fleet_ip24_data, group_data, merged_data)
    return merged_data

# *** adujust for new site ***
#displaying all the extracted data
def display_data(fleet_input, iw38_filtered_df, fleet_ip24_data, group_data, merged_data):
    if iw38_filtered_df is not None and fleet_ip24_data is not None:
        st.subheader("Filtered IW38 Data (PM02) with Average Cost")
        st.write(iw38_filtered_df)
        st.header(f"{fleet_input} IP24 Data (Filtered by Unit Numbers with Average Cost per WO)")
        fleet_ip24_data = st.data_editor(fleet_ip24_data)
    elif group_data is not None:
        st.write(group_data)
    elif merged_data is not None:
        st.header("IK17 Component Data")
        st.data_editor(merged_data)
        st.header("IK17 Component Data with Maintenance Interval in Hours")
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





# *** adujust for new site ***
def cumulative_costs(merged_pivots, group_data, unit_numbers): #### maybe not any use yet
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


    return merged_pivots, machine_stats

# *** adujust for new site ***
def forecast_unit_costs(start_month, end_of_life, current_hours, avg_hours_per_day, replacement_hours, coefficients_costX, coefficients_costY):
    months = pd.date_range(start=start_month, end=end_of_life, freq='MS')
    operating_hours = 0
    cumulative_hours = current_hours
    cumulative_costX = cumulative_costY = 0
    monthly_costX = monthly_costY = 0
    forecast_data = []

    for month in months:
        # Get the number of days in the month
        days_in_month = pd.Period(month, freq='M').days_in_month

        if cumulative_hours >= replacement_hours:
            # If the unit needs replacement, record the current month and break
            replacement_start_month = month
            break

        operating_hours = avg_hours_per_day * days_in_month
        cumulative_hours += operating_hours

        new_cumulative_costX = coefficients_costX[0] * cumulative_hours ** 2 + coefficients_costX[1] * cumulative_hours
        new_cumulative_costY = coefficients_costY[0] * cumulative_hours ** 2 + coefficients_costY[1] * cumulative_hours

        if month == start_month:
            monthly_costX = 0
            monthly_costY = 0
        else:
            monthly_costX = new_cumulative_costX - cumulative_costX
            monthly_costY = new_cumulative_costY - cumulative_costY

        cumulative_costX = new_cumulative_costX
        cumulative_costY = new_cumulative_costY

        forecast_data.append({
            'Month': month.strftime('%Y-%m'),
            'Operating Hours': operating_hours,
            'Cumulative Hours': cumulative_hours,
            'Cumulative CostX': cumulative_costX,
            'Cumulative CostY': cumulative_costY,
            'Monthly CostX': monthly_costX,
            'Monthly CostY': monthly_costY
        })
    else:
        # If loop did not break, all months are forecasted, and no replacement is needed
        replacement_start_month = None

    return forecast_data, replacement_start_month

def calculate_average_daily_hours(units_data):
    total_hours = sum(data['avg_daily_hours'] for data in units_data.values())
    return total_hours / len(units_data)

def forecast_for_scenario(unit, units_data, start_month, scenario_name, scenario_replacement_hours, coefficients_costX, coefficients_costY, end_of_life):
    current_hours = units_data[unit]['current_hours']
    avg_hours_per_day = units_data[unit]['avg_daily_hours']
    
    forecast, replacement_start_month = forecast_unit_costs(
        start_month, end_of_life, current_hours, avg_hours_per_day,
        scenario_replacement_hours, coefficients_costX, coefficients_costY
    )
    
    return forecast, replacement_start_month

def forecast_all_units_scenarios(unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life):
    all_scenarios_forecasts = {}
    average_daily_hours = calculate_average_daily_hours(units_data)
    start_month = pd.Timestamp('2024-04-01')  # Starting month (adjust if needed)

    for scenario_name, replacement_hours in scenarios.items():
        scenario_forecasts = {}
        
        for unit in unit_numbers:
            forecast, replacement_start_month = forecast_for_scenario(
                unit, units_data, start_month, scenario_name, replacement_hours,
                coefficients_costX, coefficients_costY, end_of_life
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
                    replacement_hours, coefficients_costX, coefficients_costY
                )
        
        all_scenarios_forecasts[scenario_name] = scenario_forecasts

    return all_scenarios_forecasts

def forecast_all_units_scenarios_to_csv(unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life, output_csv):
    all_scenarios_forecasts = forecast_all_units_scenarios(unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life)
    
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
def main(POSTGRES):
    st.title("Tundra Resource Analytics - Equipment Strategy Optimization Tool")
    current_month = pd.Timestamp("2024-03-13")

    unit_scenarios, iw38_data, ik17_component_data, ik17_master_data, fleet_input, eol_date, unit_numbers, uploaded_linked_ip24 = get_user_input(POSTGRES)
    st.header("User Input")
    pm01_pm03_df = None
    for unit, scenarios in unit_scenarios.items():
        st.write(f"Unit {unit}: {scenarios}")
    replacement_schedule = None
    if st.button("Show"):
        
        selected_fleet = unit_scenarios.keys()
        # *** adujust for new site ***
        if not selected_fleet and not iw38_data:
            st.info("Please choose a fleet option and upload the IW38 data to view Master IP24 Data with Average Cost.")
        elif not iw38_data:
            st.info("Please upload the IW38 data.")
        elif not selected_fleet:
            st.info("Please choose a fleet option.")   
        elif not ik17_component_data:
            st.info("Please upload the IW17 component data.")
        elif not ik17_master_data:
            st.info("Please upload the IW17 master data.")
        else:
            iw38_df = read_file(iw38_data)
            ik17_component_df = read_file(ik17_component_data)
            ik17_master_df = read_file(ik17_master_data)
            pivot_smu = pivot_ik17_master_data(ik17_master_df)
            print(pivot_smu)
            st.success("All data uploaded successfully!")
            complete_df = process_data(fleet_input, unit_scenarios,iw38_df,ik17_component_df,ik17_master_df, uploaded_linked_ip24)
            ##pm01_pm03_df = pm13_iw39_data(iw38_df)
            if complete_df is not None:
                st.header("Complete Data")
                st.dataframe(complete_df)
                replacement_schedule = create_replacement_schedule(complete_df, current_month, eol_date)
                st.session_state['replacement_schedule'] = replacement_schedule
                cost_missing_indices = replacement_schedule[replacement_schedule["Cost Missing"]].index
                for idx in cost_missing_indices:
                    replacement_schedule.loc[idx, "TotSum (actual)"] = st.number_input(f"Enter cost for maintenance item {replacement_schedule.loc[idx, 'MaintItem']}:", value=0.0)
                overdue_indices = replacement_schedule[replacement_schedule["Overdue"]].index
                for idx in overdue_indices:
                    replacement_schedule.loc[idx, "First Replacement Month"] = st.date_input(f"Enter first replacement date for overdue component {replacement_schedule.loc[idx, 'MaintItem']}:", value=pd.to_datetime('today'))
            
            pmo1_cost_pivot = process_iw38_data_pm01(iw38_df)
            print(pmo1_cost_pivot)
            pmo3_cost_pivot = process_iw38_data_pm03(iw38_df)
            print(pmo3_cost_pivot)
            #################
            group_data = process_ik17_master_data(ik17_master_df)
            print(group_data)
            print(group_data["Max_Hour_Reading"])
            merged_pivots, machine_stats = combine_pivots(pivot_smu, pmo1_cost_pivot, pmo3_cost_pivot, group_data, unit_numbers)
            print(merged_pivots)
            merged_pivots.to_csv('merged_pivots_test.csv', index=False)

            coeff_PMO1, Rsquare_PMO1 = PMO1_fit(merged_pivots)
            print("PMO1 Coeff = ", coeff_PMO1 , "Rsquare = ", Rsquare_PMO1)
            coeff_PMO3, Rsquare_PMO3 = PMO3_fit(merged_pivots)
            print("PMO3 Coeff = ", coeff_PMO3 , "Rsquare = ", Rsquare_PMO3)
            #############################

            # *** adujust for new site ***
            current_hours = group_data.set_index('MeasPosition')['Max_Hour_Reading'].to_dict()
            average_daily_hours = group_data.set_index('MeasPosition')['Average_Hours_Per_Day'].to_dict()
            units_data = {unit: {'current_hours': current_hours[unit],
                     'avg_daily_hours': average_daily_hours[unit]}
              for unit in unit_numbers}
            scenarios = next(iter(unit_scenarios.values()))
            coefficients_costX = coeff_PMO1  # Replace with actual coefficients
            coefficients_costY = coeff_PMO3  # Replace with actual coefficients
            end_of_life = eol_date

            all_scenarios_forecasts = forecast_all_units_scenarios(unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life)
            st.session_state['all_scenarios_forecasts'] = all_scenarios_forecasts

            output_csv = 'PMO13forecasts.csv'  # Specify your output CSV file name here
            forecast_all_units_scenarios_to_csv(unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life, output_csv)
            ################################
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
        main(POSTGRES)
    else:
        POSTGRES = 0
        main(POSTGRES)
