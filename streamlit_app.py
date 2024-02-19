import streamlit as st
import psycopg2
import pandas as pd
from io import StringIO

fleet_options = [
    "", "Excavator - Large", "Excavator - Large H1200", "Excavator - Large H2600", "D11R Dozers", "D11 Dozers", "D10 Dozers", "992 Loaders", "993 Loaders", "777F Haul Trucks", "777F Water Trucks", "Floats", "Graders", "Support Dozers", "Scrub Dozer", "Drills", "Road Trains"
]

st.set_page_config(layout="wide")

def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

@st.cache_data(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

def get_unit_numbers_by_fleet(fleet_name):
    query = f"SELECT unit FROM masterdata WHERE fleet = '{fleet_name}';"
    result = run_query(query)
    return [row[0] for row in result]

def get_user_input():
    st.sidebar.header("Input Assumptions")
    selected_fleet = st.sidebar.selectbox("Choose Fleet Option", fleet_options, index=0)
    unit_numbers = get_unit_numbers_by_fleet(selected_fleet)
    if unit_numbers:
        st.sidebar.write(f"Unit Numbers: {unit_numbers}")
    eol_date = st.sidebar.date_input("Choose End of Life (EOL) Date", value=pd.to_datetime("2037-06-30"))
    st.sidebar.subheader("Scenarios")
    num_strategies = st.sidebar.selectbox("Number of Strategies", list(range(1, 11)), index=2)
    strategy_hours = {}
    for i in range(1, num_strategies + 1):
        strategy_name = f"Scenario {i}"
        strategy_hours[strategy_name] = st.sidebar.number_input(f"{strategy_name} - Replacement Hours", min_value=0, value=20000, step=1000)
    st.sidebar.subheader("File Uploads")
    iw38_data = st.sidebar.file_uploader("Upload IW38 Data", type=["csv", "xlsx"])
    ik17_component_data = st.sidebar.file_uploader("Upload IK17 Component Data", type=["csv", "xlsx"])
    ik17_master_data = st.sidebar.file_uploader("Upload IK17 Master Data", type=["csv", "xlsx"])
    unit_scenarios = {}
    for unit_number in unit_numbers:
        unit_scenarios[unit_number] = strategy_hours
    eol_date = pd.Timestamp(eol_date)
    return unit_scenarios, iw38_data, ik17_component_data, ik17_master_data, selected_fleet, eol_date

def calculate_npv(scenario_hours):
    return 123456.78

def output_page(unit_scenarios):
    st.title("Output Page")
    st.header("Scenario NPV")
    for unit, scenarios in unit_scenarios.items():
        st.subheader(f"Unit {unit}")
        for scenario, replacement_hours in scenarios.items():
            npv_value = calculate_npv(replacement_hours)
            st.write(f"{scenario}: NPV - ${npv_value:.2f}")

def read_iw38_data_pm02(df):
    df_filtered = df[df.iloc[:, 0] == "PM02"]
    st.write(df_filtered)
    avg_cost_df = df_filtered[df_filtered["TotSum (actual)"] > 10].groupby("MaintItem")["TotSum (actual)"].mean()
    return avg_cost_df

def get_master_ip24_data(selected_units):
    unit_numbers_str = ", ".join(f"'{unit}'" for unit in selected_units)
    query = f"SELECT * FROM linkedip24 WHERE unit IN ({unit_numbers_str});"
    result = run_query(query)
    result_df = pd.DataFrame(result, columns=["Unit", "Functional Loc.", "MaintenancePlan", "MaintItem", "MaintItemText", "Desc","MaintItemDesc","MaintItemInterval","ik17component"])
    return result_df

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

def convert_interval_to_hours(interval, average_usage_per_day):
    time_period = interval[-1]
    if time_period == "H":
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
        raise ValueError(f"Invalid time period: {time_period}. Supported time periods are H, M, W, D, K, and Y.")

def filter_iw38_data_pm02(iw38_df, selected_fleet):
    iw38_filtered_df = read_iw38_data_pm02(iw38_df)
    fleet_ip24_data = get_master_ip24_data(selected_fleet)
    if fleet_ip24_data is not None:
        fleet_ip24_data = pd.merge(fleet_ip24_data, iw38_filtered_df, on="MaintItem", how="left")
    return iw38_filtered_df, fleet_ip24_data

def process_ik17_master_data(ik17_master_df):
    relevant_columns = ["MeasPosition", "Counter reading", "Date"]
    ik17_df = ik17_master_df[relevant_columns].copy()
    ik17_df["Date"] = pd.to_datetime(ik17_df["Date"])
    group_data = ik17_df.groupby("MeasPosition").agg(
        Max_Date_Reading=("Counter reading", "max"),
        Min_Date_Reading=("Counter reading", "min"),
        Min_Date=("Date", "min"),
        Max_Date=("Date", "max")
    )
    group_data["Hours_Used"] = group_data["Max_Date_Reading"] - group_data["Min_Date_Reading"]
    group_data["Total_Days"] = (group_data["Max_Date"] - group_data["Min_Date"]).dt.days + 1
    group_data["Average_Hours_Per_Day"] = group_data["Hours_Used"] / group_data["Total_Days"]
    group_data.reset_index(inplace=True)
    return group_data

def merge_ik17_data(fleet_ip24_data, ik17_component_df, group_data):
    ik17_component_df.rename(columns={"Description": "ik17component"}, inplace=True)
    merged_data = pd.merge(fleet_ip24_data, ik17_component_df, on="ik17component", how="left")
    ik17_component_df.rename(columns={"Counter reading": "Counter reading (IK17)"}, inplace=True)
    merged_data = pd.merge(merged_data, group_data[["MeasPosition", "Average_Hours_Per_Day"]], left_on="Unit", right_on="MeasPosition", how="left")
    merged_data["MaintItemInterval"] = merged_data.apply(lambda row: convert_interval_to_hours(row["MaintItemInterval"], row["Average_Hours_Per_Day"]), axis=1)
    return merged_data

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

def forecast_monthly_costs(iw39_df):
    # Placeholder for the forecasting method
    # For demonstration purposes, we'll assume a simple forecasting method
    # that multiplies the current cost by a fixed factor for each month.
    # Replace this with your actual forecasting logic.
    return

def pm13_iw39_data(df):
    if df is not None:
        # Filter the data for PM01 and PM03 in the "Order Type" column
        df_filtered = df[df["Order Type"].isin(["PM01", "PM03"])]
        return df_filtered
    else:
        return None

def process_data(fleet_input, unit_scenarios, iw38_df, ik17_component_df, ik17_master_df):
    selected_fleet = unit_scenarios.keys()
    iw38_filtered_df, fleet_ip24_data = filter_iw38_data_pm02(iw38_df, selected_fleet)
    if iw38_filtered_df is None:
        return
    group_data = process_ik17_master_data(ik17_master_df) if ik17_master_df is not None else None
    merged_data = merge_ik17_data(fleet_ip24_data, ik17_component_df, group_data) if ik17_component_df is not None else None
    display_data(fleet_input, iw38_filtered_df, fleet_ip24_data, group_data, merged_data)
    return merged_data

def main():
    st.title("Tundra Resource Analytics - Equipment Strategy Optimization Tool")
    current_month = pd.Timestamp("2023-07-01")
    unit_scenarios, iw38_data, ik17_component_data, ik17_master_data, fleet_input, eol_date = get_user_input()
    st.header("User Input")
    pm01_pm03_df = None
    for unit, scenarios in unit_scenarios.items():
        st.write(f"Unit {unit}: {scenarios}")
    replacement_schedule = None
    if st.button("Show"):
        iw38_df = read_file(iw38_data)
        ik17_component_df = read_file(ik17_component_data)
        ik17_master_df = read_file(ik17_master_data)
        selected_fleet = unit_scenarios.keys()
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
            st.success("All data uploaded successfully!")
            complete_df = process_data(fleet_input, unit_scenarios,iw38_df,ik17_component_df,ik17_master_df)
            pm01_pm03_df = pm13_iw39_data(iw38_df)
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
    if st.button("Confirm"):
        if 'replacement_schedule' in st.session_state:
            replacement_schedule = st.session_state['replacement_schedule']
            st.header("Replacement Schedule")
            st.dataframe(replacement_schedule)
            fy_overview = show_fy_overview(replacement_schedule, current_month, eol_date)
            st.header("FY Overview")
            st.write(fy_overview)

if __name__ == "__main__":
    conn = init_connection()
    main()
