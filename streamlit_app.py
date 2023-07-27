# streamlit_app.py

import streamlit as st
import psycopg2
import pandas as pd
from io import StringIO

# Initialize connection.
# Uses st.cache_resource to only run once.
st.set_page_config(layout="wide")
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

conn = init_connection()

# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

def get_unit_numbers_by_fleet(fleet_name):
    # Query the database to fetch unit numbers based on the fleet name
    query = f"SELECT unit FROM masterdata WHERE fleet = '{fleet_name}';"
    result = run_query(query)
    return [row[0] for row in result]

def get_user_input():
    st.subheader("Input Fleet Options")

    # Choose fleet options
    fleet_options = [
        "",
        "Excavator - Large",
        "Excavator - Large H1200",
        "Excavator - Large H2600",
        "D11R Dozers",
        "D11 Dozers",
        "D10 Dozers",
        "992 Loaders",
        "993 Loaders",
        "777F Haul Trucks",
        "777F Water Trucks",
        "Floats",
        "Graders",
        "Support Dozers",
        "Scrub Dozer",
        "Drills",
        "Road Trains"
    ]
    selected_fleet = st.selectbox("Choose Fleet Option", fleet_options, index=0)
    unit_numbers = get_unit_numbers_by_fleet(selected_fleet)
    if unit_numbers:
        st.write(f"Unit Numbers: {unit_numbers}")
    # Input scenarios for each unit number
    st.subheader("Scenarios")

    num_strategies = st.selectbox("Number of Strategies", list(range(1, 11)), index=2)

    strategy_hours = {}
    for i in range(1, num_strategies + 1):
        strategy_name = f"Scenario {i}"
        strategy_hours[strategy_name] = st.number_input(f"{strategy_name} - Replacement Hours", min_value=0, value=20000, step=1000)

    # File uploads
    st.subheader("File Uploads")
    col1,col2,col3=st.columns(3)
    iw38_data = col1.file_uploader("Upload IW38 Data", type=["csv", "xlsx"])
    ik17_component_data = col2.file_uploader("Upload IK17 Component Data", type=["csv", "xlsx"])
    ik17_master_data = col3.file_uploader("Upload IK17 Master Data", type=["csv", "xlsx"])

    unit_scenarios = {}
    
    for unit_number in unit_numbers:
        unit_scenarios[unit_number] = strategy_hours

    return unit_scenarios, iw38_data, ik17_component_data, ik17_master_data, selected_fleet

def calculate_npv(scenario_hours):
    # Placeholder for NPV calculation, replace with your actual NPV logic
    return 123456.78

def output_page(unit_scenarios):
    st.title("Output Page")
    st.header("Scenario NPV")
    for unit, scenarios in unit_scenarios.items():
        st.subheader(f"Unit {unit}")
        for scenario, replacement_hours in scenarios.items():
            npv_value = calculate_npv(replacement_hours)
            st.write(f"{scenario}: NPV - ${npv_value:.2f}")

def read_iw38_data(df):
    # Filter the first column by values "PM02"
    df_filtered = df[df.iloc[:, 0] == "PM02"]

    st.write(df_filtered)

    # Group by maintenance item and calculate average cost greater than 10$
    avg_cost_df = df_filtered[df_filtered["TotSum (actual)"] > 10].groupby("MaintItem")["TotSum (actual)"].mean()

    return avg_cost_df

def get_master_ip24_data(selected_units):
    # Convert the selected units list to a comma-separated string
    unit_numbers_str = ", ".join(f"'{unit}'" for unit in selected_units)

    # Query the database to fetch all entries from the masterip24 table for the selected units
    query = f"SELECT * FROM linkedip24 WHERE unit IN ({unit_numbers_str});"
    result = run_query(query)

    # Convert the result to a pandas DataFrame
    result_df = pd.DataFrame(result, columns=["Unit", "Functional Loc.", "MaintenancePlan", "MaintItem", "MaintItemText", "Desc","MaintItemDesc","MaintItemInterval","ik17component"])
    return result_df

def read_file(file_data):
    if file_data is not None:
        # Check the file extension to determine the file type
        file_extension = file_data.name.split(".")[-1].lower()
        
        # Read the file using pandas based on the file extension
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
        # Assuming 30.75 days in a month
        return int(interval[:-1])*average_usage_per_day * 30.75
    elif time_period == "W":
        # Assuming 7 days in a week
        return int(interval[:-1])*average_usage_per_day * 7
    elif time_period == "D":
        return int(interval[:-1])*average_usage_per_day
    elif time_period == "K":
        # Assuming 100,000 hours in "K" interval
        return 100000
    elif time_period == "Y":
        # Assuming 365.25 days in a year
        return int(interval[:-1])*average_usage_per_day * 365.25
    else:
        raise ValueError(f"Invalid time period: {time_period}. Supported time periods are H, M, W, D, K, and Y.")

def output(fleet_input, unit_scenarios,iw38_df,ik17_component_df,ik17_master_df):
    
    selected_fleet = unit_scenarios.keys()
    iw38_filtered_df = read_iw38_data(iw38_df)
    if iw38_filtered_df is not None:
        st.subheader("Filtered IW38 Data (PM02) with Average Cost")
        st.write(iw38_filtered_df)
        fleet_ip24_data = get_master_ip24_data(selected_fleet)
        st.header(f"{fleet_input} IP24 Data (Filtered by Unit Numbers)")
        fleet_ip24_data = st.data_editor(fleet_ip24_data)
    else: 
        return
    if fleet_ip24_data is not None:
            # Merge the calculated average cost from IW38 data into masterip24 data
            fleet_ip24_data = pd.merge(fleet_ip24_data, iw38_filtered_df, on="MaintItem", how="left")
            st.subheader(f"{fleet_input} IP24 Data with Average Cost")
            st.data_editor(fleet_ip24_data)
    else:
        return
    
    if ik17_master_df is not None:
         # Filter the relevant columns from the IK17 Master data
        relevant_columns = ["MeasPosition", "Counter reading", "Date"]
        ik17_df = ik17_master_df[relevant_columns].copy()

        # Convert the "Date" column to datetime type
        ik17_df["Date"] = pd.to_datetime(ik17_df["Date"])

        # Group by unit and get the counter reading for the max date and min date
        group_data = ik17_df.groupby("MeasPosition").agg(
            Max_Date_Reading=("Counter reading", "max"),
            Min_Date_Reading=("Counter reading", "min"),
            Min_Date=("Date", "min"),
            Max_Date=("Date", "max")
        )

        # Calculate the hours used between the max date and min date
        group_data["Hours_Used"] = group_data["Max_Date_Reading"] - group_data["Min_Date_Reading"]

        # Calculate the total number of days for each unit
        group_data["Total_Days"] = (group_data["Max_Date"] - group_data["Min_Date"]).dt.days + 1

        # Calculate the average hours per day
        group_data["Average_Hours_Per_Day"] = group_data["Hours_Used"] / group_data["Total_Days"]

        # Reset the index for the DataFrame
        group_data.reset_index(inplace=True)
        st.write(group_data)
    else:
        return
    if ik17_component_df is not None:
        # Merge ip24_data and ik17_component_df based on matching columns
        ik17_component_df.rename(columns={"Description": "ik17component"}, inplace=True)
        merged_data = pd.merge(fleet_ip24_data, ik17_component_df, on="ik17component", how="left")
        # Rename the "Counter reading" column to avoid overwriting the original column in ip24_data
        ik17_component_df.rename(columns={"Counter reading": "Counter reading (IK17)"}, inplace=True)
        st.header("IK17 Component Data")
        st.data_editor(merged_data)
        #add average hours per day to the merged data
        merged_data = pd.merge(merged_data, group_data[["MeasPosition", "Average_Hours_Per_Day"]], left_on="Unit", right_on="MeasPosition", how="left")

        #calculate the maintenance interval in hours
        merged_data["MaintItemInterval"] = merged_data.apply(lambda row: convert_interval_to_hours(row["MaintItemInterval"], row["Average_Hours_Per_Day"]), axis=1)
        st.header("IK17 Component Data with Maintenance Interval in Hours")
        st.data_editor(merged_data)
        return merged_data        
    else:
        return
    
def create_replacement_schedule(complete_df, current_month, end_month):
    # Generate a list of all months from the current month until June 2037
    all_months = pd.date_range(current_month, end_month, freq='M')
    
    # Create an empty DataFrame to hold the replacement schedule
    replacement_schedule = pd.DataFrame(columns=["Interval", "Usual Days Until Replacement", "unit"] + [month.strftime('%b-%y') for month in all_months])
    
    # Iterate through each row in the complete_df
    for index, row in complete_df.iterrows():
        # Variable for first replacement month TODO: Add user input for first replacement month
        first_replacement_month = None
        # Add the MaintItem and Interval to the replacement schedule
        replacement_schedule.loc[index, "MaintItem"] = row["MaintItem"]
        replacement_schedule.loc[index, "Interval"] = row["MaintItemInterval"]
        replacement_schedule.loc[index, "Unit"] = row["Unit"]  # Add the unit number to the 'unit' column
        # Get MaintItem information (usage, interval, etc.)
        interval = row["MaintItemInterval"]
        # Get usage information
        usage = row["Average_Hours_Per_Day"]
        # Get current counter reading
        current = row["Counter reading"]
        # Calculate normal replacement time
        usual = interval / usage
        if interval < current:
            first_replacement_month = current_month
        else:
            # Calculate the first replacement month
            first_replacement_month = current_month + pd.DateOffset(days=(interval - current) / usage)
        replacement_schedule.loc[index, "Usual Days Until Replacement"] = usual

        # Calculate the replacement month for each month in the all_months list
        for month in all_months:
            month_str = month.strftime('%b-%y')
            if first_replacement_month > end_month:
                # If the first replacement month is after the end month, break the loop
                break
            if month >= first_replacement_month:
                # Calculate the number of days between first_replacement_month and month
                days_difference = (month - first_replacement_month).days
                # Calculate the number of replacements in this month
                replacements_this_month = int(days_difference / usual) + 1
                replacement_schedule.loc[index, month_str] = row["TotSum (actual)"] * replacements_this_month
                # Calculate the next replacement month based on the formula: further_replacements = interval / usage
                further_replacements = int(interval / usage)
                first_replacement_month += pd.DateOffset(days=further_replacements)

    return replacement_schedule


def show_fy_overview(replacement_schedule):
    # Create an empty list to hold the rows
    fy_rows = []

    # Iterate through each unit in the replacement schedule
    for unit in replacement_schedule["Unit"].unique():
        # Filter the replacement schedule for the current unit
        unit_schedule = replacement_schedule[replacement_schedule["Unit"] == unit]

        # Initialize a new row for the unit
        unit_row = {"Unit": unit}

        # Calculate the cost for each fiscal year
        for year in range(23, 33):
            # Define the start and end of the fiscal year
            fy_start = pd.Timestamp(f"20{year-1}-07-01")
            fy_end = pd.Timestamp(f"20{year}-06-30")

            # Get the columns for this fiscal year
            fy_columns = []
            for col in unit_schedule.columns:
                try:
                    date = pd.to_datetime(col, format='%b-%y')
                    if fy_start <= date <= fy_end:
                        fy_columns.append(col)
                except ValueError:
                    continue  # Ignore columns that are not dates

            # Sum up the costs for this fiscal year
            unit_row[f"FY{year}"] = unit_schedule[fy_columns].sum(axis=1).sum()

        # Calculate the total cost for all fiscal years
        unit_row["Total (NOMINAL)"] = sum(unit_row[f"FY{year}"] for year in range(23, 33))

        # Append the row to the fy_rows list
        fy_rows.append(pd.Series(unit_row))

    # Convert the list of rows into a DataFrame
    fy_overview = pd.concat(fy_rows, axis=1).transpose()

    return fy_overview

def main():
    st.title("Tundra Resource Analytics - Equipment Strategy Optimization Tool")

    # Get user input
    unit_scenarios, iw38_data, ik17_component_data, ik17_master_data, fleet_input = get_user_input()

    # Display the user input
    st.header("User Input")
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
        else:
            st.success("IW39 Data uploaded successfully!")
            # File upload status
            if ik17_component_data:
                st.success("IK17 Component Data uploaded successfully!")
            if ik17_master_data:
                st.success("IK17 Master Data uploaded successfully!")
            complete_df = output(fleet_input, unit_scenarios,iw38_df,ik17_component_df,ik17_master_df)
            if complete_df is not None:
                st.header("Complete Data")
                st.dataframe(complete_df)
                current_month = pd.Timestamp("2023-07-01")
                end_month = pd.Timestamp("2037-06-30")
                replacement_schedule = create_replacement_schedule(complete_df, current_month, end_month)
                st.header("Replacement Schedule")
                st.dataframe(replacement_schedule)
                fy_overview = show_fy_overview(replacement_schedule)
                st.header("FY Overview")
                st.write(fy_overview)
        
    # Show output page when the user clicks the button
    if st.button("Show Output"):
        output_page(unit_scenarios)

if __name__ == "__main__":
    main()
