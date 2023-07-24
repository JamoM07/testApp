# streamlit_app.py

import streamlit as st
import psycopg2
import pandas as pd
from io import StringIO

# Initialize connection.
# Uses st.cache_resource to only run once.

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
    st.header("Input Fleet and Scenarios")

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
    # Input scenarios for each unit number
    st.subheader("Scenarios")

    num_strategies = st.selectbox("Number of Strategies", list(range(1, 11)), index=2)

    strategy_hours = {}
    for i in range(1, num_strategies + 1):
        strategy_name = f"Scenario {i}"
        strategy_hours[strategy_name] = st.number_input(f"{strategy_name} - Replacement Hours", min_value=0, value=20000, step=1000)

    # File uploads
    st.subheader("File Uploads")
    iw38_data = st.file_uploader("Upload IW38 Data", type=["csv", "xlsx"])
    ik17_component_data = st.file_uploader("Upload IK17 Component Data", type=["csv", "xlsx"])
    ik17_master_data = st.file_uploader("Upload IK17 Master Data", type=["csv", "xlsx"])

    unit_scenarios = {}
    unit_numbers = get_unit_numbers_by_fleet(selected_fleet)
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
    if ik17_component_df is not None:
        # Merge ip24_data and ik17_component_df based on matching columns
        ik17_component_df.rename(columns={"Description": "ik17component"}, inplace=True)
        merged_data = pd.merge(fleet_ip24_data, ik17_component_df, on="ik17component", how="left")
        # Rename the "Counter reading" column to avoid overwriting the original column in ip24_data
        ik17_component_df.rename(columns={"Counter reading": "Counter reading (IK17)"}, inplace=True)

        st.header("IK17 Component Data")
        st.data_editor(merged_data)
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
    
def main():
    st.title("Your Web Application")

    # Get user input
    unit_scenarios, iw38_data, ik17_component_data, ik17_master_data, fleet_input = get_user_input()

    # Display the user input
    st.header("User Input")
    for unit, scenarios in unit_scenarios.items():
        st.write(f"Unit {unit}: {scenarios}")


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
            output(fleet_input, unit_scenarios,iw38_df,ik17_component_df,ik17_master_df)

    # Show output page when the user clicks the button
    if st.button("Show Output"):
        output_page(unit_scenarios)


if __name__ == "__main__":
    main()



        
