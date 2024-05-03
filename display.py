# TUNDRA ESO Application code v1
# Written by Jamieson Mulready and Samantha McMaster

# import necessary libraries
import ETL
#import calc
import helper

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# set up streamlit page
def page_setup():
    st.set_page_config(layout="wide")
    st.title("Tundra Resource Analytics - Equipment Strategy Optimization Tool")
    st.header("User Input")
    return

def manual_inputs():
    st.warning("Could not connect to the SQL database. Please upload the extra required files.")
    st.sidebar.subheader("Database file uploads")
    uploaded_fleet_list = st.sidebar.file_uploader("Upload Fleet List", type=["csv", "xlsx"])
    uploaded_linked_strategy = st.sidebar.file_uploader("Upload Linked Strategy", type=["csv", "xlsx"])
    uploaded_fleet_list = helper.read_file(uploaded_fleet_list)
    uploaded_linked_strategy = helper.read_file(uploaded_linked_strategy)
    # Extract unique fleet names
    return uploaded_fleet_list['Fleet'].unique(), uploaded_fleet_list, uploaded_linked_strategy

def database_inputs():
    query = f"SELECT DISTINCT fleet FROM fleet_list WHERE fleet IS NOT NULL ORDER BY fleet;"
    fleets = helper.run_query(query)
    return [fleet[0] for fleet in fleets if fleet[0]]

# get the user inputs and datasets
def get_user_input(fleets, uploaded_fleet_list):
    # cache data for 10 minutes
    st.cache_data(ttl=600)
    #set up sidebar in streamlit
    st.sidebar.header("Input Assumptions")
    fleet_options = [""] + list(fleets)
    #list fleet options and select required fleet
    selected_fleet = st.sidebar.selectbox("Choose Fleet Option", fleet_options, index=0)
    # determine unit numbers based on chosen fleet
    unit_numbers, repl_cost = ETL.get_unit_numbers_and_repl_costs(selected_fleet, uploaded_fleet_list)
    if unit_numbers:
        st.sidebar.write(f"Unit Numbers: {unit_numbers}")
    # input end of life date for the mine
    eol_date = st.sidebar.date_input("Choose End of Life (EOL) Date", value=pd.to_datetime("2027-06-30"))
    st.sidebar.subheader("Scenarios")
    # choose number of strategies/ scenarios you want to run through
    num_strategies = st.sidebar.selectbox("Number of Strategies", list(range(1, 11)), index=2)
    strategy_hours = {}
    # loop through scenario hours based on the number of strategies you want to run
    ### unused looping currently?
    for i in range(1, num_strategies + 1):
        strategy_name = f"Scenario {i}"
        strategy_hours[strategy_name] = st.sidebar.number_input(f"{strategy_name} - Replacement Hours", min_value=0, value=60000, step=20000)
    # upload the data files
    st.sidebar.subheader("File Uploads")
    cost_data = st.sidebar.file_uploader("Upload Cost Data", type=["csv", "xlsx"])
    component_counter_data = st.sidebar.file_uploader("Upload Component Counter Data", type=["csv", "xlsx"])
    master_counter_data = st.sidebar.file_uploader("Upload Master Counter Data", type=["csv", "xlsx"])
    unit_scenarios = {}
    # determine replacement hours for each unit
    for unit_number in unit_numbers:
        unit_scenarios[unit_number] = strategy_hours
    eol_date = pd.Timestamp(eol_date)
    # return all the input data
    return unit_scenarios, cost_data, component_counter_data, master_counter_data, selected_fleet, eol_date, unit_numbers, repl_cost

# reading in the uploaded files and warning if any are missing
def load_data(fleet_input, unit_scenarios, cost_data, component_counter_data, master_counter_data, uploaded_linked_strategy):
    df_cost = None
    df_component_counter = None
    df_master_counter = None
    if not unit_scenarios and not cost_data:
        st.info("Please choose a fleet option and upload the cost data to view master strategy data with average cost.")
    elif not cost_data:
        st.info("Please upload the cost data.")
    elif not unit_scenarios:
        st.info("Please choose a fleet option.")
    elif not component_counter_data:
        st.info("Please upload the component counter data.")
    elif not master_counter_data:
        st.info("Please upload the master counter data.")
    else:
        df_cost = helper.read_file(cost_data)
        df_component_counter = helper.read_file(component_counter_data)
        df_master_counter = helper.read_file(master_counter_data)
        st.success("All data uploaded successfully!")
    return df_cost, df_master_counter, df_component_counter

# displaying all the extracted data
def display_data(fleet_input, df_cost_filtered_PM02, fleet_strategy_data, summary_data, merged_data):
    if df_cost_filtered_PM02 is not None and fleet_strategy_data is not None:
        st.subheader("Filtered Cost Data (PM02) with Average Cost")
        st.write(df_cost_filtered_PM02)
        st.header(f"{fleet_input} Strategy Data (Filtered by Unit Numbers with Average Cost per WO)")
        fleet_strategy_data = st.data_editor(fleet_strategy_data)
    if summary_data is not None:
        st.header("Counter data summary")
        st.write(summary_data)
    if merged_data is not None:
        st.header("Component Counter Data with Maintenance Interval in Hours")
        st.data_editor(merged_data)
    
def display_outputs(replacement_schedule, formatted_forecasts, formatted_forecasts_long, pm02_fy_overview):
    st.header("Replacement Schedule")
    st.dataframe(replacement_schedule)
    #fy_overview = calc.pm02_fy_overview(replacement_schedule, current_month, eol_date)
    st.header("FY Overview")
    st.write(pm02_fy_overview)
    #df, df_long_format = calc.format_forecast_outputs(all_scenarios_forecasts)
    st.header("PMO1 and PMO3 Cost Forecast")
    st.dataframe(formatted_forecasts)
    #PM01_3_FY_summary = calc.fy_summary_PM01_3(formatted_forecasts_long, repl_cost)
    #st.write(PM01_3_FY_summary)
    st.success("Data processing complete!")



    # configuring the output page, print npv for each scenario (NPV not complete yet)
######## currently unused
# need to import calc when this is brought back in
def output_page(unit_scenarios):
    st.title("Output Page")
    st.header("Scenario NPV")
    for unit, scenarios in unit_scenarios.items():
        st.subheader(f"Unit {unit}")
        for scenario, replacement_hours in scenarios.items():
            npv_value = calc.calculate_npv(replacement_hours)
            st.write(f"{scenario}: NPV - ${npv_value:.2f}")


