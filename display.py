## LEGEND ##
# IW38 = cost
# IP24/18/19 = strategy
# IK17 = counter

#import necessary libraries
import ETL
import calc
import helper

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def page_setup():
    st.set_page_config(layout="wide")
    st.title("Tundra Resource Analytics - Equipment Strategy Optimization Tool")
    current_month = pd.Timestamp('2024-04-13')

    st.header("User Input")
    return current_month

#function to get the user inputs
def get_user_input(POSTGRES):
    st.cache_data(ttl=600)
    #set up sidebar in streamlit
    st.sidebar.header("Input Assumptions")
    uploaded_fleet_list = None
    uploaded_linked_strategy = None
    df_fleets = None
    if POSTGRES == False:
        #Connection failed, request file uploads
        st.warning("Could not connect to the SQL database. Please upload the extra required files.")
        st.sidebar.subheader("Database file uploads")
        uploaded_fleet_list = st.sidebar.file_uploader("Upload Fleet List", type=["csv", "xlsx"])
        uploaded_linked_strategy = st.sidebar.file_uploader("Upload Linked Strategy", type=["csv", "xlsx"])
        # Extract unique fleet names
        df_fleets = helper.read_file(uploaded_fleet_list)
        fleets = df_fleets['Fleet'].unique()
    else:
        query = f"SELECT DISTINCT fleet FROM fleet_list WHERE fleet IS NOT NULL ORDER BY fleet;"
        fleets = helper.run_query(query)
        fleets = [fleet[0] for fleet in fleets if fleet[0]] 
    # Assuming 'sorted_fleets' contains your unique, sorted fleet names
    fleet_options = [""] + fleets

    #list fleet options and select required fleet
    selected_fleet = st.sidebar.selectbox("Choose Fleet Option", fleet_options, index=0)
    #determine unit numbers based on chosen fleet

    unit_numbers, rep_cost = ETL.get_unit_numbers_by_fleet(selected_fleet, POSTGRES, df_fleets)#uploaded_fleet_list)
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
    return unit_scenarios, cost_data, component_counter_data, master_counter_data, selected_fleet, eol_date, unit_numbers, uploaded_linked_strategy, rep_cost


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
        # st.header("Component Counter Data")
        # st.data_editor(merged_data)
        st.header("Component Counter Data with Maintenance Interval in Hours")
        st.data_editor(merged_data)


#configuring the output page, print npv for each scenario
def output_page(unit_scenarios):
    st.title("Output Page")
    st.header("Scenario NPV")
    for unit, scenarios in unit_scenarios.items():
        st.subheader(f"Unit {unit}")
        for scenario, replacement_hours in scenarios.items():
            npv_value = calc.calculate_npv(replacement_hours)
            st.write(f"{scenario}: NPV - ${npv_value:.2f}")

def load_and_display_data(fleet_input, unit_scenarios, cost_data, component_counter_data, master_counter_data, uploaded_linked_strategy):
    if not unit_scenarios and not cost_data:
        st.info("Please choose a fleet option and upload the cost data to view Master Strategy Data with Average Cost.")
    elif not cost_data:
        st.info("Please upload the cost data.")
    elif not unit_scenarios:
        st.info("Please choose a fleet option.")
    elif not component_counter_data:
        st.info("Please upload the IW17 component data.")
    elif not master_counter_data:
        st.info("Please upload the IW17 master data.")
    else:
        df_cost = helper.read_file(cost_data)
        df_component_counter = helper.read_file(component_counter_data)
        df_master_counter = helper.read_file(master_counter_data)
        pivot_smu = ETL.pivot_master_counter_data(df_master_counter)
        st.success("All data uploaded successfully!")
        df_complete = ETL.merge_all_data(fleet_input, unit_scenarios, df_cost, df_component_counter, df_master_counter, uploaded_linked_strategy)
        st.session_state['df_complete'] = df_complete  # Save DataFrame to session state
    return df_cost, df_component_counter, df_master_counter, pivot_smu

def process_data(df_complete, current_month, eol_date, df_cost, df_master_counter, unit_numbers, unit_scenarios, pivot_smu, rep_cost):

    replacement_schedule = None
    replacement_schedule = calc.create_PMO2_replacement_schedule(df_complete, current_month, eol_date)
    st.session_state['replacement_schedule'] = replacement_schedule

    pmo1_cost_pivot = ETL.pivot_cost_data(df_cost, "PM01")
    pmo3_cost_pivot = ETL.pivot_cost_data(df_cost, "PM03")
    summary_data = calc.process_master_counter_data(df_master_counter)
    merged_pivots = ETL.combine_cost_smu_pivots(pivot_smu, pmo1_cost_pivot, pmo3_cost_pivot, summary_data, unit_numbers)
    merged_pivots.to_csv('merged_pivots_test.csv', index=False)

    coeff_PMO1, Rsquare_PMO1 = calc.smu_cost_fit(merged_pivots, "PM01")
    coeff_PMO3, Rsquare_PMO3 = calc.smu_cost_fit(merged_pivots, "PM03")
        
    current_hours = summary_data.set_index('MeasPosition')['Max_Hour_Reading'].to_dict()
    average_daily_hours = summary_data.set_index('MeasPosition')['Average_Hours_Per_Day'].to_dict()
    units_data = {unit: {'current_hours': current_hours[unit],
                'avg_daily_hours': average_daily_hours[unit]}
        for unit in unit_numbers}
    scenarios = next(iter(unit_scenarios.values()))
    coefficients_costX = coeff_PMO1
    coefficients_costY = coeff_PMO3 
    end_of_life = eol_date

    all_scenarios_forecasts = calc.forecast_all_units_scenarios(current_month, unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life)
    st.session_state['all_scenarios_forecasts'] = all_scenarios_forecasts

    output_csv = 'PMO13forecasts.csv'  # Specify your output CSV file name here
    calc.forecast_all_units_scenarios_to_csv(current_month, unit_numbers, units_data, scenarios, coefficients_costX, coefficients_costY, end_of_life, output_csv)

    #if 'replacement_schedule' in st.session_state:
    replacement_schedule = st.session_state['replacement_schedule']
    st.header("Replacement Schedule")
    st.dataframe(replacement_schedule)
    fy_overview = calc.show_fy_overview(replacement_schedule, current_month, eol_date)
    st.header("FY Overview")
    st.write(fy_overview)
    #if 'all_scenarios_forecasts' in st.session_state:
    all_scenarios_forecasts = st.session_state['all_scenarios_forecasts']
    
    df, df_long_format = calc.format_forecast_outputs(all_scenarios_forecasts)
    st.header("PMO1 and PMO3 Cost Forecast")
    st.dataframe(df)
    #PM01_3_FY_summary = calc.fy_summary_PM01_3(df_long_format, rep_cost)
    #st.write(PM01_3_FY_summary)
    st.success("Data processing complete!")