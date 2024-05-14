# TUNDRA ESO Application code v1
# Written by Jamieson Mulready and Samantha McMaster

""" 
LEGEND for SAP data
        IW38 = cost
        IP24/18/19 = strategy
        IK17 = counter
"""

#import necessary libraries
import ETL
import calc
import helper
import display

import streamlit as st
import pandas as pd
from pandas.tseries.offsets import MonthBegin

def main(POSTGRES):
    # setup and connect to database or add manual inputs
    display.page_setup()
    uploaded_fleet_list = None
    uploaded_linked_strategy = None
    if POSTGRES == False:
        fleets, uploaded_fleet_list, uploaded_linked_strategy = display.manual_inputs()
    else:
        fleets = display.database_inputs()

    # get user inputs: scenarios, EOL date, datasets
    unit_scenarios, cost_data, component_counter_data, master_counter_data, fleet_input, eol_date, unit_numbers, repl_cost = display.get_user_input(fleets, uploaded_fleet_list)

    # display units and scenarios
    for unit, scenarios in unit_scenarios.items():
        st.write(f"Unit {unit}: {scenarios}")

    # load and display data if 'Show' button is clicked 
    if st.button("Show Data"):
        # show, input and check all the required files
        df_cost, df_master_counter, df_component_counter = display.load_data(fleet_input, unit_scenarios, cost_data, component_counter_data, master_counter_data, uploaded_linked_strategy)
        # process, merge and format all the data for calculations
        summary_data, merged_pivots, df_cost_filtered_PM02, merged_data, fleet_strategy_data = ETL.main(df_cost, df_master_counter, unit_numbers, uploaded_linked_strategy, df_component_counter)
        # display the formatted input data
        display.display_data(fleet_input, df_cost_filtered_PM02, fleet_strategy_data, summary_data, merged_data)
        # perform all the calculations; replacement scheduling and forecasting etc.
        replacement_schedule, formatted_forecasts, formatted_forecasts_long, pm02fy_overviews, pm01_3_fy_overviews = calc.main(merged_data, current_month, eol_date, unit_numbers, unit_scenarios, repl_cost, merged_pivots, summary_data, df_master_counter)
        # display outputs; replacement schedules and forecasts as well as summaries of results
        display.display_outputs(replacement_schedule, formatted_forecasts, formatted_forecasts_long, pm02fy_overviews, pm01_3_fy_overviews)

# read in current month
now = pd.Timestamp.now().date()
current_month = pd.to_datetime(now)

# check connection to postgres database and run the main function
if __name__ == "__main__":
    POSTGRES = True if helper.init_connection() is not None else False
    main(POSTGRES)
