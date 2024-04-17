# clean up comments and old/ unused code
# later: pmo2 on same timeline as pmo1/3, npv and final outputs/ summaries

#import necessary libraries
import ETL
import calc
import helper
import display

import streamlit as st
import pandas as pd
from pandas.tseries.offsets import MonthBegin

def main(POSTGRES):
    # Initialize or check the connection to PostgreSQL
    #conn = helper.init_connection()
    # Setup and get user inputs
    current_month = display.page_setup()
    unit_scenarios, cost_data, component_counter_data, master_counter_data, fleet_input, eol_date, unit_numbers, uploaded_linked_strategy, rep_cost = display.get_user_input(POSTGRES)

    # Display units and scenarios
    for unit, scenarios in unit_scenarios.items():
        st.write(f"Unit {unit}: {scenarios}")

    # Load and display data if 'Show' button is clicked
    if st.button("Show Data"):
        df_cost, df_component_counter, df_master_counter, pivot_smu = display.load_and_display_data(fleet_input, unit_scenarios, cost_data, component_counter_data, master_counter_data, uploaded_linked_strategy)
        # # Collect inputs and confirm to process data
        df_complete = st.session_state["df_complete"]
        display.process_data(df_complete, current_month, eol_date, df_cost, df_master_counter, unit_numbers, unit_scenarios, pivot_smu, rep_cost)

if __name__ == "__main__":
    POSTGRES = True if helper.init_connection() is not None else False
    #st.write(POSTGRES)
    main(POSTGRES)








# residual code- inputs in streamlit don't work so well as the page resets when the widgets are interacted with
# a different way to input missing costs and overdue replacement dates will be needed
def collect_inputs():
    #st.header("Edit Missing Costs and Overdue Dates")
    df_complete = st.session_state['df_complete']
    df_complete.to_csv('complete_data_editable.csv', index=False)
    if st.button("Reupload filled in data"):
        uploaded_filled_data = st.sidebar.file_uploader("Upload Filled in Dataset in sidebar", type=["csv", "xlsx"])
        df_complete = uploaded_filled_data
    if st.button("Confirm complete dataset"):
        st.session_state['df_complete'] = df_complete
        st.header("Final dataset")
        st.write(df_complete)

    # for index, row in complete_df.iterrows():
    #     if pd.isna(row["TotSum (actual)"]) or row["TotSum (actual)"] <= 0:
    #         cost_key = f"cost_{index}"
    #         new_cost = st.number_input(f"Enter cost for item {index}", key=cost_key, value=row.get("TotSum (actual)", 0))
    #         complete_df.at[index, "TotSum (actual)"] = new_cost
        
    #     if row["MaintItemInterval"] < row["Counter reading"]:
    #         date_key = f"date_{index}"
    #         new_date = st.date_input(f"Enter replacement date for item {index}", key=date_key, value=pd.to_datetime(row.get("First Replacement Month", 'today')))
    #         complete_df.at[index, "First Replacement Month"] = new_date
