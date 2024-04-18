## LEGEND ##
# IW38 = cost
# IP24/18/19 = strategy
# IK17 = counter

import calc
import helper
import display

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

##Setting up
#function for getting unit numbers for the fleet
def get_unit_numbers_by_fleet(fleet_name, POSTGRES, uploaded_fleet_list):
    
    query = f"SELECT unit FROM fleet_list WHERE fleet = '{fleet_name}';"
    query2 = f"SELECT approx_rep_cost FROM fleet_list WHERE fleet = '{fleet_name}';"
    result_final = []
    if POSTGRES == False:
        if uploaded_fleet_list is not None:
            #Read the uploaded files
            fleet_list = uploaded_fleet_list
            df_result = fleet_list[fleet_list['Fleet'] == fleet_name]['Unit']
            result_final = df_result.tolist()
            #st.write(fleet_list)
            rep_cost = fleet_list[fleet_list['Fleet'] == fleet_name]['Approx. Replacement Cost']
            rep_cost = pd.DataFrame({"Unit" : result_final, "Approx. Replacement Cost" : rep_cost})
    else:
        uploaded_linked_strategy = None
        result = helper.run_query(query)
        result_final = [row[0] for row in result]
        rep_cost = helper.run_query(query2)
        rep_cost = [row[0] for row in rep_cost]
        rep_cost = pd.DataFrame({"Unit" : result_final, "Approx. Replacement Cost" : rep_cost})
    return result_final, rep_cost



#extract required data for the chosen fleet from master ip24 database

def get_master_strategy_data(selected_units, uploaded_linked_strategy):
    if uploaded_linked_strategy is not None:
        linked_strategy = helper.read_file(uploaded_linked_strategy)
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
        result = helper.run_query(query)
        df_result = pd.DataFrame(result, columns=["Unit", "FunctionalLoc", "MaintenancePlan", "MaintItem", "MaintItemText", "Description","MaintItemDesc","MaintItemInterval","ik17component"])
    return df_result

#combine required data from iw38 and master ip24

def filter_cost_data_pm02(df_cost_data, selected_fleet, uploaded_linked_strategy):
    df_cost_filtered_PMO2 = calc.avg_cost_data_pm02(df_cost_data)
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
    df_filtered_warranty["FinMonth"] = df_filtered_warranty.apply(lambda x: helper.to_financial_month(x["Actual Finish"]) if pd.notnull(x["Actual Finish"]) else helper.to_financial_month(x["Basic fin. date"]), axis=1)
    df_filtered_warranty = df_filtered_warranty.rename(columns={'Total act.costs': 'Total act.costs_%s' %data_type})
    #create pivot table (needs to use maximum counter reading per month per unit)
    pivot_cost = df_filtered_warranty.pivot_table(values="Total act.costs_%s" %data_type, index="FinMonth", columns = "Sort field", aggfunc = "sum")
    return pivot_cost

def pivot_master_counter_data(df_master_counter):
    #ensure date is datetime type
    df_master_counter["Date"] = pd.to_datetime(df_master_counter["Date"])
    #convert to fiscal month
    df_master_counter["FinMonth"] = df_master_counter["Date"].apply(helper.to_financial_month)
    #create pivot table (needs to use maximum counter reading per month per unit)
    pivot_smu = df_master_counter.pivot_table(values="Counter reading", index="FinMonth", columns = "MeasPosition", aggfunc = "max")
    return pivot_smu


#merge ip24 fleet data with ik17 component data
def merge_counter_strategy_data(fleet_strategy_data, df_component_counter, summary_data):
    df_component_counter.rename(columns={"Measuring point": "ik17component"}, inplace=True)
    merged_data = pd.merge(fleet_strategy_data, df_component_counter, on="ik17component", how="left")
    df_component_counter.rename(columns={"Counter reading": "Counter reading (IK17)"}, inplace=True)
    merged_data = pd.merge(merged_data, summary_data[["MeasPosition", "Average_Hours_Per_Day"]], left_on="Unit", right_on="MeasPosition", how="left")
    merged_data["MaintItemInterval"] = merged_data.apply(lambda row: helper.convert_interval_to_hours(row["MaintItemInterval"], row["Average_Hours_Per_Day"]), axis=1)
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
        (pivot_cost_PM01, 'PM01'),
        (pivot_cost_PM03, 'PM03')
    ]
    # Loop over the DataFrames to reset the index and melt them
    for df_pivot, key in df_pivots:
        # Reset the index
        pivot_reset = df_pivot.reset_index()
        # Melt the DataFrame
        df_melted = pivot_reset.melt(id_vars=["FinMonth"], var_name="Sort field", value_name=f"Total act.costs_%s" %key)
        # Store the melted DataFrame in the dictionary
        melted_dataframes[key] = df_melted

    merged_pivots1 = pd.merge(smu_melted, melted_dataframes['PM01'], on=["FinMonth", "Sort field"], how = "outer")
    merged_pivots = pd.merge(merged_pivots1, melted_dataframes['PM03'], on=["FinMonth", "Sort field"], how = "outer")
    merged_pivots["Round SMU"] = (merged_pivots["Counter reading"] / 1000).round() * 1000
    
    cumulative = calc.cumulative_costs(merged_pivots)
    return cumulative



#displaying and merging data outputs
def merge_all_data(fleet_input, unit_scenarios, df_cost, df_component_counter, df_master_counter, uploaded_linked_strategy):
    selected_fleet = unit_scenarios.keys()
    df_cost_filtered_PM02, fleet_strategy_data = filter_cost_data_pm02(df_cost, selected_fleet, uploaded_linked_strategy)
    if df_cost_filtered_PM02 is None:
        return
    summary_data = calc.process_master_counter_data(df_master_counter) if df_master_counter is not None else None
    merged_data = merge_counter_strategy_data(fleet_strategy_data, df_component_counter, summary_data) if df_component_counter is not None else None
    display.display_data(fleet_input, df_cost_filtered_PM02, fleet_strategy_data, summary_data, merged_data)
    return merged_data


