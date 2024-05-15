# TUNDRA ESO Application code v1
# Written by Jamieson Mulready and Samantha McMaster

# import necessary libraries
import calc
import helper

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# extracting unit numbers for the fleet as well as unit replacement costs
# used in display.get_user_input
def get_unit_numbers_and_repl_costs(fleet_name, uploaded_fleet_list):
    query_units = f"SELECT unit FROM fleet_list WHERE fleet = '{fleet_name}';"
    query_repl_costs = f"SELECT approx_rep_cost FROM fleet_list WHERE fleet = '{fleet_name}';"
    result_final = []

    # manual file upload
    if uploaded_fleet_list is not None:
            #Read the uploaded files
            df_result = uploaded_fleet_list[uploaded_fleet_list['Fleet'] == fleet_name]['Unit']
            result_final = df_result.tolist() 
            repl_cost = uploaded_fleet_list[uploaded_fleet_list['Fleet'] == fleet_name]['Approx. Replacement Cost']
            repl_cost = pd.DataFrame({"Unit" : result_final, "Approx. Replacement Cost" : repl_cost}) # unit numbers and replacement costs
    
    # connected to postgres database
    else:
        result = helper.run_query(query_units)
        result_final = [row[0] for row in result]
        repl_cost = helper.run_query(query_repl_costs)
        repl_cost = [row[0] for row in repl_cost]
        repl_cost = pd.DataFrame({"Unit" : result_final, "Approx. Replacement Cost" : repl_cost}) # unit numbers and replacement costs

    return result_final, repl_cost

# combine required cost and strategy data
def filter_cost_data_pm02(df_cost_data, selected_units, uploaded_linked_strategy):
    df_cost_filtered_PMO2 = calc.avg_cost_data_pm02(df_cost_data)
    fleet_strategy_data = get_master_strategy_data(selected_units, uploaded_linked_strategy)

    if fleet_strategy_data is not None:
        fleet_strategy_data = pd.merge(fleet_strategy_data, df_cost_filtered_PMO2, on="MaintItem", how="left")

    return df_cost_filtered_PMO2, fleet_strategy_data

# extract required data for the chosen fleet from linked strategy
# used in filter_cost_data_pm02
def get_master_strategy_data(selected_units, uploaded_linked_strategy):
    expected_columns = ["Unit", "FunctionalLoc", "MaintenancePlan", "MaintItem", "MaintItemText", "Description", "MaintItemDesc", "MaintItemInterval", "ik17component"]

    # manual upload
    if uploaded_linked_strategy is not None:
        for column in expected_columns:
            if column not in uploaded_linked_strategy.columns:
                uploaded_linked_strategy[column] = pd.NA
        # Reorder DataFrame to match the expected format
        df_linked_strategy = uploaded_linked_strategy[expected_columns]
        df_result = df_linked_strategy[df_linked_strategy['Unit'].isin(selected_units)]

    # using postgres database
    else:
        unit_numbers_str = ", ".join(f"'{unit}'" for unit in selected_units)
        query = f"SELECT * FROM linkedip24 WHERE unit IN ({unit_numbers_str});"
        result = helper.run_query(query)
        df_result = pd.DataFrame(result, columns=expected_columns)

    return df_result

# merge strategy data with counter data
def merge_counter_strategy_data(fleet_strategy_data, df_component_counter, summary_data):
    df_component_counter.rename(columns={"Measuring point": "ik17component"}, inplace=True)
    merged_data = pd.merge(fleet_strategy_data, df_component_counter, on="ik17component", how="left")

    df_component_counter.rename(columns={"Counter reading": "Counter reading (IK17)"}, inplace=True)
    merged_data = pd.merge(merged_data, summary_data[["MeasPosition", "Average_Hours_Per_Day"]], left_on="Unit", right_on="MeasPosition", how="left")
    merged_data["MaintItemInterval"] = merged_data.apply(lambda row: helper.convert_interval_to_hours(row["MaintItemInterval"], row["Average_Hours_Per_Day"]), axis=1)

    return merged_data

# pivot the unit SMU each month
def pivot_master_counter_data(df_master_counter):
    #ensure date is datetime type
    df_master_counter["Date"] = pd.to_datetime(df_master_counter["Date"])

    #convert to fiscal month
    df_master_counter["FinMonth"] = df_master_counter["Date"].apply(helper.to_financial_month)

    #create pivot table (needs to use maximum counter reading per month per unit)
    pivot_smu = df_master_counter.pivot_table(values="Counter reading", index="FinMonth", columns = "MeasPosition", aggfunc = "max")
    return pivot_smu

# use estimated cost in place of TotSum actual if that is missing
def use_estimated_cost(df_cost):
    df_cost.loc[(df_cost['TotSum (actual)'] <= 0) & (df_cost['Estimated costs'] > 0), 'TotSum (actual)'] = df_cost['Estimated costs']
    return df_cost

#filter cost data for only relevant data type (PM01 or PM03)
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

# merging smu and cost pivot
def combine_cost_smu_pivots(pivot_smu, pivot_cost_PM01, pivot_cost_PM03, summary_data, unit_numbers):
    pivot_smu_reset = pivot_smu.reset_index()
    pivot_smu_reset.rename(columns={"MeasPosition": "Sort field"}, inplace=True)
    smu_melted = pivot_smu_reset.melt(id_vars=["FinMonth"], var_name="Sort field", value_name="Counter reading")

    # Dictionary to hold the melted DataFrames
    melted_dataframes = {}
    # List of tuples containing the DataFrame and its associated key
    df_pivots = [(pivot_cost_PM01, 'PM01'), (pivot_cost_PM03, 'PM03')]

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
    # calculate comulative costs to add to the table
    cumulative = calc.cumulative_costs(merged_pivots)
    
    return cumulative

def main(df_cost, df_master_counter, unit_numbers, uploaded_linked_strategy, df_component_counter):
    df_cost_supplemented = use_estimated_cost(df_cost)
    
    df_cost_filtered_PM02, fleet_strategy_data = filter_cost_data_pm02(df_cost_supplemented, unit_numbers, uploaded_linked_strategy)
    summary_data = calc.process_master_counter_data(df_master_counter)
    merged_data = merge_counter_strategy_data(fleet_strategy_data, df_component_counter, summary_data) if df_component_counter is not None else None
    pivot_smu = pivot_master_counter_data(df_master_counter)

    pmo1_cost_pivot = pivot_cost_data(df_cost_supplemented, "PM01")
    pmo3_cost_pivot = pivot_cost_data(df_cost_supplemented, "PM03")
    merged_pivots = combine_cost_smu_pivots(pivot_smu, pmo1_cost_pivot, pmo3_cost_pivot, summary_data, unit_numbers)

    merged_pivots.to_csv('merged_pivots_test.csv', index=False)

    return summary_data, merged_pivots, df_cost_filtered_PM02, merged_data, fleet_strategy_data