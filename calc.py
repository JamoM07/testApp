## LEGEND ##
# IW38 = cost
# IP24/18/19 = strategy
# IK17 = counter

import ETL
import helper
import display

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import MonthBegin

#currently unused *******************************
def calculate_costs(smu, coefficients):
    # takes in coefficients to find cost for given hours
    a, b = coefficients
    return a * smu**2 + b * smu

#filter IW38 for only PMO2 and calculate average cost for each maint item

def avg_cost_data_pm02(df_cost):
    df_filtered = df_cost[df_cost.iloc[:, 0] == "PM02"]
    #st.write(df_filtered)
    df_avg_cost = df_filtered[df_filtered["TotSum (actual)"] > 10].groupby("MaintItem")["TotSum (actual)"].mean()
    return df_avg_cost

#extract counter reading data from ik17 master

def process_master_counter_data(df_master_counter):
    relevant_columns = ["MeasPosition", "Counter reading", "Date"]
    df_master_counter_short = df_master_counter[relevant_columns].copy()
    df_master_counter_short["Date"] = pd.to_datetime(df_master_counter_short["Date"])
    summary_data = df_master_counter_short.groupby("MeasPosition").agg(
        Max_Hour_Reading=("Counter reading", "max"),
        Min_Hour_Reading=("Counter reading", "min"),
        Min_Date=("Date", "min"),
        Max_Date=("Date", "max")
    )
    summary_data["Hours_Used"] = summary_data["Max_Hour_Reading"] - summary_data["Min_Hour_Reading"]
    summary_data["Total_Days"] = (summary_data["Max_Date"] - summary_data["Min_Date"]).dt.days + 1
    summary_data["Average_Hours_Per_Day"] = summary_data["Hours_Used"] / summary_data["Total_Days"]
    summary_data.reset_index(inplace=True)
    return summary_data

# def create_PMO2_replacement_scheduleV2(df_complete, current_month, eol_date):
#     all_months = pd.date_range(current_month, eol_date, freq='M')
#     # Include all necessary columns in the DataFrame
#     replacement_schedule = pd.DataFrame(columns=["Interval", "Usual Days Until Replacement", "Unit", "Overdue"] + [month.strftime('%b-%y') for month in all_months])

#     for index, row in df_complete.iterrows():
#         # Check if "TotSum (actual)" is missing and replace it if necessary
#         if pd.isna(row['TotSum (actual)']) and 'Cost Missing' in row:
#             cost = row['Cost Missing']
#         else:
#             cost = row['TotSum (actual)']

#         # Determine the first replacement month
#         if row['Overdue'] and 'First Replacement Month' in row and pd.notna(row['First Replacement Month']):
#             first_replacement_month = pd.Timestamp(row['First Replacement Month'])
#         else:
#             first_replacement_month = current_month + pd.DateOffset(days=(row["MaintItemInterval"] - row["Counter reading"]) / row["Average_Hours_Per_Day"])
        
#         replacement_schedule.loc[index, "MaintItem"] = row["MaintItem"]
#         replacement_schedule.loc[index, "Interval"] = row["MaintItemInterval"]
#         replacement_schedule.loc[index, "Unit"] = row["Unit"]
#         interval = row["MaintItemInterval"]
#         usage = row["Average_Hours_Per_Day"]

#         # Calculate usual days until replacement
#         usual = interval / usage if usage else float('inf')  # Avoid division by zero
#         replacement_schedule.loc[index, "Usual Days Until Replacement"] = usual
#         replacement_schedule.loc[index, "Overdue"] = row['Overdue']

#         # Fill the schedule
#         for month in all_months:
#             month_str = month.strftime('%b-%y')
#             if first_replacement_month > eol_date:
#                 break
#             if month >= first_replacement_month:
#                 # Calculate replacements this month
#                 days_difference = (month - first_replacement_month).days
#                 replacements_this_month = int(days_difference / usual) + 1 if usual != float('inf') else 0
#                 replacement_schedule.loc[index, month_str] = cost * replacements_this_month

#                 # Update first replacement month for repeated replacements
#                 for _ in range(replacements_this_month):
#                     first_replacement_month += pd.DateOffset(days=int(interval / usage))

#     return replacement_schedule

#outline replacement schedule for each unit
def create_PMO2_replacement_schedule(df_complete, current_month, eol_date):
    all_months = pd.date_range(current_month, eol_date, freq='M')
    replacement_schedule = pd.DataFrame(columns=["Interval", "Usual Days Until Replacement", "Unit", "Overdue", "Cost Missing"] + [month.strftime('%b-%y') for month in all_months])
    last_replacement_dates = {}

    for index, row in df_complete.iterrows():
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



#one function ##########################
def smu_cost_fit(merged_pivots, X):
    pivot_table_max = merged_pivots.pivot_table(values='CumulativeCost%s' %X, 
                                                index='Round SMU', 
                                                columns='Sort field', 
                                                aggfunc='max')
    pivot_table_max['AverageCumulativeCost%s' %X] = pivot_table_max.mean(axis=1)
    # Assuming pivot_table_max is the pivot table created earlier
    cleaned_data = pivot_table_max[['AverageCumulativeCost%s' %X]].dropna()
    x = cleaned_data.index.values  # SMU values
    y = cleaned_data['AverageCumulativeCost%s' %X].values  # Average cumulative cost values
    # Construct the design matrix for quadratic fit (x^2 and x terms)
    X = np.vstack([x**2, x]).T  # .T to transpose, making it two columns
    # Solve the least squares problem
    coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # Rsquared:
    y_pred = coefficients[0] * x**2 + coefficients[1] * x  # Predicted y values based on the fit
    SS_res = np.sum((y - y_pred)**2)  # Sum of squares of residuals
    SS_tot = np.sum((y - np.mean(y))**2)  # Total sum of squares
    R_squared = 1 - (SS_res / SS_tot)
    return coefficients, R_squared

def cumulative_costs(merged_pivots): 
    # Replace NaN in cost columns with 0
    merged_pivots['Total act.costs_PM01'] = merged_pivots['Total act.costs_PM01'].fillna(0)
    merged_pivots['Total act.costs_PM03'] = merged_pivots['Total act.costs_PM03'].fillna(0)
    #First, ensure your DataFrame is sorted by "sort field" (machine) and then by "month/year" to correctly calculate cumulative costs.
    merged_pivots.sort_values(by=["Sort field", "FinMonth"], inplace=True)
    # Group by 'sort field' and then apply forward fill within each group
    merged_pivots['Round SMU'] = merged_pivots.groupby('Sort field')['Round SMU'].fillna(method='ffill')
    # Group by 'sort field' and then apply backfill within each group
    merged_pivots['Round SMU'] = merged_pivots.groupby('Sort field')['Round SMU'].fillna(method='bfill')
    #Calculate the total sum of costs, and min and max SMU (hours) for each machine.
    machine_stats = merged_pivots.groupby('Sort field').agg(
        TotalCostsPM01=("Total act.costs_PM01", 'sum'),
        TotalCostsPM03=("Total act.costs_PM03", 'sum'),
        MinSMU=("Counter reading", 'min'),
        MaxSMU=("Counter reading", 'max'),
        RoundMinSMU=("Round SMU", 'min')
    ).reset_index()
    #Now, calculate the initial cumulative cost factor based on your formula.
    machine_stats['InitialCumulativeFactorPM01'] = (machine_stats['TotalCostsPM01'] / (machine_stats['MaxSMU'] - machine_stats['MinSMU'])) * 0.5 * machine_stats['RoundMinSMU']
    machine_stats['InitialCumulativeFactorPM03'] = (machine_stats['TotalCostsPM03'] / (machine_stats['MaxSMU'] - machine_stats['MinSMU'])) * 0.5 * machine_stats['RoundMinSMU']
    #Merge this factor back into the original DataFrame on "sort field" to have the initial cumulative cost factor available for each row.
    merged_pivots = merged_pivots.merge(machine_stats[['Sort field', 'InitialCumulativeFactorPM01']], on='Sort field')
    merged_pivots = merged_pivots.merge(machine_stats[['Sort field', 'InitialCumulativeFactorPM03']], on='Sort field')
    #For each machine, calculate the cumulative costs, starting with the initial value for the earliest month, then cumulatively summing up total act.costs_x and total act.costs_y respectively.
    #Initialize columns to store the cumulative costs
    merged_pivots['CumulativeCostPM01'] = 0
    merged_pivots['CumulativeCostPM03'] = 0

    # Iterate over each machine to calculate cumulative costs#
    for machine in merged_pivots['Sort field'].unique():
        machine_mask = merged_pivots['Sort field'] == machine
        first_index = merged_pivots.loc[machine_mask].index[0]
        merged_pivots.loc[machine_mask, 'CumulativeCostPM01'] = merged_pivots.loc[machine_mask, 'Total act.costs_PM01'].cumsum() + merged_pivots.loc[first_index, 'InitialCumulativeFactorPM01'] - merged_pivots.loc[first_index, 'Total act.costs_PM01']
        merged_pivots.loc[machine_mask, 'CumulativeCostPM03'] = merged_pivots.loc[machine_mask, 'Total act.costs_PM03'].cumsum() + merged_pivots.loc[first_index, 'InitialCumulativeFactorPM03'] - merged_pivots.loc[first_index, 'Total act.costs_PM03']

    return merged_pivots


def forecast_unit_costs(start_month, end_of_life, current_hours, avg_hours_per_day, replacement_hours, coefficients_PM01, coefficients_PM03):
    months = pd.date_range(start=start_month, end=end_of_life, freq='MS')
    operating_hours = 0
    cumulative_hours = current_hours
    cumulative_costPM01 = cumulative_costPM03 = 0
    monthly_costPM01 = monthly_costPM03 = 0
    forecast_data = []

    # use first full month
    if start_month.day > 1:
        adjusted_start_month = start_month + MonthBegin(1)
    else:
        adjusted_start_month = start_month

    for month in months:
        # Get the number of days in the month
        days_in_month = pd.Period(month, freq='M').days_in_month

        if cumulative_hours >= replacement_hours:
            # If the unit needs replacement, record the current month and break
            replacement_start_month = month
            break

        operating_hours = avg_hours_per_day * days_in_month
        cumulative_hours += operating_hours

        new_cumulative_costPM01 = coefficients_PM01[0] * cumulative_hours ** 2 + coefficients_PM01[1] * cumulative_hours
        new_cumulative_costPM03 = coefficients_PM03[0] * cumulative_hours ** 2 + coefficients_PM03[1] * cumulative_hours

        if  month == adjusted_start_month:
            monthly_costPM01 = 0
            monthly_costPM03 = 0
        else:
            monthly_costPM01 = new_cumulative_costPM01 - cumulative_costPM01
            monthly_costPM03 = new_cumulative_costPM03 - cumulative_costPM03

        cumulative_costPM01 = new_cumulative_costPM01
        cumulative_costPM03 = new_cumulative_costPM03

        forecast_data.append({
            'Month': month.strftime('%Y-%m'),
            'Operating Hours': operating_hours,
            'Cumulative Hours': cumulative_hours,
            'Cumulative Cost PM01': cumulative_costPM01,
            'Cumulative Cost PM03': cumulative_costPM03,
            'Monthly Cost PM01': monthly_costPM01,
            'Monthly Cost PM03': monthly_costPM03
        })
    else:
        # If loop did not break, all months are forecasted, and no replacement is needed
        replacement_start_month = None

    return forecast_data, replacement_start_month

def calculate_average_daily_hours(units_data):
    total_hours = sum(data['avg_daily_hours'] for data in units_data.values())
    return total_hours / len(units_data)

def forecast_for_scenario(unit, units_data, start_month, scenario_replacement_hours, coefficients_costPM01, coefficients_costPM03, end_of_life):
    current_hours = units_data[unit]['current_hours']
    avg_hours_per_day = units_data[unit]['avg_daily_hours']
    
    forecast, replacement_start_month = forecast_unit_costs(
        start_month, end_of_life, current_hours, avg_hours_per_day,
        scenario_replacement_hours, coefficients_costPM01, coefficients_costPM03
    )
    
    return forecast, replacement_start_month

def forecast_all_units_scenarios(start_month, unit_numbers, units_data, scenarios, coefficients_costPM01, coefficients_costPM03, end_of_life):
    all_scenarios_forecasts = {}
    average_daily_hours = calculate_average_daily_hours(units_data)
    #start_month = pd.Timestamp('2024-04-01')  # Starting month (adjust if needed)

    for scenario_name, replacement_hours in scenarios.items():
        scenario_forecasts = {}
        
        for unit in unit_numbers:
            forecast, replacement_start_month = forecast_for_scenario(
                unit, units_data, start_month, replacement_hours,
                coefficients_costPM01, coefficients_costPM03, end_of_life
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
                    replacement_hours, coefficients_costPM01, coefficients_costPM03
                )
        
        all_scenarios_forecasts[scenario_name] = scenario_forecasts

    return all_scenarios_forecasts

def forecast_all_units_scenarios_to_csv(start_month, unit_numbers, units_data, scenarios, coefficients_costPM01, coefficients_costPM03, end_of_life, output_csv):
    all_scenarios_forecasts = forecast_all_units_scenarios(start_month, unit_numbers, units_data, scenarios, coefficients_costPM01, coefficients_costPM03, end_of_life)
    
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
    #st.write(df_long_format)

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
    return df, df_long_format



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


#####################placeholder for npv
def calculate_npv(scenario_hours):
    return 123456.78








# still work in progress
def fy_summary_PM01_3(df, rep_cost):
    # Assuming df is your DataFrame
    df['Month'] = pd.to_datetime(df['Month'])  # Converting month to datetime
    df['Year'] = df['Month'].dt.year  # Extracting the year from month for grouping
    # Group by necessary categories and calculate annual sum and last value of the year
    annual_summary = df.groupby(['Scenario', 'Unit', 'Data Type', 'Year']).agg(
        annual_sum=('Value', 'sum'),
        year_end_smu=('Value', 'last')  # Assumes the data for each month is sorted chronologically
    ).reset_index()

    # Merge fleet list to get the replacement costs for each unit
    df = df.merge(rep_cost, on='Unit', how='left')

    # Assuming replacements happen when 'Monthly Hours' go to None (or any logic you define)
    df['CAPEX'] = df.apply(lambda x: x['Approx. Replacement Cost'] if (x['Data Type'] == 'Monthly Hours' and pd.isna(x['Value'])) else 0, axis=1)

    # Aggregate CAPEX by year
    annual_capex = df.groupby(['Scenario', 'Unit', 'Year'])['CAPEX'].max().reset_index()  # Take max to avoid duplicating costs if more than one month is NaN
    final_df = annual_summary.merge(annual_capex, on=['Scenario', 'Unit', 'Year'], how='left')

    # Compute total annual SMU for each scenario and year
    total_annual_smu = df[df['Data Type'] == 'Monthly Hours'].groupby(['Scenario', 'Year'])['Value'].sum().reset_index()
    total_annual_smu.rename(columns={'Value': 'Total Annual SMU'}, inplace=True)

    # Count CAPEX investments
    capex_counts = annual_capex.groupby(['Scenario', 'Year'])['CAPEX'].apply(lambda x: (x > 0).sum()).reset_index()
    capex_counts.rename(columns={'CAPEX': 'CAPEX Investments'}, inplace=True)

    # Merge to the final DataFrame
    final_df = final_df.merge(total_annual_smu, on=['Scenario', 'Year'], how='left')
    final_df = final_df.merge(capex_counts, on=['Scenario', 'Year'], how='left')

    return final_df





