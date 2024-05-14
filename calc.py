# TUNDRA ESO Application code v1
# Written by Jamieson Mulready and Samantha McMaster

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import MonthBegin

# filter cost data for only PM02 and calculate average cost for each maint item
## used in ETL
def avg_cost_data_pm02(df_cost):
    df_filtered = df_cost[df_cost.iloc[:, 0] == "PM02"]
    df_avg_cost = df_filtered[df_filtered["TotSum (actual)"] > 10].groupby("MaintItem")["TotSum (actual)"].mean()
    return df_avg_cost

# main function to manage cumulative costs calculation
def cumulative_costs(merged_pivots):
    merged_pivots = prepare_and_sort_data(merged_pivots)
    machine_stats = compute_machine_stats(merged_pivots)
    merged_pivots = merge_and_calculate_cumulative_costs(merged_pivots, machine_stats)
    return merged_pivots

# handle NaNs and sort the DataFrame
def prepare_and_sort_data(merged_pivots):
    merged_pivots['Total act.costs_PM01'] = merged_pivots['Total act.costs_PM01'].fillna(0)
    merged_pivots['Total act.costs_PM03'] = merged_pivots['Total act.costs_PM03'].fillna(0)
    merged_pivots.sort_values(by=["Sort field", "FinMonth"], inplace=True)
    return merged_pivots

# compute machine stats and initial cumulative cost factors
def compute_machine_stats(merged_pivots):
    merged_pivots['Round SMU'] = merged_pivots.groupby('Sort field')['Round SMU'].fillna(method='ffill').fillna(method='bfill')
    machine_stats = merged_pivots.groupby('Sort field').agg(TotalCostsPM01=("Total act.costs_PM01", 'sum'), TotalCostsPM03=("Total act.costs_PM03", 'sum'), MinSMU=("Counter reading", 'min'), MaxSMU=("Counter reading", 'max'), RoundMinSMU=("Round SMU", 'min')).reset_index()
    machine_stats['InitialCumulativeFactorPM01'] = (machine_stats['TotalCostsPM01'] / (machine_stats['MaxSMU'] - machine_stats['MinSMU'])) * 0.5 * machine_stats['RoundMinSMU']
    machine_stats['InitialCumulativeFactorPM03'] = (machine_stats['TotalCostsPM03'] / (machine_stats['MaxSMU'] - machine_stats['MinSMU'])) * 0.5 * machine_stats['RoundMinSMU']
    return machine_stats

# merge machine stats and calculate cumulative costs
def merge_and_calculate_cumulative_costs(merged_pivots, machine_stats):
    merged_pivots = merged_pivots.merge(machine_stats[['Sort field', 'InitialCumulativeFactorPM01']], on='Sort field')
    merged_pivots = merged_pivots.merge(machine_stats[['Sort field', 'InitialCumulativeFactorPM03']], on='Sort field')
    for machine in merged_pivots['Sort field'].unique():
        machine_mask = merged_pivots['Sort field'] == machine
        first_index = merged_pivots.loc[machine_mask].index[0]
        merged_pivots.loc[machine_mask, 'CumulativeCostPM01'] = merged_pivots.loc[machine_mask, 'Total act.costs_PM01'].cumsum() + merged_pivots.loc[first_index, 'InitialCumulativeFactorPM01'] - merged_pivots.loc[first_index, 'Total act.costs_PM01']
        merged_pivots.loc[machine_mask, 'CumulativeCostPM03'] = merged_pivots.loc[machine_mask, 'Total act.costs_PM03'].cumsum() + merged_pivots.loc[first_index, 'InitialCumulativeFactorPM03'] - merged_pivots.loc[first_index, 'Total act.costs_PM03']
    return merged_pivots
##

# find quadratic fit between smu and cost
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

# extract master counter reading data
## calls in ETL too
def process_master_counter_data(df_master_counter):
    relevant_columns = ["MeasPosition", "Counter reading", "Date"]
    df_master_counter_short = df_master_counter[relevant_columns].copy()
    df_master_counter_short["Date"] = pd.to_datetime(df_master_counter_short["Date"])
    summary_data = df_master_counter_short.groupby("MeasPosition").agg(Max_Hour_Reading=("Counter reading", "max"), Min_Hour_Reading=("Counter reading", "min"), Min_Date=("Date", "min"), Max_Date=("Date", "max"))
    summary_data["Hours_Used"] = summary_data["Max_Hour_Reading"] - summary_data["Min_Hour_Reading"]
    summary_data["Total_Days"] = (summary_data["Max_Date"] - summary_data["Min_Date"]).dt.days + 1
    summary_data["Average_Hours_Per_Day"] = summary_data["Hours_Used"] / summary_data["Total_Days"]
    summary_data.reset_index(inplace=True)
    return summary_data

# overall PM01 and PM03 forecasting for all scenarios and units
def forecast_all_units_scenarios(start_month, unit_numbers, units_data, scenarios, coefficients_costPM01, coefficients_costPM03, end_of_life, merged_data):
    all_scenarios_forecasts = {}
    PM02_replacement_schedules = {}
    
    average_daily_hours = calculate_average_daily_hours(units_data)

    for scenario_name, replacement_hours in scenarios.items():
        scenario_forecasts = {}
        scenario_replacement_dates = {unit: [] for unit in unit_numbers}  # Initialize once per scenario for all units

        for unit in unit_numbers:
            current_hours = units_data[unit]['current_hours']
            avg_hours_per_day = units_data[unit]['avg_daily_hours']
            forecast, replacement_start_month = forecast_unit_costs(
                start_month, end_of_life, current_hours, avg_hours_per_day,
                replacement_hours, coefficients_costPM01, coefficients_costPM03
            )
            scenario_forecasts[unit] = forecast

            # Continuously check for further replacements
            while replacement_start_month:
                scenario_replacement_dates[unit].append(replacement_start_month)  # Track dates under the original unit
                replacement_unit = f"{unit}_replacement_{replacement_start_month.strftime('%Y%m%d')}_{scenario_name}"
                if replacement_unit not in units_data:
                    units_data[replacement_unit] = {
                        'current_hours': 0,
                        'avg_daily_hours': average_daily_hours
                    }
                scenario_forecasts[replacement_unit], replacement_start_month = forecast_unit_costs(
                    replacement_start_month, end_of_life, 0, average_daily_hours,
                    replacement_hours, coefficients_costPM01, coefficients_costPM03
                )

        all_scenarios_forecasts[scenario_name] = scenario_forecasts
        PM02_replacement_schedules[scenario_name] = create_PM02_replacement_schedule(merged_data, start_month, end_of_life, scenario_replacement_dates)
        
    return all_scenarios_forecasts, PM02_replacement_schedules

# find average daily operating hours
def calculate_average_daily_hours(units_data):
    total_hours = sum(data['avg_daily_hours'] for data in units_data.values())
    return total_hours / len(units_data)

# forecasting for PM01 and PM03 costs for a single unit
def forecast_unit_costs(start_month, end_of_life, current_hours, avg_hours_per_day, replacement_hours, coefficients_PM01, coefficients_PM03):

    months = pd.date_range(start=start_month, end=end_of_life, freq='MS')
    operating_hours = 0
    cumulative_hours = current_hours
    cumulative_costPM01 = cumulative_costPM03 = 0
    monthly_costPM01 = monthly_costPM03 = 0
    forecast_data = []
    adjusted_start_month = start_month + MonthBegin(1) if start_month.day > 1 else start_month

    for month in months:
        if month < adjusted_start_month:
            continue  # Skip partial month at the start, if policy allows

        days_in_month = pd.Period(month, freq='M').days_in_month
        if cumulative_hours >= replacement_hours:
            replacement_start_month = month  # Use the current month, not the next
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
        forecast_data.append({'Month': month.strftime('%Y-%m'), 'Operating Hours': operating_hours, 'Cumulative Hours': cumulative_hours, 'Cumulative Cost PM01': cumulative_costPM01, 'Cumulative Cost PM03': cumulative_costPM03, 'Monthly Cost PM01': monthly_costPM01, 'Monthly Cost PM03': monthly_costPM03})
    
    else:
        # If loop did not break, all months are forecasted, and no replacement is needed
        replacement_start_month = None

    return forecast_data, replacement_start_month

# outline PM02 replacement schedule for each unit
def create_PM02_replacement_schedule(df_complete, current_month, eol_date, replacement_dates):
    all_months = pd.date_range(current_month, eol_date, freq='M')
    replacement_schedule = pd.DataFrame(columns=["Interval", "Usual Days Until Replacement", "Unit", "Overdue", "Cost Missing"] + [month.strftime('%b-%y') for month in all_months])
    last_replacement_dates = {}

    for index, row in df_complete.iterrows():
        first_replacement_month = None
        replacement_schedule.loc[index, "MaintItem"] = row["MaintItem"]
        replacement_schedule.loc[index, "Interval"] = row["MaintItemInterval"]
        replacement_schedule.loc[index, "Unit"] = row["Unit"]
        unit = row["Unit"]
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

        unit_replacements = replacement_dates.get(unit, [])

        for month in all_months:
            month_str = month.strftime('%b-%y')
            # Check for replacement dates
            if any(pd.to_datetime(replacement_date).strftime('%Y-%m') == month.strftime('%Y-%m') for replacement_date in unit_replacements):
                
                first_replacement_month = month + pd.DateOffset(days=interval / usage if usage else float('inf'))

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

# overview of financial years for PM02
def PM02_fy_overview(replacement_schedule, start_date, end_date):
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

# correctly format the forecast for reading, keep long format as easier for processing but wide format better for reading
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

# overview of financial years for PM01 and PM03
def fy_overview_PM01_3(full_scenario_outputs, repl_cost, scenario):
    # Filter for the current scenario
    df = full_scenario_outputs[full_scenario_outputs['Scenario'] == scenario]
    # Clean and convert the 'Approx. Replacement Cost' to float after removing $ and commas
    repl_cost['Approx. Replacement Cost'] = repl_cost['Approx. Replacement Cost'].replace('[\$,]', '', regex=True).astype(float)
    repl_cost = repl_cost.rename(columns={'Unit': 'Original_Unit'})

    # Convert 'Month' to datetime and extract the year
    df['Month'] = pd.to_datetime(df['Month'])
    df['Year'] = df['Month'].dt.year
    
    # Prepare the data for aggregation
    pm01 = df[df['Data Type'].str.contains('Monthly Cost PM01')]
    pm03 = df[df['Data Type'].str.contains('Monthly Cost PM03')]
    smu_cumulative = df[df['Data Type'].str.contains('Cumulative Hours')]
    smu_monthly = df[df['Data Type'].str.contains('Operating Hours')]
    
    # Group by Unit and Year to aggregate data
    annual_pm01 = pm01.groupby(['Unit', 'Year'])['Value'].sum().reset_index(name='Annual PM01 Cost')
    annual_pm03 = pm03.groupby(['Unit', 'Year'])['Value'].sum().reset_index(name='Annual PM03 Cost')
    annual_smu = smu_monthly.groupby(['Unit', 'Year'])['Value'].sum().reset_index(name='Annual SMU')
    year_end_smu = smu_cumulative.groupby(['Unit', 'Year'])['Value'].last().reset_index(name='Year-End SMU')
    
    # Merge all the dataframes
    summary = pd.merge(annual_pm01, annual_pm03, on=['Unit', 'Year'], how='outer')
    summary = pd.merge(summary, annual_smu, on=['Unit', 'Year'], how='outer')
    summary = pd.merge(summary, year_end_smu, on=['Unit', 'Year'], how='outer')
    
    scenario_data = df[df['Scenario'] == scenario]
    summary['Original_Unit'] = summary['Unit'].str.extract(r'([^_]+)_replacement')
    
    # Identify first operational year for each replacement unit
    replacement_first_year = scenario_data[scenario_data['Unit'].str.contains('replacement')].groupby('Unit')['Year'].min().reset_index()
    replacement_first_year.rename(columns={'Year': 'First_Year'}, inplace=True)

    # Incorporate replacement costs
    # Assuming repl_cost has columns 'Unit' and 'Approx. Replacement Cost'
    summary = summary.merge(repl_cost, on='Original_Unit', how='left')
    summary = summary.merge(replacement_first_year, on='Unit', how='left')
    
    #summary['CAPEX'] = summary.apply(lambda x: x['Approx. Replacement Cost'] if 'replacement' in x['Unit'] else 0, axis=1)
    summary['CAPEX'] = summary.apply(lambda x: x['Approx. Replacement Cost'] if 'replacement' in x['Unit'] and x['Year'] == x['First_Year'] else 0, axis=1)

    summary['Annual PM01 Cost'] = summary['Annual PM01 Cost'] / 1000
    summary['Annual PM03 Cost'] = summary['Annual PM03 Cost'] / 1000

    summary.rename(columns={'Annual PM01 Cost': 'Annual PM01 Cost (000s)','Annual PM03 Cost': 'Annual PM03 Cost (000s)'}, inplace=True)

    # Reshape for final output format if needed
    # melt the dataframe
    melted = pd.melt(summary, id_vars=['Unit', 'Year'], value_vars=['Annual PM01 Cost (000s)', 'Annual PM03 Cost (000s)', 'Annual SMU', 'Year-End SMU', 'CAPEX'],
                    var_name='Data Type', value_name='Value')

    # pivot the melted dataframe
    final = melted.pivot_table(index=['Unit', 'Data Type'], columns='Year', values='Value', aggfunc='sum').fillna(0)

    # Reset index to make 'Unit' and 'Data Type' as columns
    final.reset_index(inplace=True)

    # define custom sort order
    sort_order = {'Annual PM01 Cost (000s)': 1, 'Annual PM03 Cost (000s)': 2, 'Annual SMU': 3, 'Year-End SMU': 4, 'CAPEX': 5}

    # Create a new column for sorting based on the custom order
    final['sort_order'] = final['Data Type'].map(sort_order)

    # Step 6: Sort the dataframe
    final = final.sort_values(by=['Unit', 'sort_order']).drop(columns='sort_order')

    # Optionally convert to a more conventional dataframe format if needed
    final.columns.name = None  # Removes the name 'Year' from the columns header
    summary = final.rename_axis(None, axis=1)  # Removes the index name

    return summary



# process the data and create replacement schedules and forecasts, calls most of the calc functions
def main(merged_data, current_month, eol_date, unit_numbers, unit_scenarios, repl_cost, merged_pivots, summary_data, df_master_counter):
    pm02_fy_overviews = {}
    pm01_3_fy_overviews = {}

    coeff_PM01, Rsquare_PM01 = smu_cost_fit(merged_pivots, "PM01")
    coeff_PM03, Rsquare_PM03 = smu_cost_fit(merged_pivots, "PM03")

    summary_data = process_master_counter_data(df_master_counter) if df_master_counter is not None else None
    current_hours = summary_data.set_index('MeasPosition')['Max_Hour_Reading'].to_dict()
    average_daily_hours = summary_data.set_index('MeasPosition')['Average_Hours_Per_Day'].to_dict()
    units_data = {unit: {'current_hours': current_hours[unit], 'avg_daily_hours': average_daily_hours[unit]} for unit in unit_numbers}
    
    scenarios = next(iter(unit_scenarios.values()))
    all_scenarios_forecasts, pm02_replacement_schedules = forecast_all_units_scenarios(current_month, unit_numbers, units_data, scenarios, coeff_PM01, coeff_PM03, eol_date, merged_data)
    
    for scenario_name, replacement_schedule in pm02_replacement_schedules.items():
        pm02_fy_overviews[scenario_name] = PM02_fy_overview(replacement_schedule, current_month, eol_date)
    
    formatted_forecasts, formatted_forecasts_long = format_forecast_outputs(all_scenarios_forecasts)
    
    for scenario_name in scenarios:
        pm01_3_fy_overviews[scenario_name] = fy_overview_PM01_3(formatted_forecasts_long, repl_cost, scenario_name)
        
        
    
    return pm02_replacement_schedules, formatted_forecasts, formatted_forecasts_long, pm02_fy_overviews, pm01_3_fy_overviews












##################### unused currently
# placeholder for npv calculations
def calculate_npv(scenario_hours):
    return 123456.78

# output forecast to csv (used for testing)
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