# TUNDRA ESO Application code v1
# Written by Jamieson Mulready and Samantha McMaster

import streamlit as st
import psycopg2
from psycopg2 import OperationalError
import pandas as pd
import numpy as np

#initiate connection to sql database
def init_connection():
    try:
        conn = psycopg2.connect(**st.secrets["postgres"])
        return conn
    except OperationalError:
        # Could not connect to the database
        return None
    
def run_query(query):
    conn = init_connection()
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()
    
#function to read in csv or xlsx files and return an error if incorrect format
def read_file(file_data):
    if file_data is not None:
        file_extension = file_data.name.split(".")[-1].lower()
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
    
#convert to fiscal month
def to_financial_month(date):
    if pd.isnull(date):
        return None
    # Financial year starts in July
    month_offset = 6
    financial_month = (date.month - 1 + month_offset) % 12 + 1
    financial_year = date.year if date.month < 7 else date.year + 1
    fin_month_str = f"{financial_year}-{financial_month:02d}"
    # Return as a period object
    return pd.Period(fin_month_str, freq='M')

# convert the maintenance interval to hours using the strategy and the average usage per day
##### NOTE: 100k hour strategies are conditional
def convert_interval_to_hours(interval, average_usage_per_day):
    time_period = interval[-1]
    if time_period == "H":
        return int(interval[:-1])
    elif time_period == "h":
        return int(interval[:-1])
    elif time_period == "M":
        return int(interval[:-1])*average_usage_per_day * 30.75
    elif time_period == "W":
        return int(interval[:-1])*average_usage_per_day * 7
    elif time_period == "D":
        return int(interval[:-1])*average_usage_per_day
    elif time_period == "K":
        return 100000
    elif time_period == "Y":
        return int(interval[:-1])*average_usage_per_day * 365.25
    else:
        raise ValueError(f"Invalid time period: {time_period}. Supported time periods are h, H, M, W, D, K, and Y.")