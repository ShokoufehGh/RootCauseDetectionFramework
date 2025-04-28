import pandas as pd
import numpy as np

def main():
    return

def filter_sub_log(df, concept_name, labels_set):
    df = df.copy() #  'event concept:name' is concept_name
    print(f"Original input size: {len(df)}")
    df = df[df[concept_name].isin(labels_set)]
    print(f"Sub log input size: {len(df)}")
    return df

def calculate_peak_year_month(df):
    df['year_month'] = df['event time:timestamp'].dt.to_period('M')
    counts = df.groupby('year_month').size()
    peak_year_month = counts.idxmax()
    print(f"peak_year_month: {peak_year_month}")
    df['is_peak_year_month'] = (df['year_month'] == peak_year_month).astype(int)    
    return df, peak_year_month

def calculate_peak_day_of_month(df):
    peak_day_of_month = df['event time:timestamp'].dt.day.value_counts().idxmax()
    print(f"peak_day_of_month: {peak_day_of_month}")
    df['is_peak_day_of_month'] = (df['event time:timestamp'].dt.day == peak_day_of_month).astype(int)    
    return df, peak_day_of_month

def calculate_peak_day_of_week(df):
    peak_day_of_week = df['event time:timestamp'].dt.dayofweek.value_counts().idxmax()
    print(f"peak_day_of_week: {peak_day_of_week}")
    df['is_peak_day_of_week'] = (df['event time:timestamp'].dt.dayofweek == peak_day_of_week).astype(int)    
    return df, peak_day_of_week

def select_top_users(user_workload, percentage=0.1):
    n_users = int(len(user_workload) * percentage)
    sorted_users = sorted(user_workload.items(), key=lambda x: x[1], reverse=True)
    return [user for user, count in sorted_users[:n_users]]    

def calculate_user_workload(df):
    """Calculate the number of events per user."""
    return df['event User'].value_counts().to_dict()





def calculate_month_of_year_workload_without_system(df): #peak month of year   

    # human_df = df[~df['resource_type'].str.startswith('batch')].copy()
    
    human_df = df[df['event User'].str.startswith('user_')].copy()

    
    df['year_month'] = df['event time:timestamp'].dt.to_period('M')
    human_df['year_month'] = human_df['event time:timestamp'].dt.to_period('M')
    
    monthly_counts = human_df.groupby('year_month').size()
    
    peak_year_month = monthly_counts.idxmax()
    print(f"peak_year_month: {peak_year_month}") 
    df['month_of_year_workload'] = df['year_month'].map(monthly_counts)
    
    return df, peak_year_month


def calculate_day_of_month_workload_without_system(df): #peak day of month

    # human_df = df[~df['resource_type'].str.startswith('batch')].copy() 
    
    human_df = df[df['event User'].str.startswith('user_')].copy()   
    df['day_of_month'] = df['event time:timestamp'].dt.day
    human_df['day_of_month'] = human_df['event time:timestamp'].dt.day
    
    daily_counts = human_df['day_of_month'].value_counts()
    
    peak_day_of_month = daily_counts.idxmax()
    print(f"peak_day_of_month: {peak_day_of_month}")   
    df['day_of_month_workload'] = df['day_of_month'].map(daily_counts)
    
    return df, peak_day_of_month

def calculate_day_of_week_workload_without_system(df): #peak day of week

    # human_df = df[~df['resource_type'].str.startswith('batch')].copy()
    human_df = df[df['event User'].str.startswith('user_')].copy()

    
    df['day_of_week'] = df['event time:timestamp'].dt.dayofweek
    human_df['day_of_week'] = human_df['event time:timestamp'].dt.dayofweek
    
    weekday_counts = human_df['day_of_week'].value_counts()   
    peak_day_of_week = weekday_counts.idxmax()
    print(f"peak_day_of_week: {peak_day_of_week}")
    
    df['day_of_week_workload'] = df['day_of_week'].map(weekday_counts)
    
    return df, peak_day_of_week








def calculate_month_of_year_workload(df):  #peak month of year
    """
    Calculate the number of events per month and identify the peak month.
    Returns the modified dataframe and the peak month period.
    """
    # Convert timestamp to year-month period
    df['year_month'] = df['event time:timestamp'].dt.to_period('M')
    
    # Count events per month
    monthly_counts = df.groupby('year_month').size()
    
    # Find peak month
    peak_year_month = monthly_counts.idxmax()
    print(f"peak_year_month: {peak_year_month}")
    
    # Map the monthly counts back to each row
    df['month_of_year_workload'] = df['year_month'].map(monthly_counts)
    
    return df, peak_year_month


def calculate_day_of_month_workload(df):  #peak day of month
    """
    Calculate the number of events per day of month and identify the peak day.
    Returns the modified dataframe and the peak day of month.
    """
    # Get day of month for each timestamp
    df['day_of_month'] = df['event time:timestamp'].dt.day
    
    # Count events per day of month
    daily_counts = df['day_of_month'].value_counts()
    
    # Find peak day
    peak_day_of_month = daily_counts.idxmax()
    print(f"peak_day_of_month: {peak_day_of_month}")
    
    # Map the daily counts back to each row
    df['day_of_month_workload'] = df['day_of_month'].map(daily_counts)
    
    return df, peak_day_of_month
    
    
def calculate_day_of_week_workload(df):  #peak day of week

    """
    Calculate the number of events per day of week and identify the peak day.
    Returns the modified dataframe and the peak day of week.
    """
    # Get day of week for each timestamp
    df['day_of_week'] = df['event time:timestamp'].dt.dayofweek
    
    # Count events per day of week
    weekday_counts = df['day_of_week'].value_counts()
    
    # Find peak day
    peak_day_of_week = weekday_counts.idxmax()
    print(f"peak_day_of_week: {peak_day_of_week}")
    
    # Map the weekday counts back to each row
    df['day_of_week_workload'] = df['day_of_week'].map(weekday_counts)
    
    return df, peak_day_of_week

    
def print_all_rows(element, stop):    
    pd.set_option('display.max_rows', None)
    print(element)
    pd.set_option('display.max_rows', None)
    if(stop):
        exit()
        
def printOneShapValue(label_set, row_labels, shap_value):    
    print("\t".join(label_set))
    for i, row in enumerate(shap_value):
        rounded_values = [f"{value:.4f}" for value in row]
        print("\t".join(rounded_values) + "\t" + row_labels[i])
        
