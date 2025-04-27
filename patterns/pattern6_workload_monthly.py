import pandas as pd
import numpy as np
import random
from patterns.pattern4_overall_workload import update



def main(activity_label, percentage_user, percentage_update, synonyms, csv_url, output_csv,minimum_day_activity, shuffle):
    print(f"input file{csv_url}")
    print(f"output file{output_csv}")
    try:
        df = pd.read_csv(csv_url, encoding="ISO-8859-1")  
    except UnicodeDecodeError:
        print("Encoding error! Trying another encoding...")
        df = pd.read_csv(csv_url, encoding="cp1252")   
    filtered = filter_only_user(df)
    filtered = filter_by_users(filtered, percentage_user, shuffle)
    filtered = filter_by_users_month(filtered, minimum_day_activity)
    filtered = filter_by_label(filtered, activity_label)    
    filtered = filter_part_rows(filtered, percentage_update)
    split_dfs = select_for_update(filtered, len(synonyms))
    updated_df = update_each_synonym_set(df, split_dfs, synonyms)    
    if(output_csv):
        updated_df.to_csv(output_csv, index=False)
        
    return updated_df, split_dfs
    
def filter_by_label(df, activity_label):    
    # Filter for specific activity
    activity_data = df[df['event concept:name'] == activity_label]
    print(f"Events matching activity label: {len(activity_data)}")  
    return activity_data

def filter_by_users_month(df, minimum_activity_per_month=5):
    """
    Filter users based on their monthly activity patterns.
    Selects users who are active at least a minimum number of events per month and returns their peak activity days.
    """
    df = df.copy()
    df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], errors='coerce')
    
    # Create year, month, and day columns for easier grouping
    df['year'] = df['event time:timestamp'].dt.year
    df['month'] = df['event time:timestamp'].dt.month
    df['day'] = df['event time:timestamp'].dt.day
    
    # Initialize list to store selected rows
    selected_rows = []
    
    # Process each year and month
    for (year, month), monthly_data in df.groupby(['year', 'month']):
        
        # Count total events per user in this month
        # user_total_events = monthly_data.groupby('event User').size()
        # active_users = user_total_events[user_total_events >= minimum_activity_per_month].index.tolist().# Count unique active days per user in the current month
        user_active_days = monthly_data.groupby('event User')['day'].nunique()
        active_users = user_active_days[user_active_days >= minimum_activity_per_month].index.tolist()

        
        if active_users:
            # For each active user, find their busiest day
            for user in active_users:
                user_data = monthly_data[monthly_data['event User'] == user]
                
                # Group by day and count events
                daily_counts = user_data.groupby('day').size()
                
                # Find the day with maximum events
                busiest_day = daily_counts.idxmax()
                
                # Select all rows for this user on their busiest day
                busiest_day_rows = user_data[
                    user_data['day'] == busiest_day
                ]
                
                selected_rows.append(busiest_day_rows)
    
    # Combine all selected rows
    if selected_rows:
        result_df = pd.concat(selected_rows)
    else:
        result_df = pd.DataFrame(columns=df.columns)
    
    # Print detailed statistics
    print("\nDetailed Monthly Statistics:")
    for (year, month), monthly_rows in result_df.groupby(['year', 'month']):
        print(f"\nYear {year}, Month {month}:")
        print(f"Total events selected: {len(monthly_rows)}")
        print("Users and their event counts:")
        user_counts = monthly_rows['event User'].value_counts()
        for user, count in user_counts.items():
            print(f"  {user}: {count} events")
    
    print(f"\nTotal events in final selection: {len(result_df)}")
    # print(f"re {result_df}")
    # exit()
    
    return result_df 

def filter_by_users(df, percentage, shuffle):
    df = df.copy()
    df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], dayfirst=True, errors='coerce')    
    # df_2018 = df[df['event time:timestamp'].dt.year == 2018]
    # print(f"Total rows in 2018: {len(df_2018)}")
    user_workload = calculate_user_workload(df)
    print("\nWorkload per user:")
    for user, count in user_workload.items():
        print(f"{user}: {count} events")  
            
    selected_users = select_users_by_shuffle(user_workload, percentage, shuffle)
    print(f"\nSelected top users: {selected_users}")
    
    selected_users_data = df[df['event User'].isin(selected_users)]
    print(f"\nTotal events for selected users: {len(selected_users_data)}")
    
    return selected_users_data


def select_users_by_shuffle(user_workload, percentage, shuffle):
    """Select a percentage of users randomly without considering their workload order."""
    n_users = int(len(user_workload) * percentage)
    users = list(user_workload.keys())
    if(shuffle): 
        random.shuffle(users)
    return users[:n_users]

def select_top_users(user_workload, percentage):
    """Select top percentage of users based on workload."""
    n_users = int(len(user_workload) * percentage)
    sorted_users = sorted(user_workload.items(), key=lambda x: x[1], reverse=True)
    return [user for user, count in sorted_users[:n_users]]    

def calculate_user_workload(df):
    return df['event User'].value_counts().to_dict()

def update_each_synonym_set(df, split_dfs, synonyms):
    updated_df = df.copy()
    for i, split_df in enumerate(split_dfs):
        updated_df = update(updated_df, split_df['eventID '], synonyms[i]) 
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(f"Group {i+1}:\n{split_df}\n")  
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
    return updated_df

def filter_part_rows(filtered_df, percentage_update):    
    shuffled_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)       
    num_rows_to_select = int(len(shuffled_df) * percentage_update)       
    selected_df = shuffled_df.iloc[:num_rows_to_select]    
    return selected_df

def select_for_update(selected_df, group_size):    
    shuffled_df = selected_df.sample(frac=1, random_state=42).reset_index(drop=True) 
    split_dfs = np.array_split(shuffled_df, group_size)
    
    return split_dfs

def filtered_by_label_date(df, activity_label):      
    print(f"Count of rows: {len(df)}")    
    df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], errors='coerce')
    filtered_by_label = df[df['event concept:name'] == activity_label]
    print(f"Count of rows filtered_by_label: {len(filtered_by_label)}")    
    peak_month, peak_day_of_month, peak_day_of_week = compute_event_time_parameters(df)
    filtered_by_label_date = filtered_by_label[
        (filtered_by_label['event time:timestamp'].dt.year == 2018) & 
        (filtered_by_label['event time:timestamp'].dt.month == 1)
    ]
    print(f"Count of rows filtered_by_label_date: {len(filtered_by_label_date)}") 
    return filtered_by_label_date

def filter_only_user(df):
    filter_non_batch = df[df['event User'].str.startswith('user', na=False)]
    print(f"Count of rows filter_non_batch: {len(filter_non_batch)}") 
    return filter_non_batch

def compute_event_time_parameters(df):
    df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], errors='coerce')
    # df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df_2018 = df[df['event time:timestamp'].dt.year == 2018]
    df_filtered = df[(df['event time:timestamp'].dt.year >= 2017) & (df['event time:timestamp'].dt.year <= 2019)]
    peak_month = df_filtered['event time:timestamp'].dt.month.value_counts().idxmax()
    peak_day_of_month = df_filtered['event time:timestamp'].dt.day.value_counts().idxmax()
    peak_day_of_week = df_filtered['event time:timestamp'].dt.dayofweek.value_counts().idxmax()
    return peak_month, peak_day_of_month, peak_day_of_week


    
# main()
