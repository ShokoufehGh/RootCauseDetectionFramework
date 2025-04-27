import pandas as pd
import numpy as np

def main(activity_label, percentage_user, percentage_update, synonyms, csv_url, output_csv):
    print(f"input file{csv_url}")
    print(f"output file{output_csv}")    
    try:
        df = pd.read_csv(csv_url, encoding="ISO-8859-1")  
    except UnicodeDecodeError:
        print("Encoding error! Trying another encoding...")
        df = pd.read_csv(csv_url, encoding="cp1252")         
    filtered = filter_only_user(df)
    filtered = filter_by_users(filtered, percentage_user)
    filtered = filter_by_label(filtered, activity_label)    
    filtered = filter_part_rows(filtered, percentage_update)
    # print(f"filtered {len(filtered)}")
    # exit()
    split_dfs = select_for_update(filtered, len(synonyms))
    updated_df = update_each_synonym_set(df, split_dfs, synonyms)    
    # print("Updated DataFrame after all changes:\n", updated_df)    
    if(output_csv):
        updated_df.to_csv(output_csv, index=False)
        
    return updated_df, split_dfs
    
def filter_by_label(df, activity_label):    
    # Filter for specific activity
    activity_data = df[df['event concept:name'] == activity_label]
    print(f"Events matching activity label: {len(activity_data)}")  
    return activity_data
    
def filter_by_users(df, percentage):
    df = df.copy()
    df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], dayfirst=False, errors='coerce')    
    # Filter for 2018 data
    # df_2018 = df[df['event time:timestamp'].dt.year == 2018]
    # print(f"Total rows in 2018: {len(df_2018)}")
    # Calculate workload per user
    user_workload = calculate_user_workload(df)
    print("\nWorkload per user:")
    for user, count in user_workload.items():
        print(f"{user}: {count} events")  
            
    # Select top 10% of users based on workload
    selected_users = select_top_users(user_workload, percentage)
    print(f"\nSelected top users: {selected_users}")
    
    # Filter data for selected users
    selected_users_data = df[df['event User'].isin(selected_users)]
    print(f"\nTotal events for selected users: {len(selected_users_data)}")
    
    return selected_users_data
    

def select_top_users(user_workload, percentage=0.1):
    """Select top percentage of users based on workload."""
    n_users = int(len(user_workload) * percentage)
    sorted_users = sorted(user_workload.items(), key=lambda x: x[1], reverse=True)
    return [user for user, count in sorted_users[:n_users]]    

def calculate_user_workload(df):
    """Calculate the number of events per user."""
    return df['event User'].value_counts().to_dict()

def update_each_synonym_set(df, split_dfs, synonyms):
    updated_df = df.copy()
    for i, split_df in enumerate(split_dfs):
        updated_df = update(updated_df, split_df['eventID '], synonyms[i])  
        print(f"Group {i+1}:\n{split_df}\n") 
    return updated_df

# def select_for_update(df: pd.DataFrame, percentages):
#
#     df = df.sample(frac=1, random_state=42).reset_index(drop=True)    
#
#     total_rows = len(df)    
#
#     split_indices = []
#     start_index = 0    
#     for percentage in percentages:
#
#         end_index = int(np.floor(total_rows * percentage)) + start_index
#         split_indices.append((start_index, end_index))
#         start_index = end_index    
#
#     split_dfs = [df.iloc[start:end] for start, end in split_indices]  
#
#     # pd.set_option('display.max_columns', None)      
#     print(f"split_dfs: {split_dfs}")     
#     return split_dfs
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
    filter_only_user = df[df['event User'].str.startswith('user', na=False)]
    print(f"Count of rows filter_only_user: {len(filter_only_user)}") 
    return filter_only_user

def compute_event_time_parameters(df):
    df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], errors='coerce')
    # df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df_2018 = df[df['event time:timestamp'].dt.year == 2018]
    df_filtered = df[(df['event time:timestamp'].dt.year >= 2017) & (df['event time:timestamp'].dt.year <= 2019)]
    peak_month = df_filtered['event time:timestamp'].dt.month.value_counts().idxmax()
    peak_day_of_month = df_filtered['event time:timestamp'].dt.day.value_counts().idxmax()
    peak_day_of_week = df_filtered['event time:timestamp'].dt.dayofweek.value_counts().idxmax()
    return peak_month, peak_day_of_month, peak_day_of_week

def update(df: pd.DataFrame, event_ids: pd.Series, synonym: str):
    # Find the rows to update
    rows_to_update = df[df['eventID '].isin(event_ids)].copy()   
    df.loc[df['eventID '].isin(event_ids), 'event concept:name'] = synonym
    updated_rows = df[df['eventID '].isin(event_ids)]    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("Rows after update:")
    print(updated_rows[['eventID ', 'event concept:name']])  
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    return df
    
# main()

    #
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
