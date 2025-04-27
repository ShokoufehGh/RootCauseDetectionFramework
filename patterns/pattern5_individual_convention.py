import pandas as pd
import numpy as np
from random import shuffle
from itertools import combinations

def main(percentages, synonyms, csv_url, output_csv, shuffle=True, event_seprated=True):
    print(f"input file{csv_url}")
    print(f"output file{output_csv}")
    
    try:
        df = pd.read_csv(csv_url, encoding="ISO-8859-1")  
    except UnicodeDecodeError:
        print("Encoding error! Trying another encoding...")
        df = pd.read_csv(csv_url, encoding="cp1252") 

    filtered = filtered_by_label(df, synonyms[0])
    filtered = filter_user(filtered)
    
    if(event_seprated):
        split_dfs = select_for_update_new(filtered, percentages)
    else:
        split_dfs = select_for_update(filtered, percentages, shuffle)
    updated_df = update_each_synonym_set(df, split_dfs, synonyms)    
    # print("Updated DataFrame after all changes:\n", updated_df)  
    if(output_csv):
        updated_df.to_csv(output_csv, index=False)
        
    return updated_df, split_dfs

def select_for_update_new(df, percentages):
    selected_users = set()
    grouped_batch = []
    
    
    remainded = 100
    for p in percentages:
        pure_percentage = p * 100
        target_percentage =  (100 * pure_percentage) / remainded
        remainded = remainded - pure_percentage
        
        temp_df = df[~df['event User'].isin(selected_users)]
        
        best_combination, _, _ = print_percentage_of_users(temp_df, target_percentage)
        
        grouped_batch.append(list(best_combination))
        
        selected_users.update(best_combination)
    
    print(f"grouped_batch: {grouped_batch}")
    
    split_dfs = []
    for batch_group in grouped_batch:
        split_df = df[df['event User'].isin(batch_group)]
        split_dfs.append(split_df)
        print(f"Split DataFrame for batch group {batch_group}:\n{split_df}\n")
    
    return split_dfs

def print_percentage_of_users(df, target_percentage):
    total_events = len(df)
    
    user_event_counts = df['event User'].value_counts()
    
    for user, count in user_event_counts.items():
        percentage = (count / total_events) * 100
        print(f"User {user}: {percentage:.2f}% of events")
    
    user_percentages = (user_event_counts / total_events) * 100
    
    user_percent_dict = user_percentages.to_dict()
    
    best_combination = []
    best_sum = 0
    min_diff = float('inf')  
    
    for r in range(1, len(user_percent_dict) + 1):
        for combo in combinations(user_percent_dict.keys(), r):
            combo_sum = sum(user_percent_dict[user] for user in combo)
            diff = abs(combo_sum - target_percentage)
            
            if diff < min_diff:
                min_diff = diff
                best_combination = combo
                best_sum = combo_sum
    
    print(f"Best combination of users for {target_percentage}%:")
    for user in best_combination:
        print(f"User {user}: {user_percent_dict[user]:.2f}% of events")
    print(f"Total percentage: {best_sum:.2f}%")
    
    print('----------------------------------')
    return best_combination, user_percent_dict, best_sum
    
def update_each_synonym_set(df, split_dfs, synonyms):
    updated_df = df.copy()
    for i, split_df in enumerate(split_dfs):
        updated_df = update(updated_df, split_df['eventID '], synonyms[i])  
        # print(f"Group {i+1}:\n{split_df}\n") 
    return updated_df

def select_for_update(df, percentages, shuffle):    
    unique_batch = df['event User'].dropna().unique()
    print(f"unique_users: {unique_batch}")      
    
    user_counts = df['event User'].value_counts()    
    filtered_users = user_counts[user_counts > 100].index
    unique_batch = sorted(filtered_users, key=lambda x: user_counts[x])
    
    if shuffle:
        np.random.shuffle(unique_batch)
        print(f"unique_users after shuffling: {unique_batch}")    
    
    count_unique_batch = len(unique_batch)
    group_sizes = [round(count_unique_batch * p) for p in percentages]
    
    # Ensure the sum of group_sizes equals count_unique_batch
    if sum(group_sizes) != count_unique_batch:
        group_sizes[-1] = count_unique_batch - sum(group_sizes[:-1])
    
    # Split the unique_batch into groups based on group_sizes
    grouped_batch = []
    start = 0
    for size in group_sizes:
        grouped_batch.append(unique_batch[start:start+size])
        start += size
    
    print(f"grouped_batch: {grouped_batch}") 
    print(f"count_unique_batch: {count_unique_batch}") 
    print(f"group_sizes: {group_sizes}") 
    
    split_dfs = []
    for batch_group in grouped_batch:
        split_df = df[df['event User'].isin(batch_group)]
        split_dfs.append(split_df)
        print(f"Split DataFrame for batch group {batch_group}:\n{split_df}\n")
    
    return split_dfs

def filtered_by_label(df, activity_label):      
    print(f"Count of rows: {len(df)}")    
    df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], errors='coerce', dayfirst=False)
    filtered_by_label = df[df['event concept:name'] == activity_label]
    print(f"Count of rows filtered_by_label: {len(filtered_by_label)}")
    return filtered_by_label

# def filter_user(df):
#     filter_user = df[df['event User'].str.startswith('user', na=False)]
#     print(f"Count of rows filter_user: {len(filter_user)}") 
#     return filter_user


def filter_user(df):
    # Filter rows where 'event User' starts with 'user'
    filtered_df = df[df['event User'].str.startswith('user', na=False)]
    print(f"Count of rows filtered_df: {len(filtered_df)}")
    
    # Calculate the total number of records
    total_records = len(filtered_df)
    
    # Calculate the threshold for 1% of total records
    threshold = total_records * 0.02
    
    # Count the occurrences of each user
    user_counts = filtered_df['event User'].value_counts()
    
    # Identify users with at least 1% of total records
    valid_users = user_counts[user_counts >= threshold].index
    
    # Filter the dataframe to keep only valid users
    result_df = filtered_df[filtered_df['event User'].isin(valid_users)]
    
    print(f"Count of rows after filtering users with <1% records: {len(result_df)}")
    return result_df


def filter_selected_user(df, percentage):
    unique_users = df['event User'].dropna().unique()
    print(f"unique_users: {unique_users}")     
    num_users_to_select = int(np.floor(len(unique_users) * percentage))
    print(f"num_users_to_select: {num_users_to_select}")    
    selected_users = np.random.choice(unique_users, size=num_users_to_select, replace=False) 
    print(f"selected_users: {selected_users}")    
    filtered_df = df[df['event User'].isin(selected_users)] 
    print(f"num of rows for these users: {len(filtered_df)}")
    return filtered_df

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
    rows_to_update = df[df['eventID '].isin(event_ids)].copy()    
    # # Print the rows before the update
    # print(f"Updating rows with eventID in {event_ids.tolist()} to synonym: '{synonym}'")
    # print("Rows before update:")
    # print(rows_to_update[['eventID ', 'event concept:name']])        
    df.loc[df['eventID '].isin(event_ids), 'event concept:name'] = synonym        
    updated_rows = df[df['eventID '].isin(event_ids)]
    # print("Rows after update:")
    # print(updated_rows[['eventID ', 'event concept:name']])    
    return df
    
# main()
