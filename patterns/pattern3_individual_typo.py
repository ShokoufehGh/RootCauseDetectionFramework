import pandas as pd
import numpy as np

def main(activity_label, percentage_user, percentage_update, synonyms, csv_url, output_csv, shuffle = True):
    print(f"input file{csv_url}")
    print(f"output file{output_csv}")
    
    try:
        df = pd.read_csv(csv_url, encoding="ISO-8859-1")  
    except UnicodeDecodeError:
        print("Encoding error! Trying another encoding...")
        df = pd.read_csv(csv_url, encoding="cp1252") 
        
    filtered = filtered_by_label(df, activity_label)
    filtered = filter_non_batch(filtered)
    filtered = filter_selected_user(filtered, percentage_user, shuffle)
    filtered = filter_part_rows(filtered, percentage_update)
    split_dfs = select_for_update(filtered, len(synonyms))
    updated_df = update_each_synonym_set(df, split_dfs, synonyms)    
    # print("Updated DataFrame after all changes:\n", updated_df)    
    
    if(output_csv):
        updated_df.to_csv(output_csv, index=False)
        
    return updated_df, split_dfs
    
def update_each_synonym_set(df, split_dfs, synonyms):
    updated_df = df.copy()
    for i, split_df in enumerate(split_dfs):
        updated_df = update(updated_df, split_df['eventID '], synonyms[i]) 
        print(f"Group {i+1}:\n{split_df}\n") 
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

def filtered_by_label(df, activity_label):      
    print(f"Count of rows: {len(df)}")    
    df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], errors='coerce', dayfirst=True)
    filtered_by_label = df[df['event concept:name'] == activity_label]
    print(f"Count of rows filtered_by_label: {len(filtered_by_label)}")
    return filtered_by_label

def filter_non_batch(df):
    filter_non_batch = df[df['event User'].str.startswith('user', na=False)]
    print(f"Count of rows filter_non_batch: {len(filter_non_batch)}") 
    return filter_non_batch

def filter_selected_user(df, percentage, shuffle):
    unique_users = df['event User'].dropna().unique()
    print(f"unique_users: {unique_users}") 
    
    user_counts = df['event User'].value_counts()    
    filtered_users = user_counts[user_counts > 50].index
    unique_users = sorted(filtered_users, key=lambda x: user_counts[x])
    print(f"user_counts: {user_counts}")
    print(f"unique_users: {unique_users}")
    # exit()
    

    
    
    num_users_to_select = int(np.floor(len(unique_users) * percentage))
    print(f"num_users_to_select: {num_users_to_select}")
    
    if (shuffle):
        selected_users = np.random.choice(unique_users, size=num_users_to_select, replace=False) 
    else:
        selected_users = unique_users[:num_users_to_select]
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
    # Find the rows to update
    rows_to_update = df[df['eventID '].isin(event_ids)].copy()    
    # # Print the rows before the update
    # print(f"Updating rows with eventID in {event_ids.tolist()} to synonym: '{synonym}'")
    # print("Rows before update:")
    # print(rows_to_update[['eventID ', 'event concept:name']])    
    # Update the 'event concept:name' column with the synonym
    df.loc[df['eventID '].isin(event_ids), 'event concept:name'] = synonym    
    # Print the rows after the update
    updated_rows = df[df['eventID '].isin(event_ids)]
    print("Rows after update:")
    print(updated_rows[['eventID ', 'event concept:name']])    
    return df
    
# main()
