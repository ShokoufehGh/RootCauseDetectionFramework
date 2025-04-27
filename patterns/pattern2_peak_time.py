import pandas as pd
import numpy as np

def main(percentages, synonyms, csv_url, output_csv, period):
    print(f"input file{csv_url}")
    print(f"output file{output_csv}")
    
    try:
        df = pd.read_csv(csv_url, encoding="ISO-8859-1")  
    except UnicodeDecodeError:
        print("Encoding error! Trying another encoding...")
        df = pd.read_csv(csv_url, encoding="cp1252") 

    filtered = filtered_by_label_date(df, synonyms[0], period)
    filtered = filter_non_batch(filtered)
    split_dfs = select_for_update(filtered, percentages)    
    updated_df = update_each_synonym_set(df, split_dfs, synonyms)

    if(output_csv):
        updated_df.to_csv(output_csv, index=False)
        
    return updated_df, split_dfs
    
def update_each_synonym_set(df, split_dfs, synonyms):
    updated_df = df.copy()
    for i, split_df in enumerate(split_dfs):
        updated_df = update(updated_df, split_df['eventID '], synonyms[i])  
        print(f"Group {i+1}:\n{split_df}\n") 
    return updated_df

def select_for_update(df: pd.DataFrame, percentages):
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)    
    
    total_rows = len(df)    
    
    split_indices = []
    start_index = 0    
    for percentage in percentages:
       
        end_index = int(np.floor(total_rows * percentage)) + start_index
        split_indices.append((start_index, end_index))
        start_index = end_index    
    
    split_dfs = [df.iloc[start:end] for start, end in split_indices]    
    return split_dfs

def filtered_by_label_date(df, activity_label, period):      
    print(f"Count of rows: {len(df)}")    
    # df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], errors='coerce')
    df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], dayfirst=True, errors='coerce')
    filtered_by_label = df[df['event concept:name'] == activity_label]
    print(f"Count of rows filtered_by_label: {len(filtered_by_label)}")  
    filtered_by_label_date = compute_event_time_parameters(filtered_by_label, period)  
    print(f"Count of rows filtered_by_label_date: {len(filtered_by_label_date)}") 
    return filtered_by_label_date

def filter_non_batch(df):
    filter_non_batch = df[df['event User'].str.startswith('user', na=False)]
    print(f"Count of rows filter_non_batch: {len(filter_non_batch)}") 
    return filter_non_batch

def compute_event_time_parameters(df, method="year_month"):
    df_filtered = df.copy()
    df_filtered['event time:timestamp'] = pd.to_datetime(df_filtered['event time:timestamp'], errors='coerce')
    # df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    # df_2018 = df[df['event time:timestamp'].dt.year == 2018]
    # df_filtered = df[(df['event time:timestamp'].dt.year >= 2000) & (df['event time:timestamp'].dt.year <= 2050)]    
   
    if(method=="year_month"):
        df_filtered['year_month'] = df_filtered['event time:timestamp'].dt.to_period('M')        
        counts = df_filtered.groupby('year_month').size()            
        peak_year_month = counts.idxmax()
        print(f"peak_year_month: {peak_year_month}")
        # peak_count = counts.max()
        df_filtered = df_filtered[df_filtered['event time:timestamp'].dt.to_period('M') == peak_year_month]
    elif(method=="month_day"):
        # peak_month = df_filtered['event time:timestamp'].dt.month.value_counts().idxmax()
        peak_day_of_month = df_filtered['event time:timestamp'].dt.day.value_counts().idxmax()
        df_filtered = df_filtered[df_filtered['event time:timestamp'].dt.day == peak_day_of_month]
        print(f"peak_day_of_month: {peak_day_of_month}")
    elif(method=="week_day"):
        peak_day_of_week = df_filtered['event time:timestamp'].dt.dayofweek.value_counts().idxmax()
        df_filtered = df_filtered[df_filtered['event time:timestamp'].dt.dayofweek == peak_day_of_week]
        print(f"peak_day_of_week: {peak_day_of_week}")
        
    return df_filtered


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
