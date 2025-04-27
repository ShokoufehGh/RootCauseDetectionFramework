# pattern_definition.py
import pandas as pd
from typing import List
from typing import Dict, List, Tuple

class PatternDefinition:
    def __init__(self, percentages: List[float]):
        self.percentages = percentages

    def load_csv(self, file_path: str, encoding: str = 'ISO-8859-1') -> pd.DataFrame:
        """
        Load a CSV file into a Pandas DataFrame.

        :param file_path: Path or URL to the CSV file.
        :param encoding: Encoding type for reading the file.
        :return: DataFrame containing the CSV data.
        """
        return pd.read_csv(file_path, encoding=encoding)

    def filter_by_activity(self, df: pd.DataFrame, activity_label: str) -> pd.DataFrame:
        """
        Filter the DataFrame for rows with a specific activity label.

        :param df: Input DataFrame.
        :param activity_label: Activity label to filter on.
        :return: Filtered DataFrame.
        """
        return df[df['event concept:name'] == activity_label]
    
    def update_activity_labels(self, df: pd.DataFrame, groups: Dict[str, List[str]], target_label: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Update activity labels in the DataFrame based on user groups.
    
        :param df: Input DataFrame.
        :param groups: Dictionary with group names and their respective users.
        :param target_label: The activity label to be updated.
        :return: Tuple containing the updated DataFrame and a list of change reports.
        """
        changes = []
    
        for index, row in df.iterrows():
            if row['event concept:name'] == target_label:
                user = row['event User']
                new_label = next((group for group, users in groups.items() if user in users), row['event concept:name'])
    
                # If the label changes, update and log the change
                if new_label != row['event concept:name']:
                    df.at[index, 'event concept:name'] = new_label
                    changes.append(f"User {user}: {row['event concept:name']} -> {new_label}")
    
        return df, changes
    
    def update_activity_labels_by_date(self, df: pd.DataFrame, groups: Dict[str, List[str]], target_label: str, target_date: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Update activity labels in the DataFrame based on user groups and month-year of the event.
    
        :param df: Input DataFrame.
        :param groups: Dictionary with group names and their respective users.
        :param target_label: The activity label to be updated.
        :return: Tuple containing the updated DataFrame and a list of change reports.
        """
        changes = []
    
        for index, row in df.iterrows():
            if row['event concept:name'] == target_label:
                user = row['event User']
                event_date = pd.to_datetime(row['event time:timestamp'])
                month_year = event_date.strftime('%Y-%m')#2018-10
                month_day = event_date.strftime('%Y--%d')#2018--26
                day_of_week = event_date.strftime('w%A')#w3
                date_condition = month_year == target_date or month_day == target_date or day_of_week == target_date
    
                new_label = next((group for group, users in groups.items() if user in users), row['event concept:name'])
    
                # If the label changes, update and log the change
                if new_label != row['event concept:name'] and date_condition  :
                    df.at[index, 'event concept:name'] = new_label
                    changes.append(f"User {user} on {month_year}: {row['event concept:name']} -> {new_label}")
    
        return df, changes    
    
