import pandas as pd
from common_functions import (
    calculate_user_workload,
    calculate_month_of_year_workload,
    calculate_day_of_week_workload,
    calculate_day_of_month_workload,
)
from pattern__injection_runner import Logger
from datetime import datetime
from scipy.stats import percentileofscore


class DataProcessor:
    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_csv, encoding="ISO-8859-1")
        except UnicodeDecodeError:
            print("Encoding error! Trying another encoding...")
            self.df = pd.read_csv(self.input_csv, encoding="cp1252")

    def add_new_features(self):
        self.df['event time:timestamp'] = pd.to_datetime(
            self.df['event time:timestamp'], dayfirst=True, errors='coerce'
        )
        self.df['event time:timestamp'] = pd.to_datetime(self.df['event time:timestamp'])
        user_workload = calculate_user_workload(self.df)
        self.df['user_workload'] = self.df['event User'].map(user_workload)
        self.df = self.calculate_user_daily_workload()
        self.df, peak_year_month = calculate_month_of_year_workload(self.df)
        self.df, peak_day_of_month = calculate_day_of_month_workload(self.df)
        self.df, peak_day_of_week = calculate_day_of_week_workload(self.df)

        # Add resource_type feature
        self.df['resource_type'] = self.df['event User'].apply(
            lambda x: 'human' if x.startswith('user') else 'system'
        )
        self.df['resource'] = self.df['event User']

        # Extract day, month, year, and day of week from the timestamp
        self.df['day_of_month'] = self.df['event time:timestamp'].dt.day
        self.df['month'] = self.df['event time:timestamp'].dt.month
        self.df['year'] = self.df['event time:timestamp'].dt.year
        self.df['day_of_week'] = self.df['event time:timestamp'].dt.dayofweek  # 0 = Monday

        # Set workload features to NULL for 'batch' or None users
        workload_features = [
            'user_day_of_month_workload',
            'user_workload',
            'month_of_year_workload',
            'day_of_month_workload',
            'day_of_week_workload',
        ]

        for feature in workload_features:
            self.df.loc[
                (self.df['resource_type'].str.startswith('system')), feature
            ] = None

        print("File saved to output")
        self.df.to_csv(self.output_csv, index=False)

    def calculate_user_daily_workload(self):
        df = self.df.copy()

        # Convert timestamp to datetime if not already
        df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], errors='coerce')

        # Extract time components
        df['year'] = df['event time:timestamp'].dt.year
        df['month'] = df['event time:timestamp'].dt.month
        df['day_of_month'] = df['event time:timestamp'].dt.day

        # Calculate workload for each user on each day
        user_day_of_month_workload = (
            df.groupby(['event User', 'year', 'month', 'day_of_month'])
            .size()
            .reset_index()
        )
        user_day_of_month_workload.columns = [
            'event User',
            'year',
            'month',
            'day_of_month',
            'user_day_of_month_workload',
        ]

        # Merge workload back to original dataframe
        df = df.merge(
            user_day_of_month_workload,
            on=['event User', 'year', 'month', 'day_of_month'],
            how='left',
        )

        # Print summary statistics
        print("\nDaily Workload Statistics:")
        for (year, month), monthly_data in df.groupby(['year', 'month']):
            print(f"\nYear {year}, Month {month}:")
            for user, user_data in monthly_data.groupby('event User'):
                daily_stats = user_data.groupby('day_of_month')[
                    'user_day_of_month_workload'
                ].first()
                print(f"\nUser {user} user day of month workload:")
                for day, workload in daily_stats.items():
                    print(f"  Day {day}: {workload} events")

        return df


# # Example usage
# if __name__ == "__main__":
#     csv_url = 'data/BPI_Challenge_2019.csv'
#     output_csv = 'data/BPI_with_new_numeric_features_v3_with_user.csv'
#     processor = DataProcessor(csv_url, output_csv)
#     processor.load_data()
#     processor.add_new_features()