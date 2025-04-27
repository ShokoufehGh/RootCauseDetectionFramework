

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict
import sys

def read_csv_with_encoding(file_path: str) -> pd.DataFrame:
  
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding} encoding: {str(e)}")
            continue
    
    raise ValueError(f"Could not read the file with any of these encodings: {encodings}")

def analyze_specific_features(csv_file: str):
 
    selected_features = [
        'user_day_of_month_workload',
        'user_workload',
        # 'resource_type',
        'day_of_month',
        'month',
        # 'year',
        'day_of_week',
        'month_of_year_workload',
        'day_of_month_workload',
        'day_of_week_workload',
        'resource'
    ]
    
    try:
        print("Reading CSV file...")
        df = read_csv_with_encoding(csv_file)
        
        print("\nChecking for requested features...")
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            print(f"Warning: These features were not found in the dataset: {missing_features}")
            selected_features = [f for f in selected_features if f in df.columns]
        
        print(f"\nAnalyzing the following features: {', '.join(selected_features)}")
        
        analysis_df = df[selected_features].copy()
        
        for column in analysis_df.columns:
            if analysis_df[column].dtype == 'object':
                print(f"\nConverting categorical column '{column}' to numeric...")
                analysis_df[column] = pd.Categorical(analysis_df[column]).codes
        
        print("\nCalculating correlations...")
        correlation_matrix = analysis_df.corr()
        
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                correlation = correlation_matrix.iloc[i, j]
                if abs(correlation) > 0.5:  # Threshold for strong correlation
                    strong_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': correlation
                    })
        
        print("\nCreating correlation heatmap...")
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(correlation_matrix, 
                    annot=True,
                    cmap='coolwarm',
                    center=0,
                    fmt='.2f',
                    square=True,
                    annot_kws={'size': 8})
        
        plt.title('Feature Dependency', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        print("\nCorrelation Matrix:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(correlation_matrix.round(3))
        
        if strong_correlations:
            print("\nStrong Feature Correlations (|correlation| > 0.5):")
            for corr in strong_correlations:
                print(f"{corr['feature1']} -- {corr['feature2']}: {corr['correlation']:.3f}")
        else:
            print("\nNo strong correlations (> 0.5) found between the features.")
        
        correlation_matrix.round(3).to_csv('correlation_matrix.csv')
        print("\nCorrelation matrix has been saved to 'correlation_matrix.csv'")
        
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Heatmap has been saved as 'correlation_heatmap.png'")
        
        plt.show()
        
        return correlation_matrix, strong_correlations
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def main():
    try:
        # csv_file = input("data/BPI_with_new_numeric_features_v1_with_user.csv").strip()
        # csv_file = 'data/BPI_with_new_numeric_features_v1_with_user.csv'
        csv_file = 'data/BPI_with_new_numeric_features_v3_with_user.csv'

        
        correlation_matrix, strong_correlations = analyze_specific_features(csv_file)
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from typing import Tuple, List, Dict
# import sys
#
# def read_csv_with_encoding(file_path: str) -> pd.DataFrame:
#     """
#     Try to read a CSV file with different encodings.
#
#     Parameters:
#     file_path (str): Path to the CSV file
#
#     Returns:
#     pd.DataFrame: The loaded DataFrame
#     """
#     encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
#
#     for encoding in encodings:
#         try:
#             return pd.read_csv(file_path, encoding=encoding)
#         except UnicodeDecodeError:
#             continue
#         except Exception as e:
#             print(f"Error reading file with {encoding} encoding: {str(e)}")
#             continue
#
#     raise ValueError(f"Could not read the file with any of these encodings: {encodings}")
#
# def analyze_correlations(csv_file: str) -> Tuple[pd.DataFrame, List[Dict]]:
#     """
#     Analyze correlations between features in a CSV file.
#
#     Parameters:
#     csv_file (str): Path to the CSV file
#
#     Returns:
#     tuple: (correlation_matrix, correlation_summary)
#     """
#     try:
#         # Read the CSV file with appropriate encoding
#         df = read_csv_with_encoding(csv_file)
#
#         # Select only numeric columns
#         numeric_df = df.select_dtypes(include=['int64', 'float64'])
#
#         if numeric_df.empty:
#             raise ValueError("No numeric columns found in the dataset")
#
#         # Calculate correlation matrix
#         correlation_matrix = numeric_df.corr()
#
#         # Create a summary of strong correlations
#         strong_correlations = []
#         for i in range(len(correlation_matrix.columns)):
#             for j in range(i):
#                 if abs(correlation_matrix.iloc[i, j]) > 0.5:  # Threshold for strong correlation
#                     strong_correlations.append({
#                         'feature1': correlation_matrix.columns[i],
#                         'feature2': correlation_matrix.columns[j],
#                         'correlation': correlation_matrix.iloc[i, j]
#                     })
#
#         # Create heatmap
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(correlation_matrix, 
#                     annot=True,
#                     cmap='coolwarm',
#                     center=0,
#                     fmt='.2f')
#         plt.title('Feature Correlation Heatmap')
#         plt.xticks(rotation=45, ha='right')
#         plt.yticks(rotation=0)
#         plt.tight_layout()
#
#         return correlation_matrix, strong_correlations
#
#     except Exception as e:
#         print(f"Error during correlation analysis: {str(e)}")
#         sys.exit(1)
#
# def main():
#     # Replace with your actual file path
#     csv_file = 'data/BPI_with_new_numeric_features_v1_with_user.csv'
#
#     try:
#         correlation_matrix, strong_correlations = analyze_correlations(csv_file)
#
#         # Print strong correlations
#         print("\nStrong Feature Correlations (|correlation| > 0.5):")
#         for corr in strong_correlations:
#             print(f"{corr['feature1']} -- {corr['feature2']}: {corr['correlation']:.3f}")
#
#         # Show the heatmap
#         plt.show()
#
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         sys.exit(1)
#
# if __name__ == "__main__":
#     main()



# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from typing import Tuple, List, Dict
# import sys
#
# def read_csv_with_encoding(file_path: str) -> pd.DataFrame:
#     """
#     Try to read a CSV file with different encodings.
#
#     Parameters:
#     file_path (str): Path to the CSV file
#
#     Returns:
#     pd.DataFrame: The loaded DataFrame
#     """
#     encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
#
#     for encoding in encodings:
#         try:
#             return pd.read_csv(file_path, encoding=encoding)
#         except UnicodeDecodeError:
#             continue
#         except Exception as e:
#             print(f"Error reading file with {encoding} encoding: {str(e)}")
#             continue
#
#     raise ValueError(f"Could not read the file with any of these encodings: {encodings}")
#
# def analyze_correlations(csv_file: str) -> Tuple[pd.DataFrame, List[Dict]]:
#     """
#     Analyze correlations between features in a CSV file.
#
#     Parameters:
#     csv_file (str): Path to the CSV file
#
#     Returns:
#     tuple: (correlation_matrix, correlation_summary)
#     """
#     try:
#         # Read the CSV file with appropriate encoding
#         df = read_csv_with_encoding(csv_file)
#
#         # Select only numeric columns
#         numeric_df = df.select_dtypes(include=['int64', 'float64'])
#
#         if numeric_df.empty:
#             raise ValueError("No numeric columns found in the dataset")
#
#         # Calculate correlation matrix
#         correlation_matrix = numeric_df.corr()
#
#         # Create a summary of strong correlations
#         strong_correlations = []
#         for i in range(len(correlation_matrix.columns)):
#             for j in range(i):
#                 if abs(correlation_matrix.iloc[i, j]) > 0.5:  # Threshold for strong correlation
#                     strong_correlations.append({
#                         'feature1': correlation_matrix.columns[i],
#                         'feature2': correlation_matrix.columns[j],
#                         'correlation': correlation_matrix.iloc[i, j]
#                     })
#
#         # Calculate figure size based on number of features
#         n_features = len(correlation_matrix.columns)
#         fig_size = max(12, n_features * 0.5)  # Dynamically adjust figure size
#
#         # Create heatmap with adjusted parameters
#         plt.figure(figsize=(fig_size, fig_size))
#
#         # Adjust font sizes based on number of features
#         font_size = max(8, min(10, 300 / n_features))  # Dynamically adjust font size
#
#         sns.heatmap(correlation_matrix, 
#                     annot=True,
#                     cmap='coolwarm',
#                     center=0,
#                     fmt='.2f',
#                     annot_kws={'size': font_size},  # Adjust annotation font size
#                     square=True)  # Make the plot square
#
#         plt.title('Feature Correlation Heatmap', pad=20, fontsize=font_size + 2)
#
#         # Rotate labels and adjust their size
#         plt.xticks(rotation=90, ha='center', fontsize=font_size)
#         plt.yticks(rotation=0, ha='right', fontsize=font_size)
#
#         # Adjust layout to prevent label cutoff
#         plt.tight_layout()
#
#         # Print feature count
#         print(f"\nTotal number of numeric features analyzed: {n_features}")
#
#         return correlation_matrix, strong_correlations
#
#     except Exception as e:
#         print(f"Error during correlation analysis: {str(e)}")
#         sys.exit(1)
#
# def main():
#     # Replace with your actual file path
#     csv_file = 'data/BPI_with_new_numeric_features_v1_with_user.csv'
#
#     try:
#         correlation_matrix, strong_correlations = analyze_correlations(csv_file)
#
#         # Print correlation matrix to console
#         print("\nCorrelation Matrix:")
#         print(correlation_matrix)
#
#         # Print strong correlations
#         print("\nStrong Feature Correlations (|correlation| > 0.5):")
#         for corr in strong_correlations:
#             print(f"{corr['feature1']} -- {corr['feature2']}: {corr['correlation']:.3f}")
#
#         # Show the heatmap
#         plt.show()
#
#         # Save the plot to a file
#         plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
#         print("\nHeatmap has been saved as 'correlation_heatmap.png'")
#
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         sys.exit(1)
#
# if __name__ == "__main__":
#     main()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from typing import Tuple, List, Dict, Optional
# import sys
#
# def read_csv_with_encoding(file_path: str) -> pd.DataFrame:
#
#     encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
#
#     for encoding in encodings:
#         try:
#             return pd.read_csv(file_path, encoding=encoding)
#         except UnicodeDecodeError:
#             continue
#         except Exception as e:
#             print(f"Error reading file with {encoding} encoding: {str(e)}")
#             continue
#
#     raise ValueError(f"Could not read the file with any of these encodings: {encodings}")
#
# def analyze_correlations(csv_file: str, selected_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[Dict]]:
#     """
#     Analyze correlations between features in a CSV file.
#
#     Parameters:
#     csv_file (str): Path to the CSV file
#     selected_features (List[str], optional): List of feature names to analyze. If None, all numeric features are used.
#
#     Returns:
#     tuple: (correlation_matrix, correlation_summary)
#     """
#     try:
#         # Read the CSV file
#         df = read_csv_with_encoding(csv_file)
#
#         # Print available features
#         print("\nAvailable features in the dataset:")
#         for i, col in enumerate(df.columns):
#             print(f"{i}: {col}")
#
#         # Select only numeric columns
#         numeric_df = df.select_dtypes(include=['int64', 'float64'])
#
#         if numeric_df.empty:
#             raise ValueError("No numeric columns found in the dataset")
#
#         # If specific features are selected, filter the dataframe
#         if selected_features is not None:
#             # Verify all selected features exist
#             missing_features = [f for f in selected_features if f not in numeric_df.columns]
#             if missing_features:
#                 raise ValueError(f"Features not found in dataset: {missing_features}")
#
#             numeric_df = numeric_df[selected_features]
#
#         # Calculate correlation matrix
#         correlation_matrix = numeric_df.corr()
#
#         # Create a summary of strong correlations
#         strong_correlations = []
#         for i in range(len(correlation_matrix.columns)):
#             for j in range(i):
#                 if abs(correlation_matrix.iloc[i, j]) > 0.5:
#                     strong_correlations.append({
#                         'feature1': correlation_matrix.columns[i],
#                         'feature2': correlation_matrix.columns[j],
#                         'correlation': correlation_matrix.iloc[i, j]
#                     })
#
#         # Calculate figure size based on number of features
#         n_features = len(correlation_matrix.columns)
#         fig_size = max(12, n_features * 0.5)
#
#         # Create heatmap
#         plt.figure(figsize=(fig_size, fig_size))
#         font_size = max(8, min(10, 300 / n_features))
#
#         sns.heatmap(correlation_matrix, 
#                     annot=True,
#                     cmap='coolwarm',
#                     center=0,
#                     fmt='.2f',
#                     annot_kws={'size': font_size},
#                     square=True)
#
#         plt.title('Feature Correlation Heatmap', pad=20, fontsize=font_size + 2)
#         plt.xticks(rotation=90, ha='center', fontsize=font_size)
#         plt.yticks(rotation=0, ha='right', fontsize=font_size)
#         plt.tight_layout()
#
#         print(f"\nAnalyzing {n_features} features")
#
#         return correlation_matrix, strong_correlations
#
#     except Exception as e:
#         print(f"Error during correlation analysis: {str(e)}")
#         sys.exit(1)
#
# def main():
#     # Replace with your actual file path
#     csv_file = 'your_data.csv'
#
#     try:
#         # First, load the data to show available features
#         df = read_csv_with_encoding(csv_file)
#         numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#
#         print("\nAvailable numeric features:")
#         for i, col in enumerate(numeric_columns):
#             print(f"{i}: {col}")
#
#         # Get user input for feature selection
#         print("\nEnter the numbers of the features you want to analyze (comma-separated)")
#         print("Example: 0,1,3,5 or press Enter to analyze all features")
#         feature_input = input("Selected features: ").strip()
#
#         if feature_input:
#             # Convert input to list of feature names
#             try:
#                 selected_indices = [int(i.strip()) for i in feature_input.split(',')]
#                 selected_features = [numeric_columns[i] for i in selected_indices]
#             except (ValueError, IndexError) as e:
#                 print("Invalid input. Please enter valid feature numbers.")
#                 sys.exit(1)
#         else:
#             selected_features = None
#
#         # Perform correlation analysis
#         # correlation_matrix, strong_correlations = analyze_correlations(csv_file, selected_features)
#
#         # Print strong correlations
#         print("\nStrong Feature Correlations (|correlation| > 0.5):")
#         for corr in strong_correlations:
#             print(f"{corr['feature1']} -- {corr['feature2']}: {corr['correlation']:.3f}")
#
#         # Show the heatmap
#         plt.show()
#
#         # Save the plot
#         plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
#         print("\nHeatmap has been saved as 'correlation_heatmap.png'")
#
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         sys.exit(1)
#
# if __name__ == "__main__":
#     main()
#
#
#
#
#
# def analyze_specific_features(csv_file: str):
#     """
#     Analyze correlations between specific features in the dataset.
#     """
#     # List of features to analyze
#     selected_features = [
#         'user_day_of_month_workload',
#         'user_workload',
#         'resource_type',
#         'day_of_month',
#         'month',
#         'year',
#         'day_of_week',
#         'month_of_year_workload',
#         'day_of_month_workload',
#         'day_of_week_workload',
#         'resource'
#     ]
#
#     try:
#         # Read the CSV file
#         df = pd.read_csv(csv_file)
#
#         # Verify all requested features exist in the dataset
#         missing_features = [f for f in selected_features if f not in df.columns]
#         if missing_features:
#             print(f"Warning: These features were not found in the dataset: {missing_features}")
#             selected_features = [f for f in selected_features if f in df.columns]
#
#         # Create a copy of the dataframe with selected features
#         analysis_df = df[selected_features].copy()
#
#         # Convert categorical variables to numeric if needed
#         for column in analysis_df.columns:
#             if analysis_df[column].dtype == 'object':
#                 print(f"\nConverting categorical column '{column}' to numeric...")
#                 # Convert categorical variables to numeric codes
#                 analysis_df[column] = pd.Categorical(analysis_df[column]).codes
#
#         # Calculate correlation matrix
#         correlation_matrix = analysis_df.corr()
#
#         # Create a summary of strong correlations
#         strong_correlations = []
#         for i in range(len(correlation_matrix.columns)):
#             for j in range(i):
#                 correlation = correlation_matrix.iloc[i, j]
#                 if abs(correlation) > 0.5:  # Threshold for strong correlation
#                     strong_correlations.append({
#                         'feature1': correlation_matrix.columns[i],
#                         'feature2': correlation_matrix.columns[j],
#                         'correlation': correlation
#                     })
#
#         # Create heatmap
#         plt.figure(figsize=(12, 10))