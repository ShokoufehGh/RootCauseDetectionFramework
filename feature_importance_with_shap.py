import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
import shap
from collections import Counter
from sklearn.preprocessing import label_binarize
from patterns.pattern1_sytem_user import main as p1
from patterns.pattern2_peak_time import main as p2
from patterns.pattern3_individual_typo import main as p3
from patterns.pattern4_overall_workload import main as p4,calculate_user_workload
from patterns.pattern5_individual_convention import main as p5
from patterns.pattern6_workload_monthly import main as p6
from patterns.pattern7_department import main as p7
import sys
import os
import draw_chart
from common_functions import filter_sub_log, printOneShapValue
from pattern__injection_runner import Logger
from datetime import datetime
from _overlapped import NULL
import draw_chart
from sklearn.tree import DecisionTreeClassifier
import lime
import lime.lime_tabular
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from explanation import Explanation  
from roc import Roc  

class LoopRunning:
    def __init__(self):
        self.final_result = []
        self.label_encoders = {}
        self.explanation = Explanation()
        # self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        

    def calculate_X_and_y(self, df, features):
        
        X = pd.DataFrame(index=df.index)
        self.label_encoders = {}
        
        # Encode categorical features
        for feature in features:
            if df[feature].dtype == 'object':
                le = LabelEncoder()
                X[feature] = le.fit_transform(df[feature].fillna('unknown'))
                self.label_encoders[feature] = le
            else:
                X[feature] = df[feature]
        
        # Encode target variable
        le_target = LabelEncoder()
        y = le_target.fit_transform(df['event concept:name'])
        self.label_encoders['target'] = le_target
        print(self.get_mappings())
        
        return X, y



    def get_mappings(self):
        """Returns a dictionary of mappings for all features and target."""
        mappings = {}
        for feature, le in self.label_encoders.items():
            if le is not None:  # For categorical features and target
                mappings[feature] = dict(zip(le.classes_, le.transform(le.classes_)))
            else:  # For numerical features
                mappings[feature] = "Numerical feature (no encoding)"
        return mappings
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        
        performance_report = classification_report(y_test, y_pred, zero_division=0)
        return performance_report
    
    def explain_predictions_without_shap(self, class_names, features, interaction=False):
        feature_importance = self.explanation.get_feature_importance(self.model, self.X_train)

        return {'shap_values': NULL, 'feature_importance': feature_importance}

    def explain_predictions(self, class_names, features, interaction=False):
        feature_importance = self.explanation.get_feature_importance(self.model, self.X_train)
        shap_values = self.explanation.get_shap_values(self.model, self.X_test)
        
        summed_shap_values = self.explanation.calculate_summed_shap_values(shap_values)
        self.explanation.print_shap_values(class_names, features, summed_shap_values, "summed_shap_values")
        
        summed_abs_shap_values = self.explanation.calculate_summed_abs_shap_values(shap_values)
        self.explanation.print_shap_values(class_names, features, summed_abs_shap_values, "summed_abs_shap_values")
        
        self.explain_for_resource_type(class_names, features, resource_type=1, label="system")
        self.explain_for_resource_type(class_names, features, resource_type=0, label="human")
        
        if interaction:
            self.explanation.print_feature_interactions(shap_values, class_names, self.X_train.columns)

        return {'shap_values': shap_values, 'feature_importance': feature_importance}

    def explain_for_resource_type(self, class_names, features, resource_type, label):
        mask = self.X_test['resource_type'] == resource_type
        X_test_filtered = self.X_test[mask]

        if not X_test_filtered.empty:
            shap_values_filtered = self.explanation.get_shap_values(self.model, X_test_filtered)
            summed_shap_values = self.explanation.calculate_summed_shap_values(shap_values_filtered)
            self.explanation.print_shap_values(class_names, features, summed_shap_values, f"summed_shap_values {label}")

            summed_abs_shap_values = self.explanation.calculate_summed_abs_shap_values(shap_values_filtered)
            self.explanation.print_shap_values(class_names, features, summed_abs_shap_values, f"summed_abs_shap_values {label}")


def analyze_event_log(df, labels_set, features, filename_roc):
    
    classifier = LoopRunning()
    roc = Roc()
    df = filter_by_label(df, labels_set) 
     
    X, y = classifier.calculate_X_and_y(df, features)
    performance_report = classifier.train(X, y)
    report, avg_auc = roc.roc(classifier.y_test, classifier.X_test, classifier.model, performance_report, y, filename=filename_roc)
    
    try:
        # explanations = classifier.explain_predictions_without_shap(labels_set, features)
        explanations = classifier.explain_predictions(labels_set, features)

        
        print("Explanations:", explanations)
        print("Labels set:", labels_set)
        print("Features:", features)
        
        if explanations is None or 'feature_importance' not in explanations:
            return {'feature_importance': pd.DataFrame(), 'avg_auc': avg_auc}
        
        return {'feature_importance': explanations['feature_importance'], 'avg_auc': avg_auc}
    except Exception as e:
        print(f"Error in explain_predictions: {str(e)}")
        return {'feature_importance': pd.DataFrame(), 'avg_auc': avg_auc}

def filter_by_label(df, labels_set):
    return filter_sub_log(df, 'event concept:name', labels_set)


def calculate_event_percentages(split_dfs):

    group_counts = [len(df) for df in split_dfs]
    total_events = sum(group_counts)
    
    last_three_events = sum(group_counts[1:]) 
    percentage = (last_three_events / total_events) * 100
    
    print(f"group_counts: {group_counts}")
    print(f"total_events: {total_events}")
    print(f"renamed events: {last_three_events}")
    print(f"renamed percentage: {percentage}")

    
    return percentage


def run_pattern(csv_url, input_data):
    synonyms = input_data['synonyms']
    configurations = input_data['configuration']
    report = []  
     

    features = [
            'user_day_of_month_workload',
            'user_workload',
            'day_of_month',  
            'month',
            'day_of_week',
            'month_of_year_workload',
            'day_of_month_workload',
            'day_of_week_workload',
            'resource'
    ]
    
    for i, config in enumerate(configurations):
        print(f"Running configuration {i+1}: {config}")
        # selected_part = "p1"  
        # selected_part = "p2" 
        # selected_part = "p3" 
        selected_part = "p4" 
        # selected_part = "p5" 
        # selected_part = "p6" 
        
        shuffle = False
        period = None
        minimum_day_activity = 5
        
        if selected_part == "p1":

            filename_roc = "classification_21Jan_p1(4)_system_user_tree_with_workload_roc"
            filename_bar = "classification_21Jan_p1(4)_system_user_tree_with_workload_bar"


            event_seprated=True
            df, split_dfs = p1(config, synonyms, csv_url, NULL, shuffle, event_seprated)
            event_percentage = calculate_event_percentages(split_dfs)
            print(f"Renamed event percentages (event_percentage): {event_percentage}")

        
        elif selected_part == "p2":
            
            
            period = "year_month"  
            
            filename_roc = "classification_21Jan_p2(6-1)_peak_yearmonth_forest_with_workload_roc"
            filename_bar = "classification_21Jan_p2(6-1)_peak_yearmonth_forest_with_workload_bar"
        
            
            df = p2(config, synonyms, csv_url, NULL, period)
        
        elif selected_part == "p3":
            
            filename_roc = "classification_22Jan_p3(3)_typo_forest_with_workload_roc"
            filename_bar = "classification_22Jan_p3(3)_typo_forest_with_workload_bar"

            
            df = p3(synonyms[0], config[0], config[1], synonyms, csv_url, NULL, shuffle)
        
        elif selected_part == "p4":

            
            filename_roc = "classification_22Jan_p4(1)_overall_workload_tree_with_workload_roc"
            filename_bar = "classification_22Jan_p4(1)_overall_workload_tree_with_workload_bar"
            
            df = p4(synonyms[0], config[0], config[1], synonyms, csv_url, NULL)
        
        elif selected_part == "p5":
            

            
            filename_roc = "classification_21Jan_p5(2)_individual_convention_tree_with_workload_roc"
            filename_bar = "classification_21Jan_p5(2)_individual_convention_tree_with_workload_bar"

            
            event_seprated=True
            
            df, split_dfs = p5(config, synonyms, csv_url, NULL, shuffle, event_seprated)
            event_percentage = calculate_event_percentages(split_dfs)
            print(f"Renamed event percentages (event_percentage): {event_percentage}")
        
        elif selected_part == "p6":
            
            filename_roc = "classification_21Jan_p6(5)_workload_monthly_forest_with_workload_roc"
            filename_bar = "classification_21Jan_p6(5)_workload_monthly_forest_with_workload_bar"

            
            minimum_day_activity = 5
            df = p6(synonyms[0], config[0], config[1], synonyms, csv_url, NULL, minimum_day_activity, shuffle)
        
        result = analyze_event_log(df, synonyms, features, filename_roc)
        report.extend([result, {"config:": config}]) 
        
       
    xchart = features    
    ychart = []
    ychart.append(result['feature_importance'])
    
    draw_chart.draw_chart_func2(xchart, ychart, chart_type='bar', title='Feature Importance Chart', xlabel='Features', ylabel='Feature Importance', yscale=(0, 1), filename=filename_bar)
    draw_chart.draw_chart_func2_without_scale(xchart, ychart, chart_type='bar', title='Feature Importance Chart', xlabel='Features', ylabel='Feature Importance', filename=filename_bar+"_without_scale")
    return report


def main():

    log_file = "log/classification_22April_p4(1)_overall_workload_tree_with_workload_with_new_log.txt"

    
    sys.stdout = Logger(log_file)
    csv_url ='data/BPI_with_new_numeric_features_v3_with_user.csv'
    
    
    
    input_data_pattern1 = {
        'pattern': 'Pattern Batch',
        'synonyms': ["Remove Payment Block", "Clear Payment Block", "Eliminate Payment", "Cancel Payment Block"], #6      
        'configuration': [[0.35, 0.2, 0.2, 0.25]]
        
    }
    
    input_data_pattern2YearMonth =  {
        'pattern': 'Pattern PeakTime: YearMonth',
        'synonyms': ["Create Purchase Requisition Item", "Create Purchse Requisition Item", "Create Purchase Requisision Item-", "Create Purchase Requisision Item"],   #year_month
        'configuration': [ [0.7, 0.1, 0.1, 0.1]    ]
    }
    
    
    input_data_pattern2DayMonth =  {
        'pattern': 'Pattern PeakTime: DayMonth',
        'synonyms': ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],   #month_day
        'configuration': [ [0.7, 0.1, 0.1, 0.1]    ]
    }
    
    input_data_pattern2DayWeek =  {
        'pattern': 'Pattern PeakTime: DayWeek',
        'synonyms': ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],
        'configuration': [ [0.7, 0.1, 0.1, 0.1]    ]
    }
    
    
    input_data_pattern3 =  {
        'pattern': 'Pattern Individual Typo',
        'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
        'configuration': [ [0.2, 0.3]]
    }
    
    
    input_data_pattern4 =  {
        'pattern': 'Pattern Overall Workload',     
        'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
        'configuration': [[0.2, 0.3]]
    }
    
    input_data_pattern5 =  {
        'pattern': 'Pattern Individual Convention',
        'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1      
        'configuration': [ [0.35, 0.2, 0.2, 0.25]    ]
    }
    
    input_data_pattern6 =  {
        'pattern': 'Pattern Workload Monthly',
        'synonyms' : [ "Create Purchase Order Item", "Create Purchase Order Itm", "Create Parchase Order Item", "Create Purchas Order Item"], #new_label  
        'configuration': [ [0.2, 0.3]]
    }
    
    
    
    
    # report = run_pattern(csv_url, input_data_pattern1)
    # report = run_pattern(csv_url, input_data_pattern2)
    # report = run_pattern(csv_url, input_data_pattern2YearMonth)
    # report = run_pattern(csv_url, input_data_pattern2DayMonth)
    # report = run_pattern(csv_url, input_data_pattern2DayWeek)
    # report = run_pattern(csv_url, input_data_pattern3)
    report = run_pattern(csv_url, input_data_pattern4)
    # report = run_pattern(csv_url, input_data_pattern5)
    # report = run_pattern(csv_url, input_data_pattern6)
    
    print(report)
    
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal



if __name__ == "__main__":
    main()