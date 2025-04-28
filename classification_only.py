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
        

    def calculate_X_and_y(self, df, features, column_name):
        
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
        # y = le_target.fit_transform(df['Activity'])
        y = le_target.fit_transform(df[column_name])

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


def analyze_event_log(df, labels_set, features, filename_roc, column_name):
    
    classifier = LoopRunning()
    roc = Roc()
    df = filter_by_label(df, labels_set, column_name) 
     
    X, y = classifier.calculate_X_and_y(df, features, column_name)
    performance_report = classifier.train(X, y)
    report, avg_auc = roc.roc(classifier.y_test, classifier.X_test, classifier.model, performance_report, y, filename=filename_roc)
    
    try:
        explanations = classifier.explain_predictions_without_shap(labels_set, features)
        print("Explanations:", explanations)
        print("Labels set:", labels_set)
        print("Features:", features)
        
        if explanations is None or 'feature_importance' not in explanations:
            return {'feature_importance': pd.DataFrame(), 'avg_auc': avg_auc}
        
        return {'feature_importance': explanations['feature_importance'], 'avg_auc': avg_auc}
    except Exception as e:
        print(f"Error in explain_predictions: {str(e)}")
        return {'feature_importance': pd.DataFrame(), 'avg_auc': avg_auc}

def filter_by_label(df, labels_set, column_name):
    # return filter_sub_log(df, 'MineTypeLong (1A)', labels_set)
    return filter_sub_log(df, column_name, labels_set)


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
            #'resource_type',
            # 'day_of_month',  
            # 'month',
            # 'year',
            # 'day_of_week',
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
            
            # filename_roc = "classification_21Jan_p1(4)_system_user_forest_with_workload_roc"
            # filename_bar = "classification_21Jan_p1(4)_system_user_forest_with_workload_bar"
            
            filename_roc = "classification_21Jan_p1(4)_system_user_tree_with_workload_roc"
            filename_bar = "classification_21Jan_p1(4)_system_user_tree_with_workload_bar"
            
            # filename_roc = "classification_21Jan_p1(4)_system_user_forest_without_workload_roc"
            # filename_bar = "classification_21Jan_p1(4)_system_user_forest_without_workload_bar"
            
            # filename_roc = "classification_21Jan_p1(4)_system_user_tree_without_workload_roc"
            # filename_bar = "classification_21Jan_p1(4)_system_user_tree_without_workload_bar"

            event_seprated=True
            # df = p1(config, synonyms, csv_url, NULL, shuffle, event_seprated)
            df, split_dfs = p1(config, synonyms, csv_url, NULL, shuffle, event_seprated)
            event_percentage = calculate_event_percentages(split_dfs)
            print(f"Renamed event percentages (event_percentage): {event_percentage}")

        
        elif selected_part == "p2":
            
            
            period = "year_month"  
            
            filename_roc = "classification_21Jan_p2(6-1)_peak_yearmonth_forest_with_workload_roc"
            filename_bar = "classification_21Jan_p2(6-1)_peak_yearmonth_forest_with_workload_bar"
            
            # filename_roc = "classification_21Jan_p2(6-1)_peak_yearmonth_tree_with_workload_roc"
            # filename_bar = "classification_21Jan_p2(6-1)_peak_yearmonth_tree_with_workload_bar"
            
            #**********************************************************************************************************
            # period = "month_day"
            
            # filename_roc = "classification_21Jan_p2(6-2)_peak_daymonth_forest_with_workload_roc"
            # filename_bar = "classification_21Jan_p2(6-2)_peak_daymonth_forest_with_workload_bar"
            
            # filename_roc = "classification_21Jan_p2(6-2)_peak_daymonth_tree_with_workload_roc"
            # filename_bar = "classification_21Jan_p2(6-2)_peak_daymonth_tree_with_workload_bar"
            
            #**********************************************************************************************************
            # period = "week_day"
            
            # filename_roc = "classification_21Jan_p2(6-3)_peak_dayweek_forest_with_workload_roc"
            # filename_bar = "classification_21Jan_p2(6-3)_peak_dayweek_forest_with_workload_bar"
            
            # filename_roc = "classification_21Jan_p2(6-3)_peak_dayweek_tree_with_workload_roc"
            # filename_bar = "classification_21Jan_p2(6-3)_peak_dayweek_tree_with_workload_bar"
            
            
            df = p2(config, synonyms, csv_url, NULL, period)
        
        elif selected_part == "p3":
            
            filename_roc = "classification_22Jan_p3(3)_typo_forest_with_workload_roc"
            filename_bar = "classification_22Jan_p3(3)_typo_forest_with_workload_bar"
            
            # filename_roc = "classification_21Jan_p3(3)_typo_tree_with_workload_roc"
            # filename_bar = "classification_21Jan_p3(3)_typo_tree_with_workload_bar"
            
            df = p3(synonyms[0], config[0], config[1], synonyms, csv_url, NULL, shuffle)
        
        elif selected_part == "p4":
            
            # filename_roc = "classification_22Jan_p4(1)_overall_workload_forest_with_workload_roc"
            # filename_bar = "classification_22Jan_p4(1)_overall_workload_forest_with_workload_bar"
            
            filename_roc = "classification_22Jan_p4(1)_overall_workload_tree_with_workload_roc"
            filename_bar = "classification_22Jan_p4(1)_overall_workload_tree_with_workload_bar"
            
            df = p4(synonyms[0], config[0], config[1], synonyms, csv_url, NULL)
        
        elif selected_part == "p5":
            
            # filename_roc = "classification_21Jan_p5(2)_individual_convention_forest_with_workload_roc"
            # filename_bar = "classification_21Jan_p5(2)_individual_convention_forest_with_workload_bar"
            
            filename_roc = "classification_21Jan_p5(2)_individual_convention_tree_with_workload_roc"
            filename_bar = "classification_21Jan_p5(2)_individual_convention_tree_with_workload_bar"
            
            # filename_roc = "classification_21Jan_p5(2)_individual_convention_forest_without_workload_roc"
            # filename_bar = "classification_21Jan_p5(2)_individual_convention_forest_without_workload_bar"
            
            # filename_roc = "classification_21Jan_p5(2)_individual_convention_tree_without_workload_roc"
            # filename_bar = "classification_21Jan_p5(2)_individual_convention_tree_without_workload_bar"
            
            event_seprated=True
            # df = p5(synonyms[0], config, synonyms, csv_url, NULL, shuffle, event_seprated)
            
            df, split_dfs = p5(config, synonyms, csv_url, NULL, shuffle, event_seprated)
            event_percentage = calculate_event_percentages(split_dfs)
            print(f"Renamed event percentages (event_percentage): {event_percentage}")
        
        elif selected_part == "p6":
            
            filename_roc = "classification_21Jan_p6(5)_workload_monthly_forest_with_workload_roc"
            filename_bar = "classification_21Jan_p6(5)_workload_monthly_forest_with_workload_bar"
            
            # filename_roc = "classification_21Jan_p6(5)_workload_monthly_tree_with_workload_roc"
            # filename_bar = "classification_21Jan_p6(5)_workload_monthly_tree_with_workload_bar"
            
            minimum_day_activity = 5
            df = p6(synonyms[0], config[0], config[1], synonyms, csv_url, NULL, minimum_day_activity, shuffle)
        
        result = analyze_event_log(df, synonyms, features, filename_roc)
        report.extend([result, {"config:": config}]) 
        
       
    xchart = features    
    ychart = []
    ychart.append(result['feature_importance'])
    
    # draw_chart.draw_chart_func2(xchart, ychart, chart_type='bar', title='Feature Importance Chart', xlabel='Features', ylabel='Feature Importance', yscale=(0, 1))
    draw_chart.draw_chart_func2(xchart, ychart, chart_type='bar', title='Feature Importance Chart', xlabel='Features', ylabel='Feature Importance', yscale=(0, 1), filename=filename_bar)
    draw_chart.draw_chart_func2_without_scale(xchart, ychart, chart_type='bar', title='Feature Importance Chart', xlabel='Features', ylabel='Feature Importance', filename=filename_bar+"_without_scale")
    return report


def analyse_log(csv_url, synonyms, column_name, Label_number, features):

    report = []  
    filename_roc = "incidentLog_roc2"+ Label_number
    filename_bar = "incidentLog_bar2"+Label_number
    
    df = pd.read_csv(csv_url, encoding="utf-8") 
    
    features = [
            'user_day_of_month_workload',
            'user_workload',
            # 'resource_type',
            # 'day_of_month',  
            # 'month',
            # 'year',
            # 'day_of_week',
            'month_of_year_workload',
            'day_of_month_workload',
            'day_of_week_workload',
            'resource'
    ]
    
    
    result = analyze_event_log(df, synonyms, features, filename_roc, column_name)
    report.extend([result]) 
        
       
    xchart = features    
    ychart = []
    ychart.append(result['feature_importance'])
    
    
    
    importance_values = result['feature_importance']['importance'].values.tolist()
    
    # Calculate frequencies based on the df and synonyms in column_name
    frequencies = [df[df[column_name] == synonym].shape[0] for synonym in synonyms]
    draw_chart.draw_chart_func2_without_scale(xchart, ychart, chart_type='bar', title='Feature Importance Chart', xlabel='Features', ylabel='Feature Importance', filename=filename_bar+"_without_scale")
    return report, frequencies, importance_values


def main():
    # log_file = "log/classification_21Jan_p1(4)_system_user_forest_with_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p1(4)_system_user_tree_with_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p1(4)_system_user_forest_without_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p1(4)_system_user_tree_without_workload_with_new_log.txt"
    
    # log_file = "log/classification_21Jan_p2(6-1)_peak_yearmonth_forest_with_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p2(6-1)_peak_yearmonth_tree_with_workload_with_new_log.txt"
    
    # log_file = "log/classification_21Jan_p2(6-2)_peak_daymonth_forest_with_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p2(6-2)_peak_daymonth_tree_with_workload_with_new_log.txt"
    
    # log_file = "log/classification_21Jan_p2(6-3)_peak_dayweek_forest_with_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p2(6-3)_peak_dayweek_tree_with_workload_with_new_log.txt"
    
    # log_file = "log/classification_22Jan_p3(3)_typo_forest_with_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p3(3)_typo_tree_with_workload_with_new_log.txt"
    
    # log_file = "log/classification_22Jan_p4(1)_overall_workload_forest_with_workload_with_new_log.txt"
    log_file = "log/classification_22Jan_p4(1)_overall_workload_tree_with_workload_with_new_log.txt"
    
    # log_file = "log/classification_21Jan_p5(2)_individual_convention_forest_with_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p5(2)_individual_convention_tree_with_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p5(2)_individual_convention_forest_without_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p5(2)_individual_convention_tree_without_workload_with_new_log.txt"
    
    # log_file = "log/classification_21Jan_p6(5)_workload_monthly_forest_with_workload_with_new_log.txt"
    # log_file = "log/classification_21Jan_p6(5)_workload_monthly_tree_with_workload_with_new_log.txt"
    
    
    sys.stdout = Logger(log_file)
    # csv_url = 'data/BPI_with_new_numeric_features_v1_with_user.csv'
    # csv_url ='data/BPI_with_new_numeric_features_v3_with_user.csv'
    # csv_url ='data/IncidentEventLogWithDistortedAttributesV1Ranked.v1.csv'
    # csv_url = 'U:/Research/Projects/sef/tportrdtraumaptqld/MAIC 1 Patient Journey/Shokoufeh/HospitalEventLogWithResourceAndFeatures.v1.csv'

    # csv_url = 'U:/Research/Projects/sef/tportrdtraumaptqld/MAIC 1 Patient Journey/Shokoufeh/HospitalEventLogWithResourceAndFeaturesWithoutNullTimestamps.v1.csv'

    # csv_url = 'Z:/logs/Shokoufeh/ActionsEventLogWithDistortedAttributesV1withFeatures.csv'
    # csv_url = 'Z:/logs/Shokoufeh/InspectionEventLogWithDistortedAttributes2025V1ResourceCodedwithFeatures.csv'
    # csv_url = 'Z:/logs/Shokoufeh/InspectionEventLogWithDistortedAttributes2025V2ResourceCodedCaseIdwithFeatures.csv'
    # csv_url = 'Z:/logs/Shokoufeh/test.csv'
    csv_url = 'C:/Users/n11364653/OneDrive - Queensland University of Technology/Desktop/all important files for my confirmation/dataset/version2WithFeatures.csv'



    
    input_data_pattern1 = {
        'pattern': 'Pattern Batch',
        
        # 'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
        # 'synonyms': ["Create Purchase Order Item", "Generate Purchase Request", "Order Item Creation", "Register Order"], #2
        # 'synonyms': ["Create Purchase Requisition Item", "Initiate Purchase Request", "Add Purchase Request Item", "Log New Requisition Item"],#3
        # 'synonyms': ["Cancel Goods Receipt", "Cancel Delivery Record", "Delete Receipt Entry", "Stop Goods Record"], #4
        # 'synonyms': ["SRM: Awaiting Approval", "SRM: Under Review", "SRM: Pending Decision", "SRM: Approval in Process"], #5
        'synonyms': ["Remove Payment Block", "Clear Payment Block", "Eliminate Payment", "Cancel Payment Block"], #6
        
        'configuration': [[0.35, 0.2, 0.2, 0.25]]
        # 'configuration': [[0.8, 0.1, 0.05, 0.05]]
        
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
        # 'synonyms': ["Clear Invoice", "Clear Invoce", "Cleer Invoice", "Clear Invoise"],   #week_day        #it doesn't have system users
        'synonyms': ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],
        'configuration': [ [0.7, 0.1, 0.1, 0.1]    ]
    }
    
    
    input_data_pattern3 =  {
        'pattern': 'Pattern Individual Typo',
        'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
        # 'synonyms': ["Create Purchase Order Item", "Create Purchase Order Itm", "Create Parchase Order Item", "Create Purchas Order Item"], #2
        # 'synonyms': ["Create Purchase Requisition Item", "Create Purchse Requisition Item", "Create Purchase Requisision Item-", "Create Purchase Requisision Item"],   #3
        # 'synonyms': ["Cancel Goods Receipt", "Cancell Goods Receipt", "Cancel Goods Receit", "Cancel Goods Reciept"],   #4
        # 'synonyms': ["SRM: Awaiting Approval", "SRM:waiting Approval", "SRM: Awaiting Approvval", "SRM: Awaiting Aproval_"],    #5
        # 'synonyms': ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],    #6
        # 'synonyms': ["Clear Invoice", "Clear Invoce", "Cleer Invoice", "Clear Invoise"],    #7
        # 'synonyms': ["Remove Payment Block", "Remove Payment Blok", "Remove Paymnt Block", "Remov Paymnet Block"],  #8
        # 'synonyms': ["Receive Order Confirmation", "Receive Order Confirmations", "Recieve Order Confirmation", "Receive Order Confermation"],  #9
        # 'synonyms': ["Change Quantity", "Change Quntity", "Change Qantity", "ChangeQuantity"],  #10
        # 'synonyms': ["Change Price", "Change Pric", "Changed Price", "Change Price _"], #11
        
        
        
  
        # 'synonyms': ["Cancel Invoice Receipt", "Cancel Invoice Receit", "Cancel Invoice Reciept", "Cancel Invoise Receipt"],   
        'configuration': [ [0.2, 0.3]]
    }
    
    
    input_data_pattern4 =  {
        'pattern': 'Pattern Overall Workload',
        
        'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
        # 'synonyms': ["Create Purchase Order Item", "Create Purchase Order Itm", "Create Parchase Order Item", "Create Purchas Order Item"], #2
        # 'synonyms': ["Create Purchase Requisition Item", "Create Purchse Requisition Item", "Create Purchase Requisision Item-", "Create Purchase Requisision Item"],   #3
        # 'synonyms': ["Cancel Goods Receipt", "Cancell Goods Receipt", "Cancel Goods Receit", "Cancel Goods Reciept"],   #4
        # 'synonyms': ["SRM: Awaiting Approval", "SRM:waiting Approval", "SRM: Awaiting Approvval", "SRM: Awaiting Aproval_"],    #5
        # 'synonyms': ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],    #6
        # 'synonyms': ["Clear Invoice", "Clear Invoce", "Cleer Invoice", "Clear Invoise"],    #7
        # 'synonyms': ["Remove Payment Block", "Remove Payment Blok", "Remove Paymnt Block", "Remov Paymnet Block"],  #8
        # 'synonyms': ["Receive Order Confirmation", "Receive Order Confirmations", "Recieve Order Confirmation", "Receive Order Confermation"],  #9
        # 'synonyms': ["Change Quantity", "Change Quntity", "Change Qantity", "ChangeQuantity"],  #10
        # 'synonyms': ["Change Price", "Change Pric", "Changed Price", "Change Price _"], #11
        
        
        
        
        
        
        # 'synonyms': ["Change Quantity", "Change Quntity", "Change Qantity", "ChangeQuantity"],   
        'configuration': [[0.2, 0.3]]
    }
    
    input_data_pattern5 =  {
        'pattern': 'Pattern Individual Convention',
        'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
        # 'synonyms': ["Create Purchase Order Item", "Generate Purchase Request", "Order Item Creation", "Register Order"],#2
        # 'synonyms': ["Create Purchase Requisition Item", "Initiate Purchase Request", "Add Purchase Request Item", "Log New Requisition Item"],#3
        # 'synonyms': ["Cancel Goods Receipt", "Cancel Delivery Record", "Delete Receipt Entry", "Stop Goods Record"],#4
        # 'synonyms': ["SRM: Awaiting Approval", "SRM: Under Review", "SRM: Pending Decision", "SRM: Approval in Process"],#5
        # 'synonyms': ["Remove Payment Block", "Clear Payment Block", "Eliminate Payment", "Cancel Payment Block"], #6
        # 'synonyms': ["Record Invoice Receipt", "Log Invoice Received", "Invoice Documentation", "Register Supplier Invoice"], #7
        # 'synonyms': ["Clear Invoice", "Invoice Finalised", "Invoice Processed", "Close Invoice"], #8
        # 'synonyms': ["Receive Order Confirmation", "Accept Order Details", "Get Order Confirmation", "Retrieve Order Details"], #9
        # 'synonyms': ["Change Quantity", "Modify Amount", "Alter Quantity", "Update Count"], #10
        # 'synonyms': ["Change Price", "Modify Cost", "Revise Rate", "Update Price"], #11
        
        
        
        
        
        # 'synonyms': ["Remove Payment Block", "Clear Payment Block", "Eliminate Payment", "Cancel Payment Block"],
        'configuration': [ [0.35, 0.2, 0.2, 0.25]    ]
    }
    
    input_data_pattern6 =  {
        'pattern': 'Pattern Workload Monthly',
        'synonyms' : [ "Create Purchase Order Item", "Create Purchase Order Itm", "Create Parchase Order Item", "Create Purchas Order Item"], #new_label  
        'configuration': [ [0.2, 0.3]]
    }
    
    
    #**************************************************************************************************************
    
    #MineTypeLong
    # column_name = 'MineTypeLong'
    # synonyms = ["Coal Mine - Surface", "Coal Mine - Surface; Coal Mine - Surface"]  #MineTypeLong
    
    
    #**************************************************************************************************************
    
    #Primary_Equipment Type (5A)
    # column_name = 'Primary_Equipment Type (5A)'
    
    # Label_number = 'Label1'
    # synonyms = ["Unknown" , "unkown"] #Primary_Equipment Type (5A)    (30+6)
    
    # Label_number = 'Label2'
    # synonyms = ["Unknown gas." , "Unknown gas"] #Primary_Equipment Type (5A)    (6+6)
    

    # Label_number = 'Label3'
    # synonyms = ["Toyota landcruiser ute" , "Toyota Landcruiser ute."] #Primary_Equipment Type (5A)    (72+6)
    

    # Label_number = 'Label4'
    # # synonyms = ["Toyota Landcruiser utility" , "Toyota Landcruiser Utlility", "Toyota Landcruiser utility vehicle"] #Primary_Equipment Type (5A)   (54+6+12)    
    # synonyms = ["Toyota landcruiser ute" , "Toyota Landcruiser ute.", "Toyota Landcruiser utility" , "Toyota Landcruiser Utlility", "Toyota Landcruiser utility vehicle"]
    
    # Label_number = 'Label5'
    # synonyms = ["Toyota Landcruiser dual cab" , "Toyota Landcruiser Duel Cab", "Toyota Landcruiser - Dual Cab", "Toyota dual cab landcruiser ute", "Toyota Landcruiser Dual Cab Ute"] #Primary_Equipment Type (5A)    (18+6+6+6+6)
    
    # Label_number = 'Label6'
    # synonyms = ["Toyota Landcruiser" , "Toyota Land Cruiser", "Toyota land cruiser ute", "Toyota Lancruiser", "Toyota Lancruiser Utility", "Toyota Landcruiser Utlility", "Toyota landcruiser ute", "Toyota Landcruiser ute.", "Toyota Landcruiser utility", "Toyota Landcruiser utility vehicle"+"Toyota Landcrusier"] #Primary_Equipment Type (5A)    (192+24+12+6+6+6+72+6+54+12+6)
   
    # Label_number = 'Label7'
    # synonyms = ["N/A no equipment involved" , "Nil Equipment", "nil equipment involved", "Nil equipment involved.", "No equipment", "No Equipment / tools involved", "No equipment involved", "No equipment used in event", "No equipment/tools involved", "no machinery involved", "No primary equipment", "No specific equipment", "No Tool", "No tooling involved", "no tools", "No Tools involved", "No tools or equipment involved"] #Primary_Equipment Type (5A)    (12+18+6+6+54+6+120+6+6+6)
    
    # Label_number = 'Label8'
    # synonyms = ["Toyota Hilux twin-cab utility", "Toyota Hilux Twin Cab Utility", "Toyota Hilux Twin Cab Ute"] #Primary_Equipment Type (5A)    (6+6+6)
    
    # Label_number = 'Label9'
    # synonyms = ["Toyota HiLux Duel Cab Utility" , "Toyota Hilux Dual cab utility", "Toyota Hilux Dual Cab Ute", "Toyota Hi-Lux Dual cab","Toyota Hilux dual cab", "Toyota Hilux daul Cab", "Toyota Hilix dual cab ute"] #Primary_Equipment Type (5A)     (6+18+18+6+54+6+6)
    
    # Label_number = 'Label10'
    # synonyms = ["Toyota HI-Lux" , "Toyota Hilux", "Toyota Hi lux", "Toyota Hylux", "Toyota Hilux ute", "Toyota Hilux Utility"] #Primary_Equipment Type (5A)    (18+138+6+6+30+6)
    
    # Label_number = 'Label11'
    # synonyms = ["N/a", "Na", "N / A", "na"] #Primary_Equipment Type (5A)    (1422+582+6)
    
    # Label_number = 'Label12'
    # synonyms = [ "Nil" , "Nil.", "Nil involved", "nill", "none", "None involved", "NIL", "nil"] #Primary_Equipment Type (5A)     (1602+6+6+6+66+6)
    
    # Label_number = 'Label13'
    # synonyms = [ "Explosive product" , "Explosive products", "Explosiv product", "Explosives Product", ] #Primary_Equipment Type (5A)             (120+30+12+12)
    
    # Label_number = 'Label14'
    # synonyms = ["Explolsives", "Explosive", "Explosives", "Explosves", "Exsplosives"] #Primary_Equipment Type (5A)    (6+108+708+6+18)
    
    # Label_number = 'Label15'
    # synonyms = ["Not Applicable" , "Not applicable."] #Primary_Equipment Type (5A)    (276+6)
 
    #**************************************************************************************************************
    
    #[VFMake (1A)] 
    # column_name = 'VFMake (1A)'
    
    
    
    # Label_number = 'Label1'
    # synonyms = ["TEREX" , "Terrex"] #[VFMake (1A)]     (174+6)
    
    # Label_number = 'Label2'
    # synonyms =["Sanvik", "Sandvik", "Sandvic"]    # (6+96+6)
    
    # Label_number = 'Label3'
    # synonyms =["Liehberr", "Liebherr", "Liebher", "Lieberr", "Leibherr"] # (12+204+6+36)
    
    # Label_number = 'Label4'
    # synonyms =["Komastu", "Komatsu", "Komatsur", "Komtsu","Konmatsu"] #(6+678+6+6)
    
    # Label_number = 'Label5'
    # synonyms =["Catapillar", "Catapiller", "Caterpilar", "Caterpilla", "Caterpillar", "Caterpillari", "Caterpillat", "Caterpiller", "Caterpliiler", "Caterplillar", "Cattapiller", "Catterpillar"]  # (18+12+12+18+1812+6+114+6+6+6+24)
    
    
    
    #**************************************************************************************************************

    
    

    
    # column_name  = 'Role'
    # Label_number = 'Label3Inspectionv3-2'
    # synonyms = ["Chief Inspector of Coal Mines", "Chief Inspector of Mines (Coal)"]
    
    
    # column_name  = 'Role'
    # Label_number = 'Label4Inspectionv3-2'
    # synonyms = ["Inspector of Mines", "Inspector of Mines; Inspector of Mines"] 
    
    
    # column_name  = 'Department'
    # Label_number = 'Label5Inspectionv1'
    # synonyms = ["Mackay District Office", "Mackay District Office; Mackay District Office"]
    
    
  
    # column_name = 'CreatedBy (MRE)'
    # Label_number = 'Label6Inspectionv2'   
    # synonyms = ["Jacques le Roux", "Jacques LeRoux"]
    #

    
    # column_name = 'CompanyName (MRE)'
    # Label_number = 'Label7Inspectionv3-4'   
    # synonyms = ["Boral Construction Material & Cement", "Boral Construction Materials and Cement"]
    #

    
    # column_name = 'CompanyName (MRE)'
    # Label_number = 'Label8Inspectionv2'   
    # synonyms = ["Boral Resources (QLD) Pty Ltd", "Boral Resources QLD PTY LTD"]
    
    
    # column_name = 'CompanyName (MRE)'
    # Label_number = 'Label9Inspectionv2'   
    # synonyms = ["BUMA Australia Ptl Ltd", "BUMA Australia Pty Ltd"]
    
    
    # column_name = 'ActionTakenToComply (CA)'
    # Label_number = 'Label10Inspectionv6-2'   
    # synonyms = ["See SSE response attached to MRE", "See File attached to MRE", "Response received from SSE", "Attached to MRE",
    #             "as per email attached to MRE", "See response attached to MRE", "See responses attached to MRE",
    #             "Response attached to MRE."]
    
    
    # column_name = 'ActionTakenToComply (CA)'
    # Label_number = 'Label10-2Inspectionv6-2'   
    # synonyms = ["See SSE response attached to MRE", "See File attached to MRE", "See response attached to MRE", "See responses attached to MRE",
    #             "Response attached to MRE."]
    
    
    # column_name = 'InspectorIssuingTitle (CA)'
    # Label_number = 'Label11Inspectionv1-2'   
    # synonyms = ["Senior Inspector of Mines (Mining)", "Senior Inspector of Mines"]
    
    
    # column_name = 'InspectorIssuingTitle (CA)'
    # Label_number = 'Label12Inspectionv2'   
    # synonyms = ["Inspector of Mines (Mining)", "Inspector of Mines"]
    
    # column_name = 'ActionTakenToComply (CA)'
    # Label_number = 'Label103Inspection.v3'   
    # synonyms = ["Refer to response in attached email.", "See attached email.", "email attached", "Evidence attached",
    #             "Refer to attached email.", "See attached email", "See attachment"]
    
    
    # column_name = 'ActionTakenToComply (CA)'
    # Label_number = 'Label13-2Inspectionv2'   
    # synonyms = ["Refer to response in attached email.","Refer to attached email."]
    
    # column_name = 'ActionTakenToComply (CA)'
    # Label_number = 'Label1110Inspectionv3'   
    # synonyms = ["See attached email", "See attachment", "See attached email.", "email attached", "Evidence attached"]
    

    column_name = 'ActionTakenToComply (CA)'
    Label_number = 'Label1109Inspectionv4'   
    synonyms =[ "See attachment", "See attached email.", "email attached"]
    
    
    # column_name = 'Description_1 (CA)'
    # Label_number = 'Label14Inspectionv2'   
    # synonyms = ["Isolating locking-out and tagging plant", "Isolating, locking-out and tagging plant"]
    
    
    # column_name = 'Description_1 (CA)'
    # Label_number = 'Label108Inspection.v3'   
    # synonyms =  ["Isolation facilities", "Isolation facility"]
    #

    
    
    # column_name = 'Subject (CA)'
    # Label_number = 'Label16Inspectionv2'   
    # synonyms =  ["Control of Risk", "Control of Risks"]
    
    
    #
    # column_name = 'Subject (CA)'
    # Label_number = 'Label101Inspectionv3'   
    # synonyms =  ["Safety and Health Management System", "Safety & Health Management System"]
    
    # column_name = 'Subject (CA)'
    # Label_number = 'Label104Inspection.v3'   
    # synonyms = ["Training & Assessing", 
    #             "Training & Assessment", "Training and Assessment"]
    
    
    
    # column_name = 'Directive_disp (CA)'
    # Label_number = 'Label19Inspectionv2'   
    # synonyms = ["Directive to review safety and health management system and principal hazard management plan", 
    #             "Directive to review safety and health management system and principal hazard management plans"]
    
    
    # column_name = 'HazardType (CA)'
    # Label_number = 'Label20Inspectionv2'   
    # synonyms = ["Exposure to inorganic minerals (eg Pb - As - Cd)", "Exposure to inorganic minerals (Pb, As, Cd)"]
    #

    
    # column_name = 'PartofMine (CA)'
    # Label_number = 'Label102Inspection.v3'   
    # synonyms = ["Whole of Mine", "Whole Mine"]
    

    
    # column_name = 'LabelPersonGivenTo (CA)'
    # Label_number = 'Label22Inspectionv2'   
    # synonyms = ["Operator and SSE", "SSE and Operator"]
    
    
    #
    # column_name = 'LabelPersonGivenTo (CA)'
    # Label_number = 'Label105Inspection.v3'   
    # synonyms = ["SSE", "The SSE", "SSE Brent McKay", "SSE Ben Taylor", "Peter Edwards (SSE)"]
    
    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label106Inspection.v2'  
    # synonyms = ["Awaiting reply from SSE.",  "Awaiting reply from SSE", "Awaiting reply fom SSE."]
     
    # synonyms = ["Awaiting Reply.", "Awaiting reply"]
    
    
    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label25Inspectionv2'   
    # synonyms = ["Awaiting reply from SSE. Reminder sent.", "Awaiting reply from SSE. Reminder sent."]
    
    
    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label26Inspectionv2'   
    # synonyms = ["Awaiting reply from SSE.",  "Awaiting reply from SSE",
    #             "Awaiting reply fom SSE.",   "Awaiting reply from SSE", "Awaiting reply  from SSE", "Awaiting reply from SSE."]
    

    # synonyms = ["Awaiting reply from SSE.", "Awaiting Reply.", "Awaiting reply from SSE", "Awaiting reply from SSE. Reminder sent.",
    #             "Awaiting reply fom SSE.", "Awaiting reply from SSE", "Awaiting reply  from SSE", "Awaiting reply from SSE.",
    #             "Awaiting reply from SSE. Reminder sent.", "Awaiting reply"]
    
    
    
    
    
    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label1Inspectionv3'   
    # synonyms = ["Follow up" , "To follow up", "Follow up required.", "Follow up.", "Following up", "Folow Up",
    #             "Follow up inspection is required.", "Follow up inspection required to verify compliance.", "follow up required"]
   
    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label5Inspectionv1'   
    # synonyms = ["Request from Management", "Request from management. See attached email", "Request from Management. See attached email."
    #             ,"Request from management. See attached response.", "Management request",  "Management's request", 
    #             "management's request. See attached response."]
    #



    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label27Inspectionv2-2'   
    # synonyms = ["Follow up" , "To follow up", "Follow up required.", "Follow up.", "Following up", "Folow Up",
    #             "follow up required"]
    #
    
    
    # ["Follow up" , "To follow up", "Follow up required.", "Follow up.", "Following up", "Folow Up",
    #             "Follow up inspection is required.", "Follow up inspection required to verify compliance.", "follow up required"]
    #



    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label29Inspectionv6-2' 
    # synonyms = ["Request from management. See attached email", "Request from Management. See attached email."
    #             ,"Request from management. See attached response.",
    #             "management's request. See attached response."]
    
    
    
    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label30Inspectionv2' 
    # synonyms = ["Refer to progress update attached.", "Refer to progress update in attached email.", "Refer to progress update in the attached email."]
    
    
    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label31Inspectionv2' 
    # synonyms = ["Refer to Extension of Time (EoT) Request in Attached Email.", "Refer to EoT request in attached email.", "Refer to EoT Request in attched email."]
    
    
    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label32Inspectionv2'   
    # synonyms = ["Request from management. See attached email", "Request from Management. See attached email."
    #             ,"Request from management. See attached response.",
    #             "management's request. See attached response."]
    
    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label33Inspectionv3'   
    # synonyms = ["Request from management. See attached email"
    #             ,"Request from management. See attached response."]
    
    
    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label36Inspectionv4'   
    # synonyms = ["Refer to progress update in attached email." ,"Refer to progress update in the attached email."]
    


    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label100Inspection.v2'   
    # synonyms = ["Extension requested as per attached document to the MRE", "Extension requested as per email attached",
    #  "Extension requested as per letter attached to the MRE"]
    #

     #    synonyms = ["Extension requested as per attached document to the MRE", "Extension requested as per email attached",
     # "Extension requested as per letter attached to the MRE", "Extension requested"]
    

    # column_name = 'Review (CA_Revisions)'
    # Label_number = 'Label50Inspectionv2'   
    # synonyms = ["Extension requested as per attached document to the MRE", "Extension requested as per email attached",
    #  "Extension requested as per letter attached to the MRE"]

# Label29Inspectionv2
    #'Label29Inspectionv3' 
    
    
    
    
    # ["Managememnt request. See attached response." , "Management request", "Management request. See attached email", "Management request. See attached email.",
    #             "Management request. See attached response","Management request.See attached email.", "Management's request", "Management's request . See attached response",
    #             "Management's request. See attached email.", "Management's request. See attached notes", "management's request. see attached response",
    #             "management's request. See attached response.", "Mangement request", "Mangement request.", "Request from Management", "Request from management. See attached email", "Request from Management. See attached email.", "Request from management. See attached response."]
    #
    #
    





    # column_name = ''
    # Label_number = ''   
    # synonyms = []
    
    report = analyse_log(csv_url, synonyms, column_name, Label_number)
    
    # report = run_pattern(csv_url, input_data_pattern1)
    # report = run_pattern(csv_url, input_data_pattern2)
    #report = run_pattern(csv_url, input_data_pattern2YearMonth)
    # report = run_pattern(csv_url, input_data_pattern2DayMonth)
    # report = run_pattern(csv_url, input_data_pattern2DayWeek)
    # report = run_pattern(csv_url, input_data_pattern3)
    # report = run_pattern(csv_url, input_data_pattern4)
    # report = run_pattern(csv_url, input_data_pattern5)
    # report = run_pattern(csv_url, input_data_pattern6)
    
    print(report)
    
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal



if __name__ == "__main__":
    main()