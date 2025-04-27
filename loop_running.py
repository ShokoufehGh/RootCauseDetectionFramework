import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
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
import scipy.stats
import random
from scipy.stats import percentileofscore

import warnings
# from adodbapi.examples.xls_read import filename

warnings.filterwarnings('ignore', category=FutureWarning, 
                       message="'DataFrame.swapaxes' is deprecated")

from common_functions import (
    filter_sub_log,
    calculate_peak_day_of_month,
    calculate_peak_day_of_week,
    calculate_peak_year_month
)
from pattern__injection_runner import Logger
from datetime import datetime
from _overlapped import NULL
from sklearn.model_selection._search import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier


class LoopRunning:
    def __init__(self):
        self.final_result = []
        self.label_encoders = {}
        
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            # bootstrap = True,
            # max_features = 'auto',
            # class_weight='balanced'
        )     
        
        
        # self.model = DecisionTreeClassifier(max_depth=10, random_state=42)

        
        
        
    def tune_hyperparameters(self, X_train, y_train):
        # Define parameter grid
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
            'max_features': ['auto', 'sqrt', 'log2', 0.5, None]
        }
    
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
    
        # Fit the model
        random_search.fit(X_train, y_train)
    
        # Output the best parameters
        print("Best Hyperparameters:", random_search.best_params_)
        return random_search.best_estimator_   
        



    def calculate_X_and_y(self, df):
        # Define the features to be used
        # features = [
        #     'day_of_month_workload',
        #     'overal_workload',
        #     'resource_type',
        #     'day_of_month',  
        #     'month',
        #     'year',
        #     'day_of_week',
        #     'is_peak_year_month',
        #     'is_peak_day_of_month',
        #     'is_peak_day_of_week'
        #     # 'case Spend area text',
        #     # 'case Sub spend area text',
        #     # 'case Item Type',
        #     # 'case Spend area text',
        #     # 'case Item',
        # ]
        
        # features = [
        #     'user_day_of_month_workload',
        #     'user_workload',
        #     'resource_type',
        #     'day_of_month',  
        #     'month',
        #     'year',
        #     'day_of_week',
        #     'month_of_year_workload',
        #     'day_of_month_workload',
        #     'day_of_week_workload'
        # ]
        
        features = [
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
        
        # features = [
        #     'user_day_of_month_workload',
        #     'user_workload',
        #     'resource_type',
        #     'day_of_month',  
        #     'month',
        #     'year',
        #     'day_of_week',
        #     'month_of_year_workload',
        #     'day_of_month_workload',
        #     'day_of_week_workload',
        #     'event User'
        # ]
        
        
        # Encode categorical features and prepare feature matrix
        X = pd.DataFrame(index=df.index)
        self.label_encoders = {}
        self.scalers = {}  #store scalers
        
        for feature in features:
            if df[feature].dtype == 'object':
                le = LabelEncoder()
                X[feature] = le.fit_transform(df[feature].fillna('unknown'))
                self.label_encoders[feature] = le
            # elif feature in ['overal_workload']:
            #     values = df[feature].values
            #     X[feature] = [percentileofscore(values, x) for x in values]
            #

                
            # scaler = StandardScaler()
                # X[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1)).flatten()
                # self.scalers[feature] = scaler 
            else:
                X[feature] = df[feature]
        print(f"X: {X}")
        # exit()
        
        le_target = LabelEncoder()
        y = le_target.fit_transform(df['event concept:name'])
        self.label_encoders['target'] = le_target
        
        # Print the mapping of target labels to numbers
        mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
        print(f"Mapping concept:name to number: {mapping}")
        
        return X, y
    
    
    def train(self, X, y):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # Debug: Check shapes after splitting
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of y_train: {y_train.shape}")
    
        # hyperparameter         
        # X_train, y_train = self.HyperparameterTunning(X_train, y_train)
        # self.model = self.tune_hyperparameters(X_train, y_train)     #best_model
        
        self.model.fit(X_train, y_train)
        print("train finished")
        y_pred = self.model.predict(X_test)
        
        print("predict y")
        counts = Counter(y_pred)
        for value, count in counts.items():
            print(f"Value {value}: {count}")    
            
        print("test y")
        counts = Counter(y_test)
        for value, count in counts.items():
            print(f"Value {value}: {count}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        
        performance_report = classification_report(y_test, y_pred, zero_division=0)
        print("\nClassification Performance:")
        print(performance_report)
        return performance_report
    
    def HyperparameterTunning(self, X_train, y_train):
    # Define the hyperparameter space for RandomizedSearchCV
        param_dist = {
            'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
            'max_depth': [None, 10, 20, 30, 40],  # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],      # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 2, 4],        # Minimum number of samples required at each leaf node
            'bootstrap': [True, False],           # Whether to use bootstrap samples
            'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
        }
        
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_dist,
            n_iter=20,  # Number of parameter settings to sample
            cv=5,       # Number of cross-validation folds
            verbose=2,  # Print progress
            random_state=42,
            n_jobs=-1   # Use all available cores
        )
        
        # Perform hyperparameter tuning
        random_search.fit(X_train, y_train)
        print("Best Hyperparameters:", random_search.best_params_)
        exit()
        return (X_train, y_train)
    
    def roc(self, report, y, chart=False):                
        X_train = self.X_train 
        X_test = self.X_test 
        y_test = self.y_test 
        
        # n_classes = len(np.unique(y))
        classes = np.sort(np.unique(y))
        n_classes = len(classes)
        
        # y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))
        y_test_binarized = label_binarize(y_test, classes=classes)
        
        if n_classes == 2:
            y_test_binarized = np.hstack((1 - y_test_binarized, y_test_binarized))
        
        if hasattr(self.model, "predict_proba"):
            y_pred_proba = self.model.predict_proba(X_test)
        else:
            y_pred_proba = self.model.decision_function(X_test)
            # y_pred_proba = np.exp(y_pred_proba) / np.sum(np.exp(y_pred_proba), axis=1, keepdims=True)
            if y_pred_proba.ndim == 1:  # Binary classification
                y_pred_proba = np.vstack([(1 - y_pred_proba), y_pred_proba]).T
            else:  # Multi-class
                y_pred_proba = np.exp(y_pred_proba)
                y_pred_proba /= np.sum(y_pred_proba, axis=1, keepdims=True)

        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes_actual = y_pred_proba.shape[1]
        
        # for i in range(n_classes):
        for i in range(n_classes_actual):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        if chart:
            # Plot ROC curve for each class
            plt.figure()
            colors = ['blue', 'green', 'red', 'purple']  # Adjust colors based on the number of classes
            for i, color in zip(range(n_classes_actual), colors[:n_classes_actual]):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
            
        
            plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve (Multi-Class)')
            plt.legend(loc="lower right")
            plt.show()
        
        # Calculate the average AUC score
        avg_auc = np.mean(list(roc_auc.values()))
        print(f"Average AUC Score: {avg_auc:.2f}")
        
        return report, avg_auc
         
    def explain_predictions(self, class_names, interaction=True): 
    
        
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # pd.set_option('display.max_rows', None)    
        # print("\nFeature Importance:")
        # print(feature_importance)   
        # return feature_importance
        
        
        # human_mask = self.X_test['resource_type'] == 0  # Replace 'human' with the actual value
        # X_test_human = self.X_test[human_mask]
        #
        # # Check if any rows match the condition
        # print(self.X_test)
        # if X_test_human.empty:
        #     print("No rows found where resource_type is 'human'.")
        #     # exit()
        #     return
        
        # Calculate SHAP values for the filtered X_test
        # explainer = shap.TreeExplainer(self.model)
        # shap_values = explainer.shap_values(X_test_human)
        # shap_values = explainer.shap_values(self.X_test)
           
        # I want to calculate shap value only for X_test that the feature of resource_type is equal to human
        # np.set_printoptions(threshold=np.inf)
        # print("1 shap_values")
        # print(shap_values)
        # print("2 shap_values")
        
        # if(interaction):
        #     # print(shap_values)
        #
        #     feature_names = list(self.X_train.columns)
        #
        #     result = []
        #     print('sssssssssssssssssss')
        #     for i, feature1 in enumerate(feature_names):
        #         print('ddddddddddddddddddddddd')
        #         for j, feature2 in enumerate(feature_names):
        #             interaction_row = {
        #                 "Feature Interaction": f"{feature1}/{feature2}"
        #             }
        #             for class_index, class_name in enumerate(class_names):
        #                 interaction_row[class_name] = round(shap_values[i, j, class_index],4)*100
        #             result.append(interaction_row)
        #
        #     pd.set_option('display.max_rows', None)
        #     interaction_df = pd.DataFrame(result)
        #     print("Feature Interactions:")
        #     print(interaction_df)
        #
        # pd.reset_option('display.max_rows')
        # pd.reset_option('display.colheader_justify')
        # exit()
        return {
            # 'shap_values': shap_values,
            'feature_importance': feature_importance,
        }
# def analyze_event_log(df, labels_set, classifier, X, y):
def analyze_event_log(df, labels_set):

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    classifier = LoopRunning() 
    # df = classifier.add_new_features(df, labels_set)
    df = filter_by_label(df, labels_set)
    X, y = classifier.calculate_X_and_y(df)       
  
    performance_report = classifier.train(X, y)
    report, avg_auc = classifier.roc(performance_report, y)
    explanations = classifier.explain_predictions(labels_set)
    
    return {
            'feature_importance': explanations['feature_importance'],
            'avg_auc': avg_auc,
        }

def filter_by_label(df, labels_set):        
        
    # Filter the DataFrame based on the labels_set
    df = filter_sub_log(df, 'event concept:name', labels_set)
    
    return df  


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


def pattern1(csv_url, cmodel):    
    
    # filename_roc = "loop_27jan_p1(4)_system_user_forest_with_workload_multiple_roc"
    # filename_bar = "loop_27jan_p1(4)_system_user_forest_with_workload_multiple_line"
            
    filename_roc = "loop_27jan_p1(4)_system_user_tree_with_workload_multiple_roc6"
    filename_bar = "loop_27jan_p1(4)_system_user_tree_with_workload_multiple_line6"
            
    # filename_roc = "loop_27jan_p1(4)_system_user_forest_without_workload_multiple_roc"
    # filename_bar = "loop_27jan_p1(4)_system_user_forest_without_workload_multiple_line"
            
    # filename_roc = "loop_27jan_p1(4)_system_user_tree_without_workload_multiple_roc"
    # filename_bar = "loop_27jan_p1(4)_system_user_tree_without_workload_multiple_line"
       
    input_data =  {
            'pattern': 'Pattern Batch',
            # 'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
            # 'synonyms': ["Create Purchase Order Item", "Generate Purchase Request", "Order Item Creation", "Register Order"],#2
            # 'synonyms': ["Create Purchase Requisition Item", "Initiate Purchase Request", "Add Purchase Request Item", "Log New Requisition Item"],#3
            # 'synonyms': ["Cancel Goods Receipt", "Cancel Delivery Record", "Delete Receipt Entry", "Stop Goods Record"],#4
            # 'synonyms': ["SRM: Awaiting Approval", "SRM: Under Review", "SRM: Pending Decision", "SRM: Approval in Process"],#5
            'synonyms': ["Remove Payment Block", "Clear Payment Block", "Eliminate Payment", "Cancel Payment Block"], #6
            'configuration': [
                [0.8, 0.1, 0.05, 0.05],
                [0.7, 0.1, 0.1, 0.1],
                [0.6, 0.15, 0.15, 0.1],
                [0.5, 0.15, 0.15, 0.2],
                [0.4, 0.2, 0.2, 0.2], 
                [0.3, 0.2, 0.25, 0.25],
                [0.2, 0.3, 0.25, 0.25]  
                # [0.95, 0.01, 0.02, 0.02],
                # [0.90, 0.03, 0.03, 0.04],
                # [0.85, 0.04, 0.05, 0.06],
                # [0.80, 0.10, 0.05, 0.05],
                # [0.75, 0.11, 0.07, 0.07],
                # [0.70, 0.10, 0.10, 0.10],
                # [0.65, 0.11, 0.12, 0.12],
                # [0.60, 0.15, 0.15, 0.10],
                # [0.55, 0.16, 0.17, 0.12],
                # [0.50, 0.15, 0.15, 0.20],
                # [0.45, 0.16, 0.17, 0.22],
                # [0.40, 0.20, 0.20, 0.20]
            ] 
            # 'configuration': [
            #     [0.95, 0.01, 0.02, 0.02],
            #     [0.90, 0.03, 0.03, 0.04],
            #     [0.85, 0.04, 0.05, 0.06],
            #     [0.80, 0.10, 0.05, 0.05],
            #     [0.75, 0.11, 0.07, 0.07],
            #     [0.70, 0.10, 0.10, 0.10],
            #     [0.65, 0.11, 0.12, 0.12],
            #     [0.60, 0.15, 0.15, 0.10],
            #     [0.55, 0.16, 0.17, 0.12],
            #     [0.50, 0.15, 0.15, 0.20],
            #     [0.45, 0.16, 0.17, 0.22],
            #     [0.40, 0.20, 0.20, 0.20],
            #     [0.35, 0.21, 0.22, 0.22],
            #     [0.30, 0.20, 0.25, 0.25],
            #     [0.25, 0.21, 0.27, 0.27],
            #     [0.20, 0.30, 0.25, 0.25],
            #     [0.15, 0.31, 0.27, 0.27],
            #     [0.10, 0.32, 0.29, 0.29],
            #     [0.05, 0.33, 0.31, 0.31]
            # ]
        }
    
   
    synonyms = input_data['synonyms']
    
    # try:
    #     df = pd.read_csv(csv_url, encoding="ISO-8859-1")  
    # except UnicodeDecodeError:
    #     print("Encoding error! Trying another encoding...")
    #     df = pd.read_csv(csv_url, encoding="cp1252") 
    # df = filter_by_label(df, synonyms)
    # classifier = LoopRunning() 
    
    # X, y = classifier.calculate_X_and_y(df)
    # print(f"y: {y}")   
    # print(f"X: {X}")
    configurations = input_data['configuration']    
    print(f"Synonym set: {synonyms}")
    
    report = []
    # xchart = [20, 30, 40, 50, 60, 70, 80]
    xchart = []
    ychart = []
    
    feature_importance_dict = {  
        'user_day_of_month_workload': [],
        'user_workload': [],
        # 'resource_type': [],
        'day_of_month': [],
        'month': [],
        # 'year': [],
        'day_of_week': [],
        'month_of_year_workload': [],
        'day_of_month_workload': [],
        'day_of_week_workload': [],
        'resource': []
    }
    
    
    for i, config in enumerate(configurations):        
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ===============================================================")
        print(f"Running configuration {i+1}: {config}")
        shuffle = False
        event_seprated=True
        # df = p1(config, synonyms, csv_url, NULL, shuffle, event_seprated)
        df, split_dfs = p1(config, synonyms, csv_url, NULL, shuffle, event_seprated)

        event_percentage = calculate_event_percentages(split_dfs)
        xchart.append(event_percentage)
        # result = analyze_event_log(df, synonyms,classifier, X, y)
        result = analyze_event_log(df, synonyms)

        report.extend([result, {"config": config}])
        ychart.append(result['avg_auc'])
        # feature_importance_dict.append(result['feature_importance'])
        # print (f"ychart: {ychart}")
        
        
        # for feature, importance in result['feature_importance'].items():
        #     feature_importance_dict['feature'].append('importance')
        
        for feature in feature_importance_dict.keys():
            feature_row = result['feature_importance'][result['feature_importance']['feature'] == feature]
            if not feature_row.empty:
                importance_value = feature_row['importance'].iloc[0]
                feature_importance_dict[feature].append(importance_value)
            else:
                feature_importance_dict[feature].append(0)
        
    
    ychart = [float(value) for value in ychart]    
    feature_importance_dict = {
        feature: [float(value) for value in values]
        for feature, values in feature_importance_dict.items()
    }
    
    print(f"Actual event percentages (xchart): {xchart}")
    print (f"ychart after change: {ychart}")
    print (f"feature_importance_dict after change: {feature_importance_dict}")
    draw_chart.draw_chart_func(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', filename = filename_roc)
    draw_chart.draw_multiple_lines_chart(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', filename = filename_bar)
    # report = draw_chart.draw_charts_and_report(xchart, ychart, feature_importance_dict, report)
    draw_chart.draw_chart_func_scale(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', yscale=(0, 1), filename = filename_roc+"_with_scale")
    draw_chart.draw_multiple_lines_chart_scale(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance',yscale=(0, 1), filename = filename_bar+"_with_scale")
    # report = draw_chart.draw_charts_and_report_scale(xchart, ychart, feature_importance_dict, report, xlabel='Event Percentage', yscale=(0, 1))
    # report = draw_chart.draw_charts_and_report_scale(xchart, ychart, feature_importance_dict, report, xlabel='Event Percentage')

    return report

def pattern2(csv_url, cmodel):        
    input_data =  {
            'pattern': 'Pattern PeakTime',
            # 'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
            # 'synonyms': ["Create Purchase Order Item", "Generate Purchase Request", "Order Item Creation", "Register Order"],#2
            # 'synonyms': ["Create Purchase Requisition Item", "Initiate Purchase Request", "Add Purchase Request Item", "Log New Requisition Item"],#3
            # 'synonyms': ["Cancel Goods Receipt", "Cancel Delivery Record", "Delete Receipt Entry", "Stop Goods Record"],#4
            # 'synonyms': ["SRM: Awaiting Approval", "SRM: Under Review", "SRM: Pending Decision", "SRM: Approval in Process"],#5
            # 'synonyms': ["Remove Payment Block", "Clear Payment Block", "Eliminate Payment", "Cancel Payment Block"], #6
            # 'synonyms': ["Record Invoice Receipt", "Log Invoice Received", "Invoice Documentation", "Register Supplier Invoice"], #7
            # 'synonyms': ["Clear Invoice", "Invoice Finalised", "Invoice Processed", "Close Invoice"], #8
            # 'synonyms': ["Receive Order Confirmation", "Accept Order Details", "Get Order Confirmation", "Retrieve Order Details"], #9
            # 'synonyms': ["Change Quantity", "Modify Amount", "Alter Quantity", "Update Count"], #10
            'synonyms': ["Change Price", "Modify Cost", "Revise Rate", "Update Price"], #11
            
            
            
            
            
            
            
            
            # 'synonyms': ["Clear Invoice", "Clear Invoce", "Cleer Invoice", "Clear Invoise"],   #week_day
            # 'synonyms': ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],   #wee_day new label because it has system users
            # 'synonyms': ["Create Purchase Requisition Item", "Create Purchse Requisition Item", "Create Purchase Requisision Item-", "Create Purchase Requisision Item"],   #year_month
            # 'synonyms': ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],   #month_day
            'configuration': [ 
                # [0.9, 0.3, 0.03, 0.04],
                # [0.8, 0.1, 0.05, 0.05],
                # [0.7, 0.1, 0.1, 0.1],
                # [0.6, 0.15, 0.15, 0.1],
                # [0.5, 0.15, 0.15, 0.2],
                # [0.4, 0.2, 0.2, 0.2], 
                # [0.3, 0.2, 0.25, 0.25],
                # [0.2, 0.3, 0.25, 0.25] 
                [0.97, 0.01, 0.01, 0.01],
                [0.95, 0.01, 0.02, 0.02],
                [0.90, 0.03, 0.03, 0.04],
                [0.85, 0.04, 0.05, 0.06],
                [0.80, 0.10, 0.05, 0.05],
                [0.75, 0.11, 0.07, 0.07],
                [0.70, 0.10, 0.10, 0.10],
                [0.65, 0.11, 0.12, 0.12],
                [0.60, 0.15, 0.15, 0.10],
                [0.55, 0.16, 0.17, 0.12],
                [0.50, 0.15, 0.15, 0.20],
                # [0.45, 0.16, 0.17, 0.22],
                # [0.40, 0.20, 0.20, 0.20]
                # [0.35, 0.21, 0.22, 0.22],
                # [0.30, 0.20, 0.25, 0.25],
                # [0.25, 0.21, 0.27, 0.27],
                # [0.20, 0.30, 0.25, 0.25],
                # [0.15, 0.31, 0.27, 0.27],
                # [0.10, 0.32, 0.29, 0.29],
                # [0.05, 0.33, 0.31, 0.31]
            ]
        }
    
    synonyms = input_data['synonyms']
    configurations = input_data['configuration']    
    print(f"Synonym set: {synonyms}")
    
    xchart = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ychart = []

    
    feature_importance_dict = {  
        'user_day_of_month_workload': [],
        'user_workload': [],
        # 'resource_type': [],
        'day_of_month': [],
        'month': [],
        # 'year': [],
        'day_of_week': [],
        'month_of_year_workload': [],
        'day_of_month_workload': [],
        'day_of_week_workload': [],
        'resource': []
    }
         
    # workload_avg_chart = {  
    #     'user_day_of_month_workload': [],
    #     'user_workload': [],
    #     'month_of_year_workload': [],
    #     'day_of_month_workload': [],
    #     'day_of_week_workload': []
    # }  
    #
    # workload_sum_chart = {  
    #     'user_day_of_month_workload': [],
    #     'user_workload': [],
    #     'month_of_year_workload': [],
    #     'day_of_month_workload': [],
    #     'day_of_week_workload': []
    # }     
    
    # period = "year_month"  
            
    # filename_roc = "loop_27jan_p2(6-1)_peak_yearmonth_forest_with_workload_multiple_roc"
    # filename_bar = "loop_27jan_p2(6-1)_peak_yearmonth_forest_with_workload_multiple_line"
            
    # filename_roc = "loop_27jan_p2(6-1)_peak_yearmonth_tree_with_workload_multiple_roc11"
    # filename_bar = "loop_27jan_p2(6-1)_peak_yearmonth_tree_with_workload_multiple_line11"
            
    #**********************************************************************************************************
    # period = "month_day"
    #
    # filename_roc = "loop_27jan_p2(6-2)_peak_daymonth_forest_with_workload_multiple_roc11"
    # filename_bar = "loop_27jan_p2(6-2)_peak_daymonth_forest_with_workload_multiple_line11"
            
    # filename_roc = "loop_27jan_p2(6-2)_peak_daymonth_tree_with_workload_multiple_roc"
    # filename_bar = "loop_27jan_p2(6-2)_peak_daymonth_tree_with_workload_multiple_line"
            
    #**********************************************************************************************************
    period = "week_day"
            
    filename_roc = "loop_27jan_p2(6-3)_peak_dayweek_forest_with_workload_multiple_roc11"
    filename_bar = "loop_27jan_p2(6-3)_peak_dayweek_forest_with_workload_multiple_line11"
            
    # filename_roc = "loop_27jan_p2(6-3)_peak_dayweek_tree_with_workload_multiple_roc"
    # filename_bar = "loop_27jan_p2(6-3)_peak_dayweek_tree_with_workload_multiple_line"   
        
    
    report = []
    for i, config in enumerate(configurations):        
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ===============================================================")
        print(f"Running configuration {i+1}: {config}")
        
        df = p2(config, synonyms, csv_url, NULL, period)
        # workload_avg_chart = get_avg_workload(df, synonyms, xchart, workload_avg_chart)
        # workload_sum_chart = get_sum_workload(df, synonyms, xchart, workload_sum_chart)
        result = analyze_event_log(df, synonyms)
        report.extend([result, {"config": config}])
        
        ychart.append(result['avg_auc'])
    
    
        for feature in feature_importance_dict.keys():
            feature_row = result['feature_importance'][result['feature_importance']['feature'] == feature]
            if not feature_row.empty:
                importance_value = feature_row['importance'].iloc[0]
                feature_importance_dict[feature].append(importance_value)
            else:
                feature_importance_dict[feature].append(0)
                
    ychart = [float(value) for value in ychart]    
    feature_importance_dict = {
        feature: [float(value) for value in values]
        for feature, values in feature_importance_dict.items()
    }
    
    print (f"ychart after change: {ychart}")
    print (f"feature_importance_dict after change: {feature_importance_dict}")
    
    # draw_chart.draw_multiple_lines_chart(xchart, workload_avg_chart, title='Avg Workload Features', xlabel='Event Percentage', ylabel='Average Workload', filename=filename_bar+"_avg_Workload")    
    # draw_chart.draw_multiple_lines_chart(xchart, workload_sum_chart, title='Sum Workload Features', xlabel='Event Percentage', ylabel='Sum Workload', filename=filename_bar+"_sum_Workload")    

    draw_chart.draw_chart_func(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', filename = filename_roc)
    draw_chart.draw_multiple_lines_chart(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', filename = filename_bar)
    # report = draw_chart.draw_charts_and_report(xchart, ychart, feature_importance_dict, report)    
    # report = draw_chart.draw_charts_and_report_scale(xchart, ychart, feature_importance_dict, report, xlabel='Event Percentage', yscale=(0, 1))   

    
    draw_chart.draw_chart_func_scale(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', yscale=(0, 1), filename = filename_roc+"_with_scale")
    draw_chart.draw_multiple_lines_chart_scale(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', yscale=(0, 1), filename = filename_bar+"_with_scale")
    # report = draw_chart.draw_charts_and_report_scale(xchart, ychart, feature_importance_dict, report)   
    
            
    return report

def pattern3(csv_url, cmodel):      
    
    filename_roc = "loop_27jan_p3(3)_typo_forest_with_workload_multiple_roc11"
    filename_bar = "loop_27jan_p3(3)_typo_forest_with_workload_multiple_line11"
            
    # filename_roc = "loop_27jan_p3(3)_typo_tree_with_workload_multiple_roc"
    # filename_bar = "loop_27jan_p3(3)_typo_tree_with_workload_multiple_line"
            
              
    input_data =  {
            'pattern': 'individual_typo',
             
            # 'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
            # 'synonyms': ["Create Purchase Order Item", "Create Purchase Order Itm", "Create Parchase Order Item", "Create Purchas Order Item"], #2
            # 'synonyms': ["Create Purchase Requisition Item", "Create Purchse Requisition Item", "Create Purchase Requisision Item-", "Create Purchase Requisision Item"],   #3
            # 'synonyms': ["Cancel Goods Receipt", "Cancell Goods Receipt", "Cancel Goods Receit", "Cancel Goods Reciept"],   #4
            # 'synonyms': ["SRM: Awaiting Approval", "SRM:waiting Approval", "SRM: Awaiting Approvval", "SRM: Awaiting Aproval_"],    #5
            # 'synonyms': ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],    #6
            # 'synonyms': ["Clear Invoice", "Clear Invoce", "Cleer Invoice", "Clear Invoise"],    #7
            # 'synonyms': ["Remove Payment Block", "Remove Payment Blok", "Remove Paymnt Block", "Remov Paymnet Block"],  #8
            # 'synonyms': ["Receive Order Confirmation", "Receive Order Confirmations", "Recieve Order Confirmation", "Receive Order Confermation"],  #9
            # 'synonyms': ["Change Quantity", "Change Quntity", "Change Qantity", "ChangeQuantity"],  #10
            'synonyms': ["Change Price", "Change Pric", "Changed Price", "Change Price _"], #11
            
            
            
            
            
            # 'synonyms': ["Cancel Invoice Receipt", "Cancel_Invoice_Receipt", "Cancell Invoice Receipt", "Cancel Invoice Receipt ", "Cancel Invoice Receipt-", "Cancel Invoice Receit", "Cancel Invoice Reciept", "Cancel Invoise Receipt"],
            # 'synonyms': ["Cancel Invoice Receipt", "Cancel Invoice Receit", "Cancel Invoice Reciept", "Cancel Invoise Receipt"],

            'configuration': [
                [0.30,0.03],
                [0.30,0.05],
                [0.30,0.10],
                [0.30,0.15],
                [0.30,0.20],
                [0.30,0.25],
                [0.30,0.30],
                [0.30,0.35],
                [0.30,0.40],
                [0.30,0.45],
                [0.30,0.50],
                # [0.20,0.55],
                # [0.20,0.60]
            ]
        }
    
    synonyms = input_data['synonyms']
    configurations = input_data['configuration']    
    print(f"Synonym set: {synonyms}")
    xchart = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ychart = []
    
    feature_importance_dict = {  
        'user_day_of_month_workload': [],
        'user_workload': [],
        # 'resource_type': [],
        'day_of_month': [],
        'month': [],
        # 'year': [],
        'day_of_week': [],
        'month_of_year_workload': [],
        'day_of_month_workload': [],
        'day_of_week_workload': [],
        'resource': []
    }
    
    
    report = []
    Shuffle = False
    for i, config in enumerate(configurations):        
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ===============================================================")
        print(f"Running configuration {i+1}: {config}")
        df = p3(synonyms[0], config[0], config[1], synonyms, csv_url, NULL, Shuffle)
        result = analyze_event_log(df, synonyms)
        report.extend([result, {"config": config}])
        
        ychart.append(result['avg_auc'])
    
    
        for feature in feature_importance_dict.keys():
            feature_row = result['feature_importance'][result['feature_importance']['feature'] == feature]
            if not feature_row.empty:
                importance_value = feature_row['importance'].iloc[0]
                feature_importance_dict[feature].append(importance_value)
            else:
                feature_importance_dict[feature].append(0)
                
                
    ychart = [float(value) for value in ychart]    
    feature_importance_dict = {
        feature: [float(value) for value in values]
        for feature, values in feature_importance_dict.items()
    }
    
    print (f"ychart after change: {ychart}")
    print (f"feature_importance_dict after change: {feature_importance_dict}")
    # draw_chart.draw_chart_func(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC')
    # draw_chart.draw_multiple_lines_chart(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance')
    # report = draw_chart.draw_charts_and_report(xchart, ychart, feature_importance_dict, report)
    
    draw_chart.draw_chart_func(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', filename = filename_roc)
    draw_chart.draw_multiple_lines_chart(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', filename = filename_bar)
    
        
    draw_chart.draw_chart_func_scale(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', yscale=(0, 1), filename = filename_roc+"_with_scale")
    draw_chart.draw_multiple_lines_chart_scale(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', yscale=(0, 1), filename = filename_bar+"_with_scale")
    # report = draw_chart.draw_charts_and_report_scale(xchart, ychart, feature_importance_dict, report, xlabel='Event Percentage', yscale=(0, 1))
        
    return report

def pattern4(csv_url, cmodel):    
    
    filename_roc = "loop_28jan_p4(1)_overall_workload_forest_with_workload_multiple_roc11"
    filename_bar = "loop_28jan_p4(1)_overall_workload_forest_with_workload_multiple_line11"
            
    # filename_roc = "loop_27jan_p4(1)_overall_workload_tree_with_workload_multiple_roc"
    # filename_bar = "loop_27jan_p4(1)_overall_workload_tree_with_workload_multiple_line"
                
    input_data =  {
            'pattern': 'overall_workload',
            
            # 'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
            # 'synonyms': ["Create Purchase Order Item", "Create Purchase Order Itm", "Create Parchase Order Item", "Create Purchas Order Item"], #2
            # 'synonyms': ["Create Purchase Requisition Item", "Create Purchse Requisition Item", "Create Purchase Requisision Item-", "Create Purchase Requisision Item"],   #3
            # 'synonyms': ["Cancel Goods Receipt", "Cancell Goods Receipt", "Cancel Goods Receit", "Cancel Goods Reciept"],   #4
            # 'synonyms': ["SRM: Awaiting Approval", "SRM:waiting Approval", "SRM: Awaiting Approvval", "SRM: Awaiting Aproval_"],    #5
            # 'synonyms': ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],    #6
            # 'synonyms': ["Clear Invoice", "Clear Invoce", "Cleer Invoice", "Clear Invoise"],    #7
            # 'synonyms': ["Remove Payment Block", "Remove Payment Blok", "Remove Paymnt Block", "Remov Paymnet Block"],  #8
            # 'synonyms': ["Receive Order Confirmation", "Receive Order Confirmations", "Recieve Order Confirmation", "Receive Order Confermation"],  #9
            # 'synonyms': ["Change Quantity", "Change Quntity", "Change Qantity", "ChangeQuantity"],  #10
            'synonyms': ["Change Price", "Change Pric", "Changed Price", "Change Price _"], #11
            
            
            
            
            
            # 'synonyms': ["Change Quantity", "Change Quntity", "Change Qantity", "ChangeQuantity"],

            'configuration': [
                [0.30,0.03],
                [0.30,0.05],
                [0.30,0.10],
                [0.30,0.15],
                [0.30,0.20],
                [0.30,0.25],
                [0.30,0.30],
                [0.30,0.35],
                [0.30,0.40],
                [0.30,0.45],
                [0.30,0.50],
                # [0.30,0.55],
                # [0.30,0.60]
            ]
        }
    
    synonyms = input_data['synonyms']
    configurations = input_data['configuration']    
    print(f"Synonym set: {synonyms}")
    xchart = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # xchart = [3, 5, 10, 15, 20]
    ychart = []
    

    feature_importance_dict = {  
        'user_day_of_month_workload': [],
        'user_workload': [],
        # 'resource_type': [],
        'day_of_month': [],
        'month': [],
        # 'year': [],
        'day_of_week': [],
        'month_of_year_workload': [],
        'day_of_month_workload': [],
        'day_of_week_workload': [],
        'resource': []
    }
    
    report = []
    
    workload_avg_chart = {  
        'user_day_of_month_workload': [],
        'user_workload': [],
        'month_of_year_workload': [],
        'day_of_month_workload': [],
        'day_of_week_workload': []
    }  
    
    workload_sum_chart = {  
        'user_day_of_month_workload': [],
        'user_workload': [],
        'month_of_year_workload': [],
        'day_of_month_workload': [],
        'day_of_week_workload': []
    }
    
    
    for i, config in enumerate(configurations):        
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ===============================================================")
        print(f"Running configuration {i+1}: {config}")
        df = p4(synonyms[0], config[0], config[1], synonyms, csv_url, NULL)
        # workload_avg_chart = get_avg_workload(df, synonyms, xchart, workload_avg_chart)
        # workload_sum_chart = get_sum_workload(df, synonyms, xchart, workload_sum_chart)
        result = analyze_event_log(df, synonyms)
        report.extend([result, {"config": config}])
        
        ychart.append(result['avg_auc'])
    
    
        for feature in feature_importance_dict.keys():
            feature_row = result['feature_importance'][result['feature_importance']['feature'] == feature]
            if not feature_row.empty:
                importance_value = feature_row['importance'].iloc[0]
                feature_importance_dict[feature].append(importance_value)
            else:
                feature_importance_dict[feature].append(0)
        
    ychart = [float(value) for value in ychart]    
    feature_importance_dict = {
        feature: [float(value) for value in values]
        for feature, values in feature_importance_dict.items()
    }
    
    print (f"ychart after change: {ychart}")
    print (f"feature_importance_dict after change: {feature_importance_dict}")
    # draw_chart.draw_multiple_lines_chart(xchart, workload_avg_chart, title='Avg Workload Features', xlabel='Event Percentage', ylabel='Average Workload', filename=filename_bar+"_avg_Workload")    
    # draw_chart.draw_multiple_lines_chart(xchart, workload_sum_chart, title='Sum Workload Features', xlabel='Event Percentage', ylabel='Sum Workload', filename=filename_bar+"_sum_Workload")    

    draw_chart.draw_chart_func(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', filename = filename_roc)
    draw_chart.draw_multiple_lines_chart(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', filename = filename_bar)
    draw_chart.draw_chart_func_scale(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', yscale=(0, 1), filename = filename_roc+"_with_scale")
    draw_chart.draw_multiple_lines_chart_scale(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', yscale=(0, 1), filename = filename_bar+"_with_scale")
    
    # report = draw_chart.draw_charts_and_report(xchart, ychart, feature_importance_dict, report)  
    # report = draw_chart.draw_charts_and_report_scale(xchart, ychart, feature_importance_dict, report, xlabel='Event Percentage', yscale=(0, 1))  
    return report

def pattern5(csv_url, cmodel):    
    
    
    filename_roc = "loop_27jan_p5(2)_individual_convention_forest_with_workload_multiple_roc11"
    filename_bar = "loop_27jan_p5(2)_individual_convention_forest_with_workload_multiple_line11"
            
    # filename_roc = "loop_27jan_p5(2)_individual_convention_tree_with_workload_multiple_roc"
    # filename_bar = "loop_27jan_p5(2)_individual_convention_tree_with_workload_multiple_line"
            
    # filename_roc = "loop_27jan_p5(2)_individual_convention_forest_without_workload_multiple_roc"
    # filename_bar = "loop_27jan_p5(2)_individual_convention_forest_without_workload_multiple_line"
            
    # filename_roc = "loop_27jan_p5(2)_individual_convention_tree_without_workload_multiple_roc"
    # filename_bar = "loop_27jan_p5(2)_individual_convention_tree_without_workload_multiple_line"
            
                
    input_data =  {
            'pattern': 'Pattern individual_convention',
            # 'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
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
            
            
            
            
            
            # 'synonyms': ["Change Price", "Alter Price", "Change rate", "Alter Rate"],
            # 'synonyms': ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],    #new_label
            # 'synonyms': ["Remove Payment Block", "Clear Payment Block", "Eliminate Payment", "Cancel Payment Block"],
            'configuration': [
                # [0.9, 0.03, 0.03, 0.04],
                # [0.8, 0.1, 0.05, 0.05],
                [0.7, 0.1, 0.1, 0.1],
                [0.6, 0.15, 0.15, 0.1],
                [0.5, 0.15, 0.15, 0.2],
                [0.4, 0.2, 0.2, 0.2], 
                [0.3, 0.2, 0.25, 0.25],
                [0.2, 0.3, 0.25, 0.25]   
                # [0.95, 0.01, 0.02, 0.02],
                # [0.90, 0.03, 0.03, 0.04],
                # [0.85, 0.04, 0.05, 0.06],
                # [0.80, 0.10, 0.05, 0.05],
                # [0.75, 0.11, 0.07, 0.07],
                # [0.70, 0.10, 0.10, 0.10],
                # [0.65, 0.11, 0.12, 0.12],
                # [0.60, 0.15, 0.15, 0.10],
                # [0.55, 0.16, 0.17, 0.12],
                # [0.50, 0.15, 0.15, 0.20],
                # [0.45, 0.16, 0.17, 0.22],
                # [0.40, 0.20, 0.20, 0.20]
            ] 
        }
    
    synonyms = input_data['synonyms']
    configurations = input_data['configuration']    
    print(f"Synonym set: {synonyms}")
    
    # xchart = [20, 30, 40, 50, 60, 70, 80]
    # xchart = [30, 40, 50, 60, 70, 80]
    xchart = []
    ychart = []
  
    
    feature_importance_dict = {  
        'user_day_of_month_workload': [],
        'user_workload': [],
        # 'resource_type': [],
        'day_of_month': [],
        'month': [],
        # 'year': [],
        'day_of_week': [],
        'month_of_year_workload': [],
        'day_of_month_workload': [],
        'day_of_week_workload': [],
        'resource': []
    }
    
    workload_avg_chart = {  
        'user_day_of_month_workload': [],
        'user_workload': [],
        'month_of_year_workload': [],
        'day_of_month_workload': [],
        'day_of_week_workload': []
    }  
    
    workload_sum_chart = {  
        'user_day_of_month_workload': [],
        'user_workload': [],
        'month_of_year_workload': [],
        'day_of_month_workload': [],
        'day_of_week_workload': []
    }
    
    report = []
    for i, config in enumerate(configurations):        
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ===============================================================")
        print(f"Running configuration {i+1}: {config}")
        shuffle = False
        event_seprated=True
        # df = p5(synonyms[0], config, synonyms, csv_url, NULL, shuffle, event_seprated)
        df, split_dfs = p5(config, synonyms, csv_url, NULL, shuffle, event_seprated)
        # workload_avg_chart = get_avg_workload(df, synonyms, xchart, workload_avg_chart)
        # workload_sum_chart = get_sum_workload(df, synonyms, xchart, workload_sum_chart)
        
        event_percentage = calculate_event_percentages(split_dfs)
        xchart.append(event_percentage)

        result = analyze_event_log(df, synonyms)
        report.extend([result, {"config": config}])
        
        ychart.append(result['avg_auc'])
    
    
        for feature in feature_importance_dict.keys():
            feature_row = result['feature_importance'][result['feature_importance']['feature'] == feature]
            if not feature_row.empty:
                importance_value = feature_row['importance'].iloc[0]
                feature_importance_dict[feature].append(importance_value)
            else:
                feature_importance_dict[feature].append(0)
    ychart = [float(value) for value in ychart]    
    feature_importance_dict = {
        feature: [float(value) for value in values]
        for feature, values in feature_importance_dict.items()
    }
    
    print(f"Actual event percentages (xchart): {xchart}")
    print (f"ychart after change: {ychart}")
    print (f"feature_importance_dict after change: {feature_importance_dict}")
    
    # draw_chart.draw_multiple_lines_chart(xchart, workload_avg_chart, title='Avg Workload Features', xlabel='Event Percentage', ylabel='Average Workload', filename=filename_bar+"_avg_Workload")    
    # draw_chart.draw_multiple_lines_chart(xchart, workload_sum_chart, title='Sum Workload Features', xlabel='Event Percentage', ylabel='Sum Workload', filename=filename_bar+"_sum_Workload")    
    
    
    draw_chart.draw_chart_func(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', filename = filename_roc)
    draw_chart.draw_multiple_lines_chart(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', filename = filename_bar)
    
    draw_chart.draw_chart_func_scale(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', yscale=(0, 1), filename = filename_roc+"_with_scale")
    draw_chart.draw_multiple_lines_chart_scale(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', yscale=(0, 1), filename = filename_bar+"_with_scale")

    # report = draw_chart.draw_charts_and_report(xchart, ychart, feature_importance_dict, report)  
    # report = draw_chart.draw_charts_and_report_scale(xchart, ychart, feature_importance_dict, report, xlabel='Event Percentage', yscale=(0, 1))  
    
    return report

def pattern6(csv_url, cmodel):   
    
    filename_roc = "loop_27jan_p6(5)_workload_monthly_forest_with_workload_multiple_roc11"
    filename_bar = "loop_27jan_p6(5)_workload_monthly_forest_with_workload__multiple_line11"
            
    # filename_roc = "loop_27jan_p6(5)_workload_monthly_tree_with_workload_multiple_roc"
    # filename_bar = "loop_27jan_p6(5)_workload_monthly_tree_with_workload__multiple_line"
            
                 
    input_data =  {
            'pattern': 'workload_monthly',
            # 'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
            # 'synonyms': ["Create Purchase Order Item", "Generate Purchase Request", "Order Item Creation", "Register Order"],#2
            # 'synonyms': ["Create Purchase Requisition Item", "Initiate Purchase Request", "Add Purchase Request Item", "Log New Requisition Item"],#3
            # 'synonyms': ["Cancel Goods Receipt", "Cancel Delivery Record", "Delete Receipt Entry", "Stop Goods Record"],#4
            # 'synonyms': ["SRM: Awaiting Approval", "SRM: Under Review", "SRM: Pending Decision", "SRM: Approval in Process"],#5
            # 'synonyms': ["Remove Payment Block", "Clear Payment Block", "Eliminate Payment", "Cancel Payment Block"], #6
            # 'synonyms': ["Record Invoice Receipt", "Log Invoice Received", "Invoice Documentation", "Register Supplier Invoice"], #7
            # 'synonyms': ["Clear Invoice", "Invoice Finalised", "Invoice Processed", "Close Invoice"], #8
            # 'synonyms': ["Receive Order Confirmation", "Accept Order Details", "Get Order Confirmation", "Retrieve Order Details"], #9
            # 'synonyms': ["Change Quantity", "Modify Amount", "Alter Quantity", "Update Count"], #10
            'synonyms': ["Change Price", "Modify Cost", "Revise Rate", "Update Price"], #11
            
            
            
            
            
            
            
            
            # 'synonyms': ["Record Goods Receipt", "Receive Shipment", "Goods Delivery Confirmation", "Goods Received Log"],  #1
            # 'synonyms': ["Delete Purchase Order Item", "Delete Purchase Order Itm", "Delete Parchase Order Item", "Delete Purchas Order Item"],
            # 'synonyms' : [ "Create Purchase Order Item", "Create Purchase Order Itm", "Create Parchase Order Item", "Create Purchas Order Item"], #new_label
            'configuration': [
                [0.30,0.03],
                [0.30,0.05],
                [0.30,0.10],
                [0.30,0.15],
                [0.30,0.20],
                [0.30,0.25],
                [0.30,0.30],
                [0.30,0.35],
                [0.30,0.40],
                [0.30,0.45],
                [0.30,0.50],
                # [0.20,0.55],
                # [0.20,0.60]
            ]
        }
    
    minimum_day_activity = 5
    synonyms = input_data['synonyms']
    configurations = input_data['configuration']    
    print(f"Synonym set: {synonyms}")   
    
    xchart = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ychart = []
    # feature_importance_dict = {  
    #     'day_of_month_workload': [],
    #     'overal_workload': [],
    #     'resource_type': [],
    #     'day_of_month': [],
    #     'month': [],
    #     'year': [],
    #     'day_of_week': [],
    #     'is_peak_year_month': [],
    #     'is_peak_day_of_month': [],
    #     'is_peak_day_of_week': []
    # }
     
    feature_importance_dict = {  
        'user_day_of_month_workload': [],
        'user_workload': [],
        # 'resource_type': [],
        'day_of_month': [],
        'month': [],
        # 'year': [],
        'day_of_week': [],
        'month_of_year_workload': [],
        'day_of_month_workload': [],
        'day_of_week_workload': [],
        'resource': []
    }
    
    
    shuffle = False
    report = []
    for i, config in enumerate(configurations):        
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " ===============================================================")
        print(f"Running configuration {i+1}: {config}")
        df = p6(synonyms[0], config[0], config[1], synonyms, csv_url, NULL, minimum_day_activity, shuffle)
        result = analyze_event_log(df, synonyms)
        report.extend([result, {"config": config}])
        
        ychart.append(result['avg_auc'])
    
    
        for feature in feature_importance_dict.keys():
            feature_row = result['feature_importance'][result['feature_importance']['feature'] == feature]
            if not feature_row.empty:
                importance_value = feature_row['importance'].iloc[0]
                feature_importance_dict[feature].append(importance_value)
            else:
                feature_importance_dict[feature].append(0)
    
    
    ychart = [float(value) for value in ychart]    
    feature_importance_dict = {
        feature: [float(value) for value in values]
        for feature, values in feature_importance_dict.items()
    }
    
    print (f"ychart after change: {ychart}")
    print (f"feature_importance_dict after change: {feature_importance_dict}")
    draw_chart.draw_chart_func(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', filename = filename_roc)
    draw_chart.draw_multiple_lines_chart(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', filename = filename_bar)
    
    
    draw_chart.draw_chart_func_scale(xchart, ychart, chart_type='line', title='Avg AUC Chart', xlabel='Event Percentage', ylabel='Average AUC', yscale=(0, 1), filename = filename_roc+"_with_scale")
    draw_chart.draw_multiple_lines_chart_scale(xchart, feature_importance_dict, title='Feature Importance Chart', xlabel='Event Percentage', ylabel='Feature Importance', yscale=(0, 1), filename = filename_bar+"_with_scale")


    # report = draw_chart.draw_charts_and_report(xchart, ychart, feature_importance_dict, report)
    # report = draw_chart.draw_charts_and_report_scale(xchart, ychart, feature_importance_dict, report, xlabel='Event Percentage', yscale=(0, 1))
        
    return report

def get_avg_workload(df, synonyms, xchart, workload_avg_chart):  
    injected_labels = synonyms[1:]    
    # filtered_df = df[df['event concept:name'].isin(synonyms[1:])]   
    filtered_df = df[df['event concept:name'].isin(injected_labels)]
    
    # Debug information
    print(f"\nDebug Information:")
    print(f"Total records in original df: {len(df)}")
    print(f"Records matching injected labels: {len(filtered_df)}")
    print(f"Unique values in event concept:name: {df['event concept:name'].unique()}")
    print(f"Injected labels being searched for: {injected_labels}")
    
     
    for feature in workload_avg_chart.keys():
        # workload_avg_chart[feature].append(filtered_df[feature].mean())
        avg_value = filtered_df[feature].mean()
        workload_avg_chart[feature].append(avg_value)
        # Debug information for each feature
        print(f"\nFeature: {feature}")
        print(f"Number of non-ull values: {filtered_df[feature].count()}")
        print(f"Value range: {filtered_df[feature].min()} to {filtered_df[feature].max()}")
        print(f"Calculated average: {avg_value}")
        # print(f"values_count of concept name: {df['event concept:name'].value_counts()}")
        # print(f"describe {df[df['event concept:name'].isin(synonyms[1:])][feature].describe()}")
        # print (f"workload_avg: {workload_avg_chart[feature]}: filtered_df[feature].mean()")
        # workload_avg_chart[feature].append(filtered_df[feature].sum())    
    return workload_avg_chart

def get_sum_workload(df, synonyms, xchart, workload_sum_chart):      
    filtered_df = df[df['event concept:name'].isin(synonyms[1:])]    
    for feature in workload_sum_chart.keys():
        # workload_sum_chart[feature].append(filtered_df[feature].mean())
        workload_sum_chart[feature].append(filtered_df[feature].sum())    
        print (f"workload_sum: {workload_sum_chart[feature]}: filtered_df[feature].sum()")
    return workload_sum_chart
    

if __name__ == "__main__":
    
    from datetime import datetime

    current_date = datetime.now()
    formatted_date = current_date.strftime("%d_%b_%Y").lower()    
    # classification_model = 'forest';  
    classification_model = 'tree';    
    description = 'with_ranked_workload_mohammad';    
    log_file = "log/" + formatted_date + '_' + classification_model + '_' + description + '.txt'  
    # log_file = "log/sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss.txt"
    
    
    sys.stdout = Logger(log_file)
    # csv_url = 'data/BPI_with_new_features.csv'
    # csv_url = 'data/BPI_with_new_numeric_features_v3.csv'
    # csv_url = 'data/BPI_with_new_numeric_features_v1_with_user.csv'
    csv_url = 'data/BPI_with_new_numeric_features_v3_with_user.csv'
    # csv_url = 'data/BPI_with_new_numeric_features_v4_with_ranked_workload.csv'
    # csv_url = 'data/BPI_with_new_numeric_features_v4_with_10ranked_workload.csv'
    # csv_url = 'data/BPI2019_with_new_numeric_features_v4_with_20ranked_workload.csv'
    # csv_url = 'data/BPI2019_with_10_ranked_for_5_workload_features_v7.csv'
    # csv_url = 'data/BPI2019_with_5_ranked_for_5_workload_features_v8.csv'
    # csv_url = 'data/BPI2019_with_10_ranked_for_5_workload_features_v10.csv'
    # csv_url = 'data/BPI2019_with_20_ranked_for_5_workload_features_v11.csv'
    # csv_url = 'data/BPI2019_with_10_user_5_time_ranked_for_workload_features_v15.csv'
    # csv_url = 'data/BPI2019_with_10_user_5_time_ranked_for_workload_features_without_system_for_time_v16.csv'
    # csv_url = 'data/BPI2019_with_10_user_5_time_balanced_ranked_for_workload_features_without_system_for_time_v17.csv'




    

    
    
    
    # synonym_sets = [
    #     ["Clear Invoice", "Clear Invoce", "Cleer Invoice", "Clear Invoise"],
    #     ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"],
    #     ["Create Purchase Requisition Item", "Create Purchse Requisition Item", "Create Purchase Requisision Item-", "Create Purchase Requisision Item"],
    #     ["Cancel Invoice Receipt", "Cancel Invoice Receit", "Cancel Invoice Reciept", "Cancel Invoise Receipt"],
    #     ["Change Quantity", "Change Quntity", "Change Qantity", "ChangeQuantity"],
    #     ["Remove Payment Block", "Clear Payment Block", "Eliminate Payment", "Cancel Payment Block"],
    #     ["Delete Purchase Order Item", "Delete Purchase Order Itm", "Delete Parchase Order Item", "Delete Purchas Order Item"]
    # ]
    
        
    # report = pattern1(csv_url, classification_model)
    # report = pattern2(csv_url, classification_model)
    # report = pattern3(csv_url, classification_model)
    # report = pattern4(csv_url, classification_model)
    # report = pattern5(csv_url, classification_model)
    report = pattern6(csv_url, classification_model)
        
    # report = 'none of patterns' 
        
        
    print(report)
    
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
             
            
# df = pd.read_csv('data/BPI_Challenge_2019.csv') 
            
#  #### dont print console for this           
# # original_stdout = sys.stdout  # Save the original stdout
# # sys.stdout = open(os.devnull, 'w')  # Redirect stdout to a null device
# df = p1(False, config, synonyms, csv_url, '')
# # sys.stdout = original_stdout