from filter_data import DataProcessor
from classification import run_pattern
from classification_only import analyse_log
from frequency_features import main as frequency_features_main
from additional_feature_final_version import print_analysis
from probability_calculation import main as probability_calculation_main
from probability_chart import main as probability_chart_main
import numpy as np

if __name__ == "__main__":
    input_csv = 'data/BPI_Challenge_2019.csv'
    output_csv = 'data/BPI_with_new_numeric_features_v3_with_user.csv'

    need_process = False
    if(need_process):
        processor = DataProcessor(input_csv, output_csv)
        processor.load_data()
        processor.add_new_features()
    
    csv_url = output_csv 
        
    labels = ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"]
    input_data_pattern4 =  {
        'pattern': 'Pattern Overall Workload',        
        'synonyms': labels,
        'configuration': [[0.2, 0.3]]
    }
    
    need_inject = False
    if(need_inject):
        report, frequencies, importance_values = run_pattern(csv_url, input_data_pattern4)
    else:        
        csv_url = 'data/version2WithFeatures.csv'
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
        column_name = 'ActionTakenToComply (CA)'
        Label_number = 'Label1109Inspectionv4'   
        synonyms =[ "See attachment", "See attached email.", "email attached"]
        report, frequencies, importance_values = analyse_log(csv_url, synonyms, column_name, Label_number, features)
    
    print("Classification Report:", report)    
    print("Frequencies:", frequencies)
    print("Importance Values:", importance_values)
    
    
    human_labels = ["Record Invoice Receipt"] 
    system_labels = ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"]
    variance_metric = frequency_features_main(labels, frequencies, human_labels, system_labels)
    results = print_analysis(
        labels, 
        "Different Terminology Set",
        size_threshold=8
    )
    
    
    
    shap_feature_vector = np.array(importance_values) 
    raw_feature_vector = np.array([
        results['average_distance'],
        variance_metric,
        results['edit_similarity'],
        results['mutable_text_edit_distance']  
    ]) 
    
    probabilities = probability_calculation_main(
        shap_feature_vector,
        raw_feature_vector
    )
    
    probability_chart_main(np.array(probabilities))

    
    
