from filter_data import DataProcessor
from classification import (
    run_pattern,
)
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
    
    report, frequencies, importance_values = run_pattern(csv_url, input_data_pattern4)
    
    print("Classification Report:", report)    
    print("Frequencies:", frequencies)
    
    
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

    
    
