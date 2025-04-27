import numpy as np
from sklearn.preprocessing import MinMaxScaler
from babel.messages.frontend import log



def softmax(z):
    exp_z = np.exp(z - np.max(z)) 
    return exp_z / np.sum(exp_z)

def main(shap_feature_vector, raw_feature_vector):
    scaler = MinMaxScaler()

    raw_feature_vector = scaler.fit_transform(raw_feature_vector.reshape(-1, 1)).flatten()


    shap_weights = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 5],              # Pattern 1- system_user  with workload weights
        [0, 0 ,0.5, 1, 0, 3.5, 0, 0, 0],          # P2 - peak month of year
        [0, 0 , 0, 0, 1, 0, 0, 4, 0],             # p2 - peak day of week
        [0, 0 , 1,0, 0, 0, 4, 0, 0],              # P2 - peak day of month
        [1, 3 , 0, 0, 0, 0, 0, 0, 1],             # p4 - overall workload
        [0, 4 , 0, 0, 0, 0, 0, 0, 1],             # p5 - individual convention
        [3.6, 0.2 , 0.5, 0.5, 0, 0, 0, 0, 0.2],   # P6 - monthly workload
    ])

    raw_weights = np.array([
        [5, 2, 1 ,2],      # Pattern 1- system_user  with workload weights (SHAP)
        [1, 3, 5, 1],      # P2 - peak month of year
        [1, 3, 5, 1],      # p2 - peak day of week
        [1, 3, 5, 1],      # P2 - peak day of month
        [1, 3, 5, 1],      # p4 - overall workload
        [5, 1, 1, 3],      # p5 - individual convention
        [1, 3, 5, 1],      # P6 - monthly workload 
    ])

    raw_scores = np.dot(raw_weights, raw_feature_vector)

    shap_scores = np.dot(shap_weights, shap_feature_vector)

    final_scores = raw_scores + shap_scores

    probabilities = softmax(final_scores)

    print("Raw Scores:", raw_scores)
    print("SHAP Scores:", shap_scores)
    print("Final Combined Scores:", final_scores)
    print("Probabilities:", probabilities)
    
    return probabilities
    
    
shap_feature_vector = [
            'user_day_of_month_workload', #1
            'user_workload', #2
            'day_of_month', #3 
            'month', #4
            'day_of_week', #5
            'month_of_year_workload', #6
            'day_of_month_workload', #7
            'day_of_week_workload',#8
            'resource' #9
]

raw_feature_vector = ['edit_distance', 'variance', 'edit_similarity', 'mutable_edit_distance']

shap_feature_vector = np.array([0.000000, 0.000000, 0.000441, 0.000358, 0.002265, 0.002593, 0.000464, 0.000939, 0.992941]) 
raw_feature_vector = np.array([0.833, 0, 0.167, 0.875])  # other features

main(shap_feature_vector, raw_feature_vector)


