import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

class Explanation:
    def get_feature_importance(self, model, X_train):
        return pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })#.sort_values('importance', ascending=False)

    def get_shap_values(self, model, data):
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(data)

    def calculate_summed_shap_values(self, shap_values):
        return np.mean(shap_values, axis=0)

    def calculate_summed_abs_shap_values(self, shap_values):
        abs_shap_values = np.abs(shap_values)
        return np.mean(abs_shap_values, axis=0)

    def print_shap_values(self, class_names, features, shap_values, label):
        print(f"{label} total")
        self.printOneShapValue(class_names, features, shap_values)

    def print_feature_interactions(self, shap_values, class_names, feature_names):
        result = []
        for i, feature1 in enumerate(feature_names):
            for j, feature2 in enumerate(feature_names):
                interaction_row = {"Feature Interaction": f"{feature1}/{feature2}"}
                for class_index, class_name in enumerate(class_names):
                    interaction_row[class_name] = round(shap_values[i, j, class_index], 4) * 100
                result.append(interaction_row)
        
        interaction_df = pd.DataFrame(result)
        print("Feature Interactions:")
        print(interaction_df)
        
    def plot_shap_summary(self, results, class_names):
        plt.figure(figsize=(12, 6))
        
        # Set up positions for bars
        resource_types = list(results.keys())
        x = np.arange(len(class_names))
        width = 0.35
        
        # Create grouped bars
        for i, resource_type in enumerate(resource_types):
            resource_name = "Human" if resource_type == 1 else "System"
            avg_values = results[resource_type]['average']
            plt.bar(x + i*width, avg_values, width, 
                    label=resource_name,
                    alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Average SHAP Value')
        plt.title('Average SHAP Values by Resource Type and Class')
        plt.xticks(x + width/2, class_names)
        plt.legend()
        
        # Add value labels on top of bars
        for i, resource_type in enumerate(resource_types):
            avg_values = results[resource_type]['average']
            for j, v in enumerate(avg_values):
                plt.text(j + i*width, v, f'{v:.3f}', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    def printOneShapValue(self, label_set, row_labels, shap_value):    
        print("\t".join(label_set))
        for i, row in enumerate(shap_value):
            rounded_values = [f"{value:.4f}" for value in row]
            print("\t".join(rounded_values) + "\t" + row_labels[i])