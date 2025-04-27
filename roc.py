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
from explanation import Explanation  # Import the class, not the module

class Roc:
    def __init__(self):
        self.final_result = []
        self.label_encoders = {}
        self.explanation = Explanation()
        # self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
    
    def roc(self,y_test,X_test,model, report, y, chart=True, filename=None):
        # n_classes = len(np.unique(y))
        classes = np.sort(np.unique(y))
        n_classes = len(classes)
        
        # y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))
        y_test_binarized = label_binarize(y_test, classes=classes)
        
        if n_classes == 2:
            y_test_binarized = np.hstack((1 - y_test_binarized, y_test_binarized))
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = model.decision_function(X_test)
            if y_pred_proba.ndim == 1:  # Binary classification (NEW)
                y_pred_proba = np.vstack([(1 - y_pred_proba), y_pred_proba]).T
            else:  # Multi-class
                y_pred_proba = np.exp(y_pred_proba)
                y_pred_proba /= np.sum(y_pred_proba, axis=1, keepdims=True)
            # y_pred_proba = np.exp(y_pred_proba) / np.sum(np.exp(y_pred_proba), axis=1, keepdims=True)
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        if chart:
            self.plot_roc_curve(fpr, tpr, roc_auc, n_classes, filename)
        
        avg_auc = np.mean(list(roc_auc.values()))
        return report, avg_auc
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, n_classes, filename):
        plt.figure()
        colors = ['blue', 'green', 'red', 'purple']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve (Multi-Class)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if filename:
            plt.savefig(f"charts/{filename}.png", bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
