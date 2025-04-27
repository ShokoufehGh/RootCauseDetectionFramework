

import matplotlib.pyplot as plt

# def draw_chart_func(x, y, chart_type='line', title='Chart', xlabel='X-Axis', ylabel='Y-Axis', yscale=(0, 1)):
def draw_chart_func(x, y, chart_type='line', title='Chart', xlabel='X-Axis', ylabel='Y-Axis', filename=None):


    plt.figure(figsize=(8, 6))
    
    chart_type = chart_type.lower()
    
    if chart_type == 'line':
        plt.plot(x, y, marker='o', linestyle='-', color='blue')
    elif chart_type == 'bar':
        plt.bar(x, y)
    elif chart_type == 'scatter':
        plt.scatter(x, y)
    else:
        print(f"Unsupported chart type: {chart_type}")
        return
    
    # plt.ylim(yscale[0], yscale[1])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"charts/{filename}.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


 
def draw_chart_func_scale(x, y, chart_type='line', title='Chart', xlabel='X-Axis', ylabel='Y-Axis', yscale=(0, 1), filename=None):


    plt.figure(figsize=(8, 6))
    
    chart_type = chart_type.lower()
    
    if chart_type == 'line':
        plt.plot(x, y, marker='o', linestyle='-', color='blue')
    elif chart_type == 'bar':
        plt.bar(x, y)
    elif chart_type == 'scatter':
        plt.scatter(x, y)
    else:
        print(f"Unsupported chart type: {chart_type}")
        return
    
    plt.ylim(yscale[0], yscale[1])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    
    if filename:
        plt.savefig(f"charts/{filename}.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    

def draw_chart_func2(x, y, chart_type='line', title='Chart', xlabel='X-Axis', ylabel='Y-Axis', yscale=(0, 1), filename=None):
# def draw_chart_func2(x, y, chart_type='line', title='Chart', xlabel='X-Axis', ylabel='Y-Axis'):

    
    
    plt.figure(figsize=(12, 6))
    
    # Extract values from the feature importance DataFrame and sort
    feature_importance_df = y[0]
    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    # Get the sorted features and importance values
    sorted_features = feature_importance_df['feature'].values
    sorted_importance = feature_importance_df['importance'].values
    
    chart_type = chart_type.lower()
    
    if chart_type == 'line':
        plt.plot(sorted_features, sorted_importance, marker='o', linestyle='-', color='blue')
    elif chart_type == 'bar':
        plt.bar(sorted_features, sorted_importance)
    elif chart_type == 'scatter':
        plt.scatter(sorted_features, sorted_importance)
    else:
        print(f"Unsupported chart type: {chart_type}")
        return
    
    plt.ylim(yscale[0], yscale[1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"charts/{filename}.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
    
def draw_chart_func2_without_scale(x, y, chart_type='line', title='Chart', xlabel='X-Axis', ylabel='Y-Axis', filename=None):

    
    
    plt.figure(figsize=(12, 6))
    
    # Extract values from the feature importance DataFrame and sort
    feature_importance_df = y[0]
    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    # Get the sorted features and importance values
    sorted_features = feature_importance_df['feature'].values
    sorted_importance = feature_importance_df['importance'].values
    
    chart_type = chart_type.lower()
    
    if chart_type == 'line':
        plt.plot(sorted_features, sorted_importance, marker='o', linestyle='-', color='blue')
    elif chart_type == 'bar':
        plt.bar(sorted_features, sorted_importance)
    elif chart_type == 'scatter':
        plt.scatter(sorted_features, sorted_importance)
    else:
        print(f"Unsupported chart type: {chart_type}")
        return
    
    # plt.ylim(yscale[0], yscale[1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"charts/{filename}.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    


    
    
# def draw_multiple_lines_chart(x, y_dict, title='Chart', xlabel='X-Axis', ylabel='Y-Axis', yscale=(0, 1)):
    
def draw_multiple_lines_chart(x, y_dict, title='Chart', xlabel='X-Axis', ylabel='Y-Axis', filename=None):

 
    plt.figure(figsize=(12, 8))  
    
    # colors = plt.cm.Set3(np.linspace(0, 1, 11)) 
    
    # colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF',
    #       '#FFA500', '#800080', '#008000', '#000080', '#FFD700', '#FF1493']
    
    # colors = ['#E41A1C', '#4DAF4A', '#377EB8', '#984EA3', '#FF7F00',
    #       '#FFFF33', '#A65628', '#F781BF', '#999999', '#66C2A5', '#FC8D62']
    
     
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    #       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']
    
    # colors = ['#800000', '#808000', '#4363d8', '#000000', '#a9a9a9',
    #   '#e6194B', '#ffe119', '#f58231', '#911eb4', '#3cb44b', '#f032e6']
    
    colors = ['#e6194B', '#3cb44b', '#4363d8', '#000000', '#808000',
      '#f032e6', '#ffe119', '#f58231', '#911eb4']

    for i, (feature, y_values) in enumerate(y_dict.items()):
        plt.plot(x, y_values, marker='o', label=feature, color=colors[i])
    
    
    # for feature, y_values in y_dict.items():
    #     plt.plot(x, y_values, marker='o', label=feature)   
        # plt.plot(x, y_values, marker='o', label=feature, linestyle='-')

    # plt.ylim(yscale[0], yscale[1])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(loc='best') 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"charts/{filename}.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
def draw_multiple_lines_chart_scale(x, y_dict, title='Chart', xlabel='X-Axis', ylabel='Y-Axis', yscale=(0, 1), filename=None):

 
    plt.figure(figsize=(12, 8))  
    
    # colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF',
    #       '#FFA500', '#800080', '#008000', '#000080', '#FFD700', '#FF1493']
    
    # colors = ['#E41A1C', '#4DAF4A', '#377EB8', '#984EA3', '#FF7F00',
    #       '#FFFF33', '#A65628', '#F781BF', '#999999', '#66C2A5', '#FC8D62']
     
    
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    #       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']
    
    
    # colors = ['#800000', '#808000', '#4363d8', '#000000', '#a9a9a9',
    #   '#e6194B', '#ffe119', '#f58231', '#911eb4', '#3cb44b', '#f032e6']
    
    colors = ['#e6194B', '#3cb44b', '#4363d8', '#000000', '#808000',
      '#f032e6', '#ffe119', '#f58231', '#911eb4']

    for i, (feature, y_values) in enumerate(y_dict.items()):
        plt.plot(x, y_values, marker='o', label=feature, color=colors[i])
    
    
    # for feature, y_values in y_dict.items():
    #     plt.plot(x, y_values, marker='o', label=feature)  
        # plt.plot(x, y_values, marker='o', label=feature, linestyle='-')


    plt.ylim(yscale[0], yscale[1])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(loc='best') 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"charts/{filename}.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
        
    
# def draw_charts_and_report(xchart, ychart, feature_importance_dict, report, yscale=(0, 1)):
def draw_charts_and_report(xchart, ychart, feature_importance_dict, report, filename=None):

    # Create a figure with two subplots side by side
    plt.figure(figsize=(15, 6))
    
    
    # plt.ylim(yscale[0], yscale[1])
    # First subplot for Avg AUC Pattern
    plt.subplot(1, 2, 1)
    plt.plot(xchart, ychart, marker='o')
    plt.title('Avg AUC')
    plt.xlabel('Configurations')
    plt.ylabel('Average AUC')
    plt.grid(True)
    
    
    # plt.ylim(yscale[0], yscale[1])
    
    # Second subplot for Feature Importance
    plt.subplot(1, 2, 2)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']

    for i, (feature, y_values) in enumerate(feature_importance_dict.items()):
        plt.plot(xchart, y_values, marker='o', label=feature, color=colors[i])
        
    # for feature, values in feature_importance_dict.items():
    #     plt.plot(xchart, values, marker='o', label=feature)
        
        
    plt.title('Feature Importance Over Configurations')
    plt.xlabel('Configurations')
    plt.ylabel('Feature Importance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plots
    if filename:
        plt.savefig(f"charts/{filename}.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    
    # Print report immediately after showing plots
    print("Report:", report)
    return report
   
   
def draw_charts_and_report_scale(xchart, ychart, feature_importance_dict, report, xlabel, yscale=(0, 1), filename=None):

    # Create a figure with two subplots side by side
    plt.figure(figsize=(15, 6))
    
    
    plt.ylim(yscale[0], yscale[1])
    # First subplot for Avg AUC Pattern
    plt.subplot(1, 2, 1)
    plt.plot(xchart, ychart, marker='o')
    plt.title('Avg AUC')
    # plt.xlabel('Configurations')
    plt.xlabel(xlabel)
    plt.ylabel('Average AUC Chart')
    plt.grid(True)
    
    
    plt.ylim(yscale[0], yscale[1])
    
    # Second subplot for Feature Importance
    plt.subplot(1, 2, 2)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']

    for i, (feature, y_values) in enumerate(feature_importance_dict.items()):
        plt.plot(xchart, y_values, marker='o', label=feature, color=colors[i])
    
    # for feature, values in feature_importance_dict.items():
    #     plt.plot(xchart, values, marker='o', label=feature)
        
        
    plt.title('Feature Importance Chart')
    # plt.xlabel('Configurations')
    plt.xlabel(xlabel)
    plt.ylabel('Feature Importance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plots
    if filename:
        plt.savefig(f"charts/{filename}.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    # Print report immediately after showing plots
    print("Report:", report)
    return report
        
#
# xchart = [20, 30, 40, 50, 60, 70, 80]
# ychart = [0.98, 0.97, 0.94, 0.93, 0.925, 0.92, 0.92]    
# draw_chart_func(xchart, ychart, chart_type='line', title='Avg AUC Pattern 1', xlabel='Configurations', ylabel='Average AUC')


# xchart = [20, 30, 40, 50, 60, 70, 80]
#
#
# feature_importance_dict = {
#     'day_of_month_workload': [0.1, 0.12, 0.14, 0.15, 0.13, 0.14, 0.16],
#     'overal_workload': [0.2, 0.18, 0.19, 0.22, 0.23, 0.21, 0.25],
#     'resource_type': [0.05, 0.06, 0.05, 0.04, 0.06, 0.07, 0.08],
#     'day_of_month': [0.1, 0.11, 0.12, 0.13, 0.12, 0.12, 0.11],
#     'month': [0.08, 0.09, 0.1, 0.11, 0.1, 0.09, 0.08],
#     'year': [0.05, 0.04, 0.03, 0.02, 0.03, 0.04, 0.05],
#     'day_of_week': [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22],
#     'is_peak_year_month': [0.12, 0.14, 0.16, 0.15, 0.13, 0.14, 0.12],
#     'is_peak_day_of_month': [0.1, 0.11, 0.12, 0.12, 0.11, 0.1, 0.09],
#     'is_peak_day_of_week': [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
# }
#
#
# draw_multiple_lines_chart(xchart, feature_importance_dict, title='Chart', xlabel='X-Axis', ylabel='Y-Axis')