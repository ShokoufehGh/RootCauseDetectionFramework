

from lxml import etree
from collections import defaultdict
import csv
from typing import Dict, Set, Tuple

def analyze_activity_users(xes_file: str, namespace: str) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:

    activity_users_map = defaultdict(set)
    user_type_counts = {"NONE": 0, "user": 0, "batch": 0}
    total_users = 0
    
    print("Processing the XES file... This may take some time.")
    context = etree.iterparse(xes_file, events=("start", "end"))
    
    # Iterate over events in the XES file
    for event, elem in context:
        if event == "end" and elem.tag == f"{namespace}event":
            activity = None
            resource = None
            
            # Extract activity and resource information
            for attribute in elem:
                if attribute.tag == f"{namespace}string":
                    if attribute.attrib.get('key') == "concept:name":
                        activity = attribute.attrib.get('value')
                    elif attribute.attrib.get('key') == "org:resource":
                        resource = attribute.attrib.get('value')
            
            # Update mappings if both activity and resource are present
            if activity and resource:
                activity_users_map[activity].add(resource)
                
                # Update user type counts
                total_users += 1
                if resource.lower().startswith("user"):
                    user_type_counts["user"] += 1
                elif resource.lower().startswith("batch"):
                    user_type_counts["batch"] += 1
                else:
                    user_type_counts["NONE"] += 1
            
            # Clear element to free memory
            elem.clear()
    
    return activity_users_map, user_type_counts

def save_activity_user_analysis(activity_users_map: Dict[str, Set[str]], 
                              user_type_counts: Dict[str, int], 
                              output_base_filename: str):
    """
    Saves the analysis results to CSV files.
    
    Args:
        activity_users_map: Dictionary mapping activities to sets of users
        user_type_counts: Dictionary containing user type counts
        output_base_filename: Base filename for output files
    """
    # Save detailed activity-user mapping
    detailed_output = f"{output_base_filename}_detailed.csv"
    with open(detailed_output, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Activity", "User", "User Type"])
        
        for activity, users in sorted(activity_users_map.items()):
            for user in sorted(users):
                user_type = "user" if user.lower().startswith("user") else \
                           "batch" if user.lower().startswith("batch") else "NONE"
                writer.writerow([activity, user, user_type])
    
    # Save summary statistics
    summary_output = f"{output_base_filename}_summary.csv"
    with open(summary_output, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Activity", "Total Users", "NONE Users", "Human Users", "Batch Users"])
        
        for activity, users in sorted(activity_users_map.items()):
            none_users = sum(1 for u in users if not (u.lower().startswith("user") or u.lower().startswith("batch")))
            human_users = sum(1 for u in users if u.lower().startswith("user"))
            batch_users = sum(1 for u in users if u.lower().startswith("batch"))
            
            writer.writerow([activity, len(users), none_users, human_users, batch_users])

# Example usage
def main():
    xes_file = 'data/disco_export_BPIC2019.xes' 
    namespace = "{http://www.xes-standard.org/}"
    
    # Analyze the XES file
    activity_users_map, user_type_counts = analyze_activity_users(xes_file, namespace)
    
    # Print summary statistics
    total_users = sum(user_type_counts.values())
    if total_users > 0:
        print("\nUser Type Percentages:")
        for user_type, count in user_type_counts.items():
            percentage = (count / total_users) * 100
            print(f"{user_type}: {percentage:.2f}%")
    
    # Print activity statistics
    print("\nActivity Statistics:")
    for activity, users in sorted(activity_users_map.items()):
        none_users = sum(1 for u in users if not (u.lower().startswith("user") or u.lower().startswith("batch")))
        human_users = sum(1 for u in users if u.lower().startswith("user"))
        batch_users = sum(1 for u in users if u.lower().startswith("batch"))
        
        print(f"\nActivity: {activity}")
        print(f"Total unique users: {len(users)}")
        print(f"- NONE users: {none_users}")
        print(f"- Human users: {human_users}")
        print(f"- Batch users: {batch_users}")
    
    # Save results to CSV files
    save_activity_user_analysis(activity_users_map, user_type_counts, "activity_users_analysis")
    print("\nResults have been saved to 'activity_users_analysis_detailed.csv' and 'activity_users_analysis_summary.csv'")

if __name__ == "__main__":
    main()