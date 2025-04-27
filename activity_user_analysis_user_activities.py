

from lxml import etree
from collections import defaultdict
import csv
from typing import Dict, Set, Tuple

def analyze_activity_users(xes_file: str, namespace: str) -> Tuple[Dict[str, Set[str]], Dict[str, int], Dict[str, Set[str]]]:
    """
    Analyzes an XES file to get user lists per activity, user type counts, and activities per user.
    
    Args:
        xes_file: Path to the XES file
        namespace: XES namespace
        
    Returns:
        Tuple containing:
        - Dictionary mapping activities to sets of users
        - Dictionary containing user type counts
        - Dictionary mapping users to their activities
    """
    # Initialize dictionaries
    activity_users_map = defaultdict(set)
    user_type_counts = {"NONE": 0, "user": 0, "batch": 0}
    user_activities_map = defaultdict(set)  # New: Track activities per user
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
                user_activities_map[resource].add(activity)  # New: Track activities per user
                
                # Update user type counts (only count each user once)
                if resource not in user_activities_map:
                    total_users += 1
                    if resource.lower().startswith("user"):
                        user_type_counts["user"] += 1
                    elif resource.lower().startswith("batch"):
                        user_type_counts["batch"] += 1
                    else:
                        user_type_counts["NONE"] += 1
            
            # Clear element to free memory
            elem.clear()
    
    return activity_users_map, user_type_counts, user_activities_map

def save_activity_user_analysis(activity_users_map: Dict[str, Set[str]], 
                              user_type_counts: Dict[str, int],
                              user_activities_map: Dict[str, Set[str]],
                              output_base_filename: str):
    """
    Saves the analysis results to CSV files.
    
    Args:
        activity_users_map: Dictionary mapping activities to sets of users
        user_type_counts: Dictionary containing user type counts
        user_activities_map: Dictionary mapping users to their activities
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
    
    # New: Save user activity statistics
    user_stats_output = f"{output_base_filename}_user_activities.csv"
    with open(user_stats_output, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["User", "User Type", "Number of Different Activities", "Activities"])
        
        for user, activities in sorted(user_activities_map.items()):
            user_type = "user" if user.lower().startswith("user") else \
                       "batch" if user.lower().startswith("batch") else "NONE"
            writer.writerow([user, user_type, len(activities), "|".join(sorted(activities))])

# Example usage
def main():
    xes_file = 'data/disco_export_BPIC2019.xes' 
    namespace = "{http://www.xes-standard.org/}"
    
    # Analyze the XES file
    activity_users_map, user_type_counts, user_activities_map = analyze_activity_users(xes_file, namespace)
    
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
    
    # New: Print user activity statistics
    print("\nUser Activity Statistics:")
    user_activity_counts = defaultdict(list)
    for user, activities in user_activities_map.items():
        user_type = "user" if user.lower().startswith("user") else \
                   "batch" if user.lower().startswith("batch") else "NONE"
        user_activity_counts[user_type].append(len(activities))
    
    for user_type, counts in user_activity_counts.items():
        avg_activities = sum(counts) / len(counts) if counts else 0
        max_activities = max(counts) if counts else 0
        min_activities = min(counts) if counts else 0
        print(f"\n{user_type} users:")
        print(f"- Average different activities per user: {avg_activities:.2f}")
        print(f"- Maximum different activities: {max_activities}")
        print(f"- Minimum different activities: {min_activities}")
    
    # Save results to CSV files
    save_activity_user_analysis(activity_users_map, user_type_counts, user_activities_map, "activity_users_analysis")
    print("\nResults have been saved to:")
    print("- 'activity_users_analysis_detailed.csv'")
    print("- 'activity_users_analysis_summary.csv'")
    print("- 'activity_users_analysis_user_activities.csv'")

if __name__ == "__main__":
    main()