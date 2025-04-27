

import numpy as np




# Overall Relative Frequency
def calculate_orf(frequencies):
    total = sum(frequencies)
    return [freq / total for freq in frequencies]

# Unique Label Count, 
def calculate_label_variance(frequencies):
    mean = np.mean(frequencies)
    std_dev = np.std(frequencies)
    print(f"std_dev{std_dev}")
    cv = std_dev / mean
    # 0 for low variance (CV < 1) and 1 for high variance (CV >= 1)
    variance_metric = 0 if cv < 1 else 1
    return len(frequencies), cv, variance_metric

def calculate_diversity(labels, frequencies, human_labels, system_labels):
    dcp_human = len([label for label in labels if label in human_labels]) / len(labels)
    dcp_system = len([label for label in labels if label in system_labels]) / len(labels)
    
    di = 1 - sum([freq ** 2 for freq in frequencies])
    
    return dcp_human, dcp_system, di

def calculate_label_variance_group(labels, frequencies, group_labels):
    group_frequencies = [frequencies[i] for i, label in enumerate(labels) if label in group_labels]
    ulc = len(group_frequencies)  
    
    if len(group_frequencies) > 0:
        mean = np.mean(group_frequencies)
        std_dev = np.std(group_frequencies)
        cv = std_dev / mean
        variance_metric = 0 if cv < 1 else 1
    else:
        mean, std_dev, cv, variance_metric = 0, 0, 0, 0 

    return ulc, cv, variance_metric

def calculate_label_diversity_ratio(human_labels, system_labels, labels):
    unique_human_labels = set(human_labels) & set(labels)  
    unique_system_labels = set(system_labels) & set(labels) 
    
    total_unique_labels = len(unique_human_labels) + len(unique_system_labels)
    if total_unique_labels == 0:
        return 0  
    
    ldr = len(unique_system_labels) / total_unique_labels
    ldr2 = len(unique_human_labels) / total_unique_labels

    return ldr, ldr2


def determine_patterns(dcp_human, dcp_system, ulc, cv):
    if dcp_human > dcp_system:
        pattern = "Pattern 5: Human-Dominant Labels"
    elif dcp_system > dcp_human:
        pattern = "Pattern 1: System-Dominant Labels"
    else:
        pattern = "Neutral Label Usage"
    
    if ulc > 10 and cv >= 1:
        typo_pattern = "Typo or Multiple Terminologies"
    else:
        typo_pattern = "Standard Label Usage"
    
    return pattern, typo_pattern

def main(labels, frequencies, human_labels, system_labels):

    orf = calculate_orf(frequencies)
    ulc, cv, variance_metric = calculate_label_variance(frequencies)
    ulc_human, cv_human, variance_metric_human = calculate_label_variance_group(labels, frequencies, human_labels)
    ulc_system, cv_system, variance_metric_system = calculate_label_variance_group(labels, frequencies, system_labels)
    dcp_human, dcp_system, di = calculate_diversity(labels, frequencies, human_labels, system_labels)
    ldr_system , ldr_human = calculate_label_diversity_ratio(human_labels, system_labels, labels)
    pattern, typo_pattern = determine_patterns(dcp_human, dcp_system, ulc, cv)

    print("Metrics:")
    print(f"1. Overall Relative Frequency (ORF): {orf}")
    print(f"2. Unique Label Count (ULC): {ulc}")
    print(f"3. Coefficient of Variation (CV): {cv:.2f} (Variance Metric: {variance_metric})")
    print(f"2. Human Label Variance (ULC, CV): {ulc_human}, {cv_human:.2f} (Variance Metric: {variance_metric_human})")
    print(f"3. System Label Variance (ULC, CV): {ulc_system}, {cv_system:.2f} (Variance Metric: {variance_metric_system})")


    print(f"4. Diversity by Category Proportion (DCP) - Human: {dcp_human:.2f}, System: {dcp_system:.2f}")
    print(f"5. Diversity Index (DI): {di:.2f}")
    print(f"1. Label Diversity Ratio (LDR_System): {ldr_system:.2f}, 1. Label Diversity Ratio (LDR_Human): {ldr_human:.2f}")
    print("Patterns:")
    print(f"   - Primary Pattern: {pattern}")
    print(f"   - Typo Pattern: {typo_pattern}")
    
    return variance_metric
