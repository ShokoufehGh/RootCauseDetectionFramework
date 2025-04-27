from itertools import combinations
import numpy as np
from typing import List, Set, Tuple, Dict
from collections import Counter

def get_canonical_term(group: set) -> str:
    cleaned_terms = {term.rstrip('- ').strip() for term in group}
    for term in group:
        cleaned = term.rstrip('- ').strip()
        if list(group).count(term) > 1:
            return term
    term_scores = {}
    for term in cleaned_terms:
        score = 0
        if '-' in term or term.endswith(' '):
            score -= 1
        if all(word[0].isupper() for word in term.split()):
            score += 2
        base_term = ''.join(c.lower() for c in term if c.isalnum())
        similar_count = sum(1 for t in group if ''.join(c.lower() for c in t if c.isalnum()) == base_term)
        score += similar_count
        term_scores[term] = score
    return max(term_scores.items(), key=lambda x: x[1])[0]

def analyze_synonym_set(synonyms: List[str], size_threshold: int = 10) -> Dict:
    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def normalized_distance(s1: str, s2: str) -> float:
        distance = levenshtein_distance(s1, s2)
        max_length = max(len(s1), len(s2))
        return distance / max_length if max_length > 0 else 0

    def extract_mutable_part(label: str) -> str:
        parts = label.split()
        return parts[-1] if len(parts) > 1 else label

    unique_terms = set(synonyms)
    set_size = len(unique_terms)
    pairs = list(combinations(unique_terms, 2))
    distances = [(s1, s2, normalized_distance(s1, s2)) for s1, s2 in pairs]
    distance_values = [d for _, _, d in distances]
    avg_distance = np.mean(distance_values) if distance_values else 0
    std_distance = np.std(distance_values) if distance_values else 0
    edit_similarity = 1 - avg_distance

    # Mutable Text Edit Distance
    mutable_parts = [extract_mutable_part(label) for label in synonyms]
    mutable_pairs = list(combinations(set(mutable_parts), 2))
    mutable_distances = [normalized_distance(s1, s2) for s1, s2 in mutable_pairs]
    mutable_text_edit_distance = np.mean(mutable_distances) if mutable_distances else 0

    distance_threshold = 0.2
    typo_groups = []
    processed = set()

    for s1 in unique_terms:
        if s1 in processed:
            continue
        group = {s1}
        for s2 in unique_terms:
            if s1 != s2 and normalized_distance(s1, s2) < distance_threshold:
                group.add(s2)
        if len(group) > 1:
            typo_groups.append(group)
            processed.update(group)

    different_terms = set()
    different_terms.update(set(unique_terms) - set().union(*typo_groups) if typo_groups else set(unique_terms))
    for group in typo_groups:
        canonical_term = get_canonical_term(group)
        different_terms.add(canonical_term)

    is_likely_typo_set = set_size >= size_threshold
    has_single_terminology = len(different_terms) == 1

    return {
        "set_size": set_size,
        "average_distance": avg_distance,
        "edit_similarity": edit_similarity,
        "mutable_text_edit_distance": mutable_text_edit_distance,
        "std_distance": std_distance,
        "detailed_distances": distances,
        "likely_typo_groups": typo_groups,
        "likely_different_terms": different_terms,
        "is_likely_typo_set": is_likely_typo_set,
        "has_single_terminology": has_single_terminology,
        "single_term": next(iter(different_terms)) if has_single_terminology else None
    }

def print_analysis(synonyms: List[str], description: str = "", size_threshold: int = 10):
    print(f"\nAnalyzing set: {description}")
    print("-" * 50)
    results = analyze_synonym_set(synonyms, size_threshold)
    print(f"Number of unique terms: {results['set_size']}")
    print(f"Average normalized distance: {results['average_distance']:.3f}")
    print(f"Edit similarity: {results['edit_similarity']:.3f}")
    print(f"Mutable text edit distance: {results['mutable_text_edit_distance']:.3f}")
    print(f"Standard deviation: {results['std_distance']:.3f}")
    print(f"\nSet classification: {'Likely typos' if results['is_likely_typo_set'] else 'Likely different terminology'}")
    print(f"(Based on set size threshold of {size_threshold})")

    if results['likely_typo_groups']:
        print("\nLikely typo groups (based on edit distance):")
        for group in results['likely_typo_groups']:
            print(f"  {group}")

    if results['has_single_terminology']:
        print("\nSingle terminology detected:")
        print(f"  {results['single_term']}")
    else:
        if results['likely_different_terms']:
            print("\nDifferent terminology (including canonical forms from typo groups):")
            for term in sorted(results['likely_different_terms']):
                print(f"  {term}")
                
    return results
                
        

if __name__ == "__main__":
    # new_set = ["Cancel Invoice Receipt", "Cancel Invoice Receit", "Cancel Invoice Reciept", "Cancel Invoise Receipt"]
    # new_set  = ["Refer to progress work attached.", "Refer to progress task in attached email.", "Refer to progress update in the attached email."]
    # new_set  = ["Record Invoice Receipt", "Record Invoice Receit", "Record Invoice Reciept", "Record Invoise Receipt"]
    
    # new_set  = ["Accepted + Wait - User", "Approval Pending – Staff", "Request Approved by Agent", "Confirmed Later User"]
    # new_set  = ["Winton Hospital", "QH WINTON HOSPITAL"]
    new_set  = ["Princess Alexandra Hospital", "QH PRINCESS ALEXANDRA HOSPITAL"]
    # new_set  = ["work" , "task in email" , "update the"]
    # new_set = [
    # "Request from management. See attached response",
    # "Request from Management. See attached email",
    # "Request from Management. See attached answer"]
    
    print_analysis(new_set, "Different Terminology Set", size_threshold=8)
