import matplotlib.pyplot as plt
import numpy as np


def main(probabilities):
        
    # Pattern names
    pattern_names = [
        'Different system terminologies',
        'Busy month',
        'Busy day of week',
        'Busy day of month',
        'Busy user/role',
        'Different human conventions',
        'Busy day of month for a user'
    ]

    # Plotting
    plt.figure(figsize=(12, 6))
    colors = ['#e6194B', '#3cb44b', '#4363d8', '#000000', '#808000', '#f032e6', '#ffe119']

    x = np.arange(len(pattern_names))

    plt.plot(x, probabilities, marker='o', color='#4363d8', label='Probability Score')
    for i, (x_val, y_val) in enumerate(zip(x, probabilities)):
        # plt.text(x_val, y_val + 0.01, f"{y_val:.3f}", ha='center', va='bottom', fontsize=10)
        plt.text(x_val, y_val + 0.01, f"{y_val * 100:.2f}%", ha='center', va='bottom', fontsize=10)


    plt.xticks(x, pattern_names, rotation=30, ha='right')
    plt.title('Probability for Different Root Causes')
    plt.xlabel('Root Cause')
    plt.ylabel('Probability')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend()
    plt.show()


probabilities = np.array([9.50515114e-01, 1.17347183e-04, 1.16921634e-04, 1.16487027e-04,  3.13695859e-04, 4.86786270e-02, 1.41807515e-04])
main(probabilities)