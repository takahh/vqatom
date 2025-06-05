import matplotlib.pyplot as plt
import os
import re

# Original list of experiments
exp_list = [
    '20000_8', '20000_16', '20000_32', '25000_8', '25000_16', '25000_32',
    '30000_8', '30000_16', '30000_32', '35000_8', '35000_16', '35000_32',
    '40000_8', '40000_16', '40000_32', '45000_8', '45000_16', '45000_32'
]

# Function to extract data from the log
def get_cbmax_from_log(pair_name):
    effective_cb_size_list = []
    cb_num_mean_list = []
    effective_cb_rate_list = []

    filepath = f'/Users/taka/Documents/vqatom_train_output/bothloss_{pair_name}/outputs/log'
    fallback_path = f'/Users/taka/Documents/vqatom_train_output/bothloss_{pair_name}/0227version/outputs/log'

    # Special case
    if pair_name == '30000_32':
        filepath = f'/Users/taka/Documents/vqatom_train_output/log_bothloss_{pair_name}'

    try:
        if os.path.getsize(filepath) < 1024:
            filepath = fallback_path
    except FileNotFoundError:
        filepath = fallback_path

    with open(filepath, 'r') as file:
        for line in file:
            if "observed" in line:
                value = float(line.split(" ")[-1].strip())
                fraction = value / float(pair_name.split('_')[0])
                effective_cb_size_list.append(value)
                effective_cb_rate_list.append(fraction)
            elif "unique_cb_vecs mean" in line:
                value = float(line.split()[10].split(',')[0])
                cb_num_mean_list.append(value)

    max_eff_cb = max(effective_cb_size_list)
    idx_eff_cb = effective_cb_size_list.index(max_eff_cb)
    max_cb_mean = cb_num_mean_list[idx_eff_cb]
    max_cb_rate = effective_cb_rate_list[idx_eff_cb]

    print(f"{pair_name} -> best epoch: {idx_eff_cb}")
    return max_eff_cb, max_cb_mean, max_cb_rate

# Plotting function
def plot(data, name, sorted_exp_list):
    plt.figure(figsize=(12, 6), dpi=200)
    plt.plot(range(len(data)), data, marker='o', label=name)

    plt.title(name, fontsize=20, pad=20)  # Title with more padding
    plt.xticks(ticks=range(len(data)), labels=sorted_exp_list, rotation=90, fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Codebook Size_Dimension", fontsize=18, labelpad=20)  # Bigger + padded
    plt.ylabel("Value", fontsize=18, labelpad=20)

    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend(fontsize=14)
    plt.show()


# Sorting helper
def sort_key(exp):
    match = re.match(r"(\d+)_([0-9]+)", exp)
    return (int(match.group(1)), int(match.group(2)))

def main():
    results = {}

    for exp in exp_list:
        eff_cb, cb_mean, eff_cb_rate = get_cbmax_from_log(exp)
        results[exp] = (eff_cb, cb_mean, eff_cb_rate)

    # Sort by (codebook size, dim)
    sorted_exps = sorted(results.keys(), key=sort_key)

    # Extract sorted metrics
    eff_cb_data = [results[exp][0] for exp in sorted_exps]
    uni_cb_means = [results[exp][1] for exp in sorted_exps]
    eff_cb_rates = [results[exp][2] for exp in sorted_exps]

    # Plot
    plot(eff_cb_data, "Effective CB Max", sorted_exps)
    plot(eff_cb_rates, "Effective CB Rates", sorted_exps)
    plot(uni_cb_means, "Unique CB Count Avg", sorted_exps)

if __name__ == '__main__':
    main()
