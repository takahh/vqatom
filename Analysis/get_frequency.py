import json
import matplotlib.pyplot as plt

# Load your data
with open("/Users/taka/Documents/final_infer_on_all_data/40000_16/ind_frequencies.json", "r") as f:
    data = json.load(f)

# Convert keys to float and get (key, freq) pairs
items = [(float(k), v) for k, v in data.items()]

# Sort by frequency (value)
items_sorted = sorted(items, key=lambda x: x[1], reverse=True)  # reverse=True for descending

# Unpack the sorted items
sorted_keys, sorted_freqs = zip(*items_sorted)
plt.rcParams.update({'font.size': 20})  # ← change 14 to whatever size you want

plt.figure(figsize=(12, 6))
plt.hist(sorted_freqs, bins=100, color='green', edgecolor='black', log=True)
plt.xlabel("Frequency")
plt.ylabel("Number of Codebook Vectors (log scale)")
plt.title("Histogram of Frequencies")
plt.tight_layout()
plt.savefig("/Users/taka/Documents/freq_per_cb.png")
plt.show()
