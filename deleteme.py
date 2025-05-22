from collections import Counter
import logging

# Sample nested list
ind_list = [
    [0.0, 1.0, 2.0],
    [2.0, 3.0, 0.0],
    [1.0, 4.0, 0.0]
]

# Flatten the list
flat = [item for sublist in ind_list for item in sublist]
count = Counter(flat)

# Sort by key
sorted_count = dict(sorted(count.items()))
print("sorted_count")
# Print only first 100 items
print(dict(list(sorted_count.items())[:100]))

# Count values
num_zero = count[0.0] if 0.0 in count else 0
num_unique_nonzero = len([k for k in count if k != 0.0])

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

logger.info(f"\nNumber of zero values: {num_zero}")
logger.info(f"Number of nonzero unique values: {num_unique_nonzero}")
