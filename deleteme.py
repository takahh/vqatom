from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load sample dataset
digits = load_digits()
X = digits.data
y = digits.target

# Different perplexities
perplexities = [5, 30, 50]

plt.figure(figsize=(15, 4))
for i, perp in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=0)
    X_tsne = tsne.fit_transform(X)

    plt.subplot(1, 3, i + 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10)
    plt.title(f"Perplexity = {perp}")
plt.tight_layout()
plt.show()
