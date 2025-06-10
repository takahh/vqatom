import cuml
from cuml.manifold import UMAP
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# データの読み込み（例: 手書き数字データ）
digits = load_digits()
X = digits.data
y = digits.target

# GPUベースのUMAP
umap_gpu = UMAP(n_neighbors=15, n_components=2, random_state=42)
X_embedded = umap_gpu.fit_transform(X)

# 結果をDataFrameにまとめる
df = pd.DataFrame(X_embedded, columns=['UMAP1', 'UMAP2'])
df['label'] = y

# 描画
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['UMAP1'], df['UMAP2'], c=df['label'], cmap='Spectral', s=10)
plt.colorbar(scatter, label='Digit Label')
plt.title('UMAP (GPU) Projection of Digits Dataset')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')

# 画像として保存
plt.savefig("umap_gpu_output.png", dpi=300)
plt.close()
