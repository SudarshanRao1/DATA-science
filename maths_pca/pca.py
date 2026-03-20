import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('TCGA_LAML_PCA_READY.csv')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio per component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
print(f"  Total Variance Explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
           facecolors='none', edgecolors='steelblue', s=50, linewidths=1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA - 3D Projection')

plt.tight_layout()
plt.savefig('pca_sklearn_3d.png', dpi=150)
plt.show()

