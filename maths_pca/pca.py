import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/SUDARSHAN/OneDrive/Documents/COLLAGE/SEMISTER-2/MATHS/maths_pca/TCGA_LAML_PCA_READY.csv')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio per component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"   PC{i+1}: {var:.4f} ({var*100:.2f}%)")

total_var_3pc = sum(pca.explained_variance_ratio_[:3]) * 100
print(f"   Total Variance Explained (First 3 PCs): {total_var_3pc:.2f}%")
# variance graph
plt.figure(figsize=(10, 5))
components = range(1, len(pca.explained_variance_ratio_) + 1)
plt.bar(components, pca.explained_variance_ratio_, alpha=0.6, color='steelblue', label='Individual Variance')
plt.step(components, np.cumsum(pca.explained_variance_ratio_), where='mid', color='red', label='Cumulative Variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot: Variance Explained by Components')
plt.xticks(components)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
# 2D projection
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], facecolors='none', edgecolors='steelblue', s=50, alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - 2D Projection')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
# 3D projection
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
           facecolors='none', edgecolors='steelblue', s=50, linewidths=1)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA - 3D Projection')
plt.tight_layout()
plt.show()
