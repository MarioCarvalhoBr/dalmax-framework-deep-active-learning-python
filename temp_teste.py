from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'high_dimensional_data' is your input data
# Example: Generate some dummy high-dimensional data
#old dummy: high_dimensional_data = np.random.rand(100, 50) 

# Gere o high_dimensional_data com 3 diferentes grupos de aneis de pontos
group1 = np.random.randn(33, 50) + np.array([5]*50)  # Grupo 1 centrado em (5,5,...,5)
group2 = np.random.randn(33, 50) + np.array([-5]*50) # Grupo 2 centrado em (-5,-5,...,-5)
group3 = np.random.randn(34, 50) + np.array([0]*50)  # Grupo 3 centrado em (0,0,...,0)
high_dimensional_data = np.vstack([group1, group2, group3])

# Initialize t-SNE with 3 components
tsne = TSNE(n_components=3, random_state=42) 

# Fit and transform the data
tsne_3d_embedding = tsne.fit_transform(high_dimensional_data)

# Define vibrant color palette for selected items
vibrant_color_palette = ['black', 'yellow', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 
                         'magenta', 'lime', 'olive', 'navy', 'teal', 'maroon', 'gold', 'silver', 'coral']

# Define markers for sampled data (one marker per class)
markers = ['o', 's', '^', '*', 'v', '<', '>', 'p', 'D', 'h', 'H', '+', 'x', 'd', '|', '_']

# Create fake labels for demonstration (5 classes)
num_classes = 3
labels = np.random.randint(0, num_classes, size=100)

# Create a 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each class with different color and marker
for class_idx in range(num_classes):
    mask = labels == class_idx
    color = vibrant_color_palette[class_idx % len(vibrant_color_palette)]
    marker = markers[class_idx % len(markers)]
    
    ax.scatter(tsne_3d_embedding[mask, 0], 
               tsne_3d_embedding[mask, 1], 
               tsne_3d_embedding[mask, 2],
               c=color,
               marker=marker,
               s=100,
               alpha=0.8,
               edgecolors='black',
               linewidths=0.5,
               label=f'Class {class_idx}')

# Add labels and title
ax.set_xlabel('t-SNE Component 1', fontsize=12)
ax.set_ylabel('t-SNE Component 2', fontsize=12)
ax.set_zlabel('t-SNE Component 3', fontsize=12)
ax.set_title('3D t-SNE Visualization', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()