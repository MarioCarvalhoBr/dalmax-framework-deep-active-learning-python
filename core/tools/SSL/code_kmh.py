import torch
import numpy as np
import time 

import os
import pickle

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from core.tools.SSL.src.clusters import HierarchicalCluster
from core.tools.SSL.src import (
  hierarchical_kmeans_gpu as hkmg,
  hierarchical_sampling as hs
)


def get_features_2d(features_dict):
        """Use t-SNE to convert each image features to 2D and plot with class colors"""
        
        if not features_dict:
            print("No features dictionary found. Skipping t-SNE plot.")
            return
        
        print("Creating t-SNE visualization of features...")
        
        # Get image IDs and convert features to matrix
        image_ids = list(features_dict.keys())
        features_matrix = np.vstack([features_dict[img_id] for img_id in image_ids])
        
        
        print(f"Running t-SNE on {features_matrix.shape[0]} samples with {features_matrix.shape[1]} features...")
        
        # Apply t-SNE
        perplexity = min(30, len(image_ids) - 1)  # Ensure perplexity is valid
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', 
                   perplexity=perplexity, random_state=42)
        features_2d = tsne.fit_transform(features_matrix)
        
        return features_2d



# Verifica se o arquivo results/features_dict_ssrae.pkl existe. Se sim, carrega e adiciona a features_dict
features_dict = {}
path_pkl = 'results/features_dict_ssrae.pkl'
if os.path.exists(path_pkl):
    with open(path_pkl, 'rb') as f:
        features_dict = pickle.load(f)
    print(f"Features dictionary loaded from {path_pkl}")
    # Print shape of path_pkl
    print(f"Features dictionary contains {len(features_dict)} items.")
    print(f"Example feature vector shape: {next(iter(features_dict.values())).shape}")

else:
    print(f"File {path_pkl} does not exist. Proceeding with synthetic data.")
    exit()


# features_dict é um dicionário com 7986 items e cada item é um torch.Size([756])
# Crie o novo_features sendo [7986, 756]
novo_features = np.vstack([features_dict[key].numpy() for key in features_dict])
print(novo_features.shape)



features_2d = get_features_2d(features_dict)
print(features_2d.shape)
print(next(iter(features_2d)))

data = features_2d

data = novo_features


time_start = time.time()

clusters = hkmg.hierarchical_kmeans_with_resampling(
  data=torch.tensor(data, device="cuda", dtype=torch.float32),
  n_clusters=[800, 756],
  n_levels=2,
  sample_sizes=[15, 2],
  verbose=False,
)

cl = HierarchicalCluster.from_dict(clusters)
sampled_indices = hs.hierarchical_sampling(cl, target_size=10)
print(f'sampled_indices: {sampled_indices}')

path_pkl = 'results/Y_train.pkl'
Y_train = []
if os.path.exists(path_pkl):
    with open(path_pkl, 'rb') as f:
        Y_train = pickle.load(f)
    print(f"Features dictionary loaded from {path_pkl}")
else:
    print(f"File {path_pkl} does not exist. Proceeding without Y_train.")
    exit()
    
print(f'Y_train sampled_indices: {Y_train}')


# Show matplotlib figure
import matplotlib.pyplot as plt

# Original data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
plt.title("Original Data")
plt.axis('equal')   
plt.grid(True)
# Sampled data
plt.subplot(1, 2, 2)
plt.scatter(data[sampled_indices, 0], data[sampled_indices, 1], s=10, color='red', alpha=0.7)
plt.title("Sampled Data")
plt.axis('equal')
plt.grid(True)
# Save figure pdf
# Save the plot
plt.savefig('ssl_features_visualization.png', dpi=300, bbox_inches='tight')
print("t-SNE visualization saved as 'ssl_features_visualization.png'")
plt.show()

print(f"Time taken: {time.time() - time_start:.2f} seconds")