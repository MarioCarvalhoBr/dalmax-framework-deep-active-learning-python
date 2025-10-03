import numpy as np
import matplotlib.pyplot as plt
from .strategy import Strategy
from sklearn.cluster import KMeans

import torch
import numpy as np



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

class SSRAEKmeansHCSampling(Strategy):
    def __init__(self, dataset, net, logger):
        super(SSRAEKmeansHCSampling, self).__init__(dataset, net, logger)
        
        self.index = 0  # To keep track of the next sample to select
        self.flag = True
        self.features_2d = None

    def query(self, n):
        print(f"Initializing the DAL strategy with SSRAEKmeansHCSampling query {n} samples")
        
        
        features_dict = self.dataset.features_dict
        # Print shape of path_pkl
        print(f"Features dictionary contains {len(features_dict)} items.")
        print(f"Example feature vector shape: {next(iter(features_dict.values())).shape}")
    
        if self.flag:
            self.features_2d = get_features_2d(features_dict)
            self.flag = False

        print(f'Shape of features_2d: {self.features_2d.shape}')
        print(f'Item example of features_2d: {next(iter(self.features_2d))}')

        data = self.features_2d
        
        # Get image IDs and labels for coloring
        image_ids = list(features_dict.keys())
        labels = [self.dataset.Y_train[img_id] for img_id in image_ids]
        
        clusters = hkmg.hierarchical_kmeans_with_resampling(
            data=torch.tensor(data, device="cuda", dtype=torch.float32),
            n_clusters=[800, 756],
            n_levels=2,
            sample_sizes=[15, 2],
            verbose=False
        )

        cl = HierarchicalCluster.from_dict(clusters)
        sampled_indices = hs.hierarchical_sampling(cl, target_size=10)

        selected_samples = np.array(sampled_indices)
        print(f"\nSelected samples using SSRAE + K-means HierarchicalCluster: {selected_samples}")
        
        
        # Create single plot with overlapping circles and stars
        plt.figure(figsize=(14, 10))
        
        # Get unique classes and assign colors
        unique_classes = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
        color_map = {class_idx: colors[i] for i, class_idx in enumerate(unique_classes)}
        
        # Plot all data as circles colored by class
        for class_idx in unique_classes:
            mask = np.array(labels) == class_idx
            class_name = self.dataset.get_class_name(class_idx)
            plt.scatter(data[mask, 0], data[mask, 1], 
                       c=[color_map[class_idx]], 
                       label=f'{class_name}', 
                       s=30, alpha=0.5, marker='o')
        
        # Plot sampled data as stars on top, colored by their class
        sampled_classes_plotted = set()
        for idx in sampled_indices:
            label = labels[idx]
            class_name = self.dataset.get_class_name(label)
            
            # Add label only once per class for sampled items
            if label not in sampled_classes_plotted:
                plt.scatter(data[idx, 0], data[idx, 1], 
                           c=[color_map[label]], 
                           s=200, alpha=1.0, marker='*',
                           edgecolors='black', linewidths=1.5,
                           label=f'{class_name} (sampled)')
                sampled_classes_plotted.add(label)
            else:
                plt.scatter(data[idx, 0], data[idx, 1], 
                           c=[color_map[label]], 
                           s=200, alpha=1.0, marker='*',
                           edgecolors='black', linewidths=1.5)
        
        plt.title(f"t-SNE Visualization - Round {self.index}\nCircles: All Data | Stars: Sampled Data", fontsize=14)
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'results/{self.index}_ssl_features_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Increment index for next round
        self.index += 1

        # Remove selected_samples do features_dict
        for img_id in selected_samples:
            if img_id in features_dict:
                del features_dict[img_id]
                
        # Remove selected_samples do self.features_2d
        self.features_2d = np.delete(self.features_2d, selected_samples, axis=0)
                
        return selected_samples
    

    def query1(self, n):
        print(f"Initializing the DAL strategy with SSRAEKmeansHCSampling query {n} samples")
        
        
        features_dict = self.dataset.features_dict
        # Print shape of path_pkl
        print(f"Features dictionary contains {len(features_dict)} items.")
        print(f"Example feature vector shape: {next(iter(features_dict.values())).shape}")
    
        data = np.vstack([features_dict[key].numpy() for key in features_dict])
            
        
        clusters = hkmg.hierarchical_kmeans_with_resampling(
            data=torch.tensor(data, device="cuda", dtype=torch.float32),
            n_clusters=[800, 756],
            n_levels=2,
            sample_sizes=[15, 2],
            verbose=False
        )

        cl = HierarchicalCluster.from_dict(clusters)
        sampled_indices = hs.hierarchical_sampling(cl, target_size=10)

        selected_samples = np.array(sampled_indices)
        print(f"\nSelected samples using SSRAE + K-means HierarchicalCluster: {selected_samples}")
        

        # Remove selected_samples do features_dict
        for img_id in selected_samples:
            if img_id in features_dict:
                del features_dict[img_id]
                
        return selected_samples
                
