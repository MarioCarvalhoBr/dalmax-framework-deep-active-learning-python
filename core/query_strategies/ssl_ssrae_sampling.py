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
            # Create and store a stable ordering of image ids that aligns with features_2d rows
            self.image_ids = list(features_dict.keys())
            self.features_2d = get_features_2d({img_id: features_dict[img_id] for img_id in self.image_ids})
            self.flag = False

        print(f'Shape of features_2d: {self.features_2d.shape}')
        print(f'Item example of features_2d: {next(iter(self.features_2d))}')

        data = self.features_2d

        # Use the stored image id ordering (rows of features_2d correspond to these ids)
        if not hasattr(self, 'image_ids'):
            # Fallback: derive from current features_dict (best-effort)
            self.image_ids = list(features_dict.keys())

        labels = [self.dataset.Y_train[img_id] for img_id in self.image_ids]
        
        clusters = hkmg.hierarchical_kmeans_with_resampling(
            data=torch.tensor(data, device="cuda", dtype=torch.float32),
            n_clusters=[800, 756],
            n_levels=2,
            sample_sizes=[15, 2],
            verbose=False
        )

        cl = HierarchicalCluster.from_dict(clusters)
        sampled_indices = hs.hierarchical_sampling(cl, target_size=n)

        selected_samples = np.array(sampled_indices)
        print(f"\nSelected samples using SSRAE + K-means HierarchicalCluster: {selected_samples}")
        
        
        # Create single plot with overlapping circles and stars
        plt.figure(figsize=(14, 10))
        
        # Get unique classes and assign colors
        unique_classes = np.unique(labels)
        
        # Define soft color palette for non-selected items (original palette)
        soft_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
        
        # Define vibrant color palette for selected items
        vibrant_color_palette = ['black', 'yellow', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 
                         'magenta', 'lime', 'olive', 'navy', 'teal', 'maroon', 'gold', 'silver', 'coral']
        
        # Define markers for sampled data (one marker per class)
        markers = ['o', 's', '^', '*', 'v', '<', '>', 'p', 'D', 'h', 'H', '+', 'x', 'd', '|', '_']
        
        # Create color maps: soft colors for all data, vibrant for selected
        soft_color_map = {class_idx: soft_colors[i] for i, class_idx in enumerate(unique_classes)}
        vibrant_color_map = {class_idx: vibrant_color_palette[i % len(vibrant_color_palette)] for i, class_idx in enumerate(unique_classes)}
        marker_map = {class_idx: markers[i % len(markers)] for i, class_idx in enumerate(unique_classes)}
        
        # Plot all data as circles colored by class (soft colors)
        for class_idx in unique_classes:
            mask = np.array(labels) == class_idx
            class_name = self.dataset.get_class_name(class_idx)
            # Ensure mask length matches data length
            if mask.shape[0] != data.shape[0]:
                # If lengths mismatch, try to trim or pad mask to avoid IndexError
                min_len = min(mask.shape[0], data.shape[0])
                mask = mask[:min_len]
            plt.scatter(data[mask, 0], data[mask, 1], 
                       c=[soft_color_map[class_idx]], 
                       label=f'{class_name}', 
                       s=80, alpha=0.6, marker='o',
                       edgecolors='black', linewidths=0.5)
        
        # Plot sampled data as stars on top, colored by their class
        # Determine whether sampled indices are positions (relative to data rows) or image ids
        sampled_positions = []
        sampled_image_ids = []
        if len(selected_samples) > 0 and np.max(selected_samples) < len(self.image_ids):
            # They look like positions into the data array
            sampled_positions = selected_samples.tolist()
            sampled_image_ids = [self.image_ids[pos] for pos in sampled_positions]
        else:
            # Treat them as image ids (keys)
            sampled_image_ids = [int(x) for x in selected_samples.tolist()]
            # Map image ids to positions; ignore ids not found
            sampled_positions = [self.image_ids.index(img_id) for img_id in sampled_image_ids if img_id in self.image_ids]

        sampled_classes_plotted = set()
        for pos, img_id in zip(sampled_positions, sampled_image_ids):
            if pos < 0 or pos >= data.shape[0]:
                continue
            label = labels[pos]
            class_name = self.dataset.get_class_name(label)

            # Add label only once per class for sampled items
            if label not in sampled_classes_plotted:
                plt.scatter(data[pos, 0], data[pos, 1], 
                           c=vibrant_color_map[label],  # Vibrant color for selected items
                           s=250, alpha=1.0, marker=marker_map[label],  # Different marker per class
                           edgecolors='black', linewidths=2.0,
                           label=f'{class_name} (sampled)')
                sampled_classes_plotted.add(label)
            else:
                plt.scatter(data[pos, 0], data[pos, 1], 
                           c=vibrant_color_map[label],  # Vibrant color for selected items
                           s=250, alpha=1.0, marker=marker_map[label],  # Different marker per class
                           edgecolors='black', linewidths=2.0)
        
        plt.title(f"t-SNE Visualization - Round {self.index}\nCircles: All Data | Markers: Sampled Data", fontsize=14)
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)
        
        # Create two separate legends
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Separate handles and labels into two groups
        non_sampled_handles = []
        non_sampled_labels = []
        sampled_handles = []
        sampled_labels = []
        
        for handle, label in zip(handles, labels):
            if '(sampled)' in label:
                sampled_handles.append(handle)
                sampled_labels.append(label)
            else:
                non_sampled_handles.append(handle)
                non_sampled_labels.append(label)
        
        # Create first legend for non-sampled classes
        first_legend = plt.legend(non_sampled_handles, non_sampled_labels, 
                                 bbox_to_anchor=(1.05, 1), loc='upper left', 
                                 fontsize=9, title='Non-Selected Samples', title_fontsize=10)
        plt.gca().add_artist(first_legend)
        
        # Create second legend for sampled classes
        plt.legend(sampled_handles, sampled_labels, 
                  bbox_to_anchor=(1.05, 0.5), loc='center left', 
                  fontsize=9, title='Selected Samples', title_fontsize=10, markerscale=0.4)
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'results/nq_{n}_round_{self.index}_ssl_features_visualization.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'results/nq_{n}_round_{self.index}_ssl_features_visualization.pdf', dpi=600, bbox_inches='tight')
        
        # plt.show()
        
        # Increment index for next round
        self.index += 1

        # Remove selected items from features_dict and features_2d using mapped image ids and positions
        for img_id in sampled_image_ids:
            if img_id in features_dict:
                del features_dict[img_id]

        if len(sampled_positions) > 0:
            # Sort positions descending so deletions do not shift earlier indices
            positions_to_delete = sorted(sampled_positions, reverse=True)
            for pos in positions_to_delete:
                if 0 <= pos < self.features_2d.shape[0]:
                    self.features_2d = np.delete(self.features_2d, pos, axis=0)
                    # Also remove the corresponding image id
                    try:
                        del self.image_ids[pos]
                    except Exception:
                        pass
                
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
                
