import numpy as np
import matplotlib.pyplot as plt
from .strategy import Strategy
from sklearn.cluster import KMeans
import time 

from ..tools.SSRAE.extractor import ColorFeatureExtractor

class SSRAEKmeansSampling(Strategy):
    def __init__(self, dataset, net, logger):
        super(SSRAEKmeansSampling, self).__init__(dataset, net, logger)

    def query(self, n):
        print(f"Initializing the DAL strategy with SSRAESampling query {n} samples")
        samples = np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)
        print(f"Samples selected: {samples}")
        
        # Imprime todos os ids samples do dataset inteiro que já estão anotados e que não estão anotados (pool)
        labeled_ids = np.where(self.dataset.labeled_idxs==1)[0]
        unlabeled_ids = np.where(self.dataset.labeled_idxs==0)[0]
        # print(f"Labeled IDs: {labeled_ids}")
        # print(f"Unlabeled IDs: {unlabeled_ids}")
        # Len
        print(f"---->len Labeled IDs length: {len(labeled_ids)}")
        print(f"---->len Unlabeled IDs length: {len(unlabeled_ids)}")

        features_dict = self.dataset.features_dict
        print(f"---->len features_dict IDs: {len(features_dict)}")

        # Convert features dictionary to matrix for K-means
        image_ids = list(features_dict.keys())
        features_matrix = np.vstack([features_dict[img_id].numpy() for img_id in image_ids])
        
        print(f"\nRunning K-means clustering with n={n} clusters...")
        print(f"Features matrix shape: {features_matrix.shape}")
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n, random_state=3 , n_init=10)
        cluster_labels = kmeans.fit_predict(features_matrix)
        centroids = kmeans.cluster_centers_
        
        # Find the sample closest to each centroid
        selected_samples = []
        
        for i in range(n):
            # Calculate distances from all points to centroid i
            distances = np.linalg.norm(features_matrix - centroids[i], axis=1)
            
            # Find the index of the closest point
            closest_idx = np.argmin(distances)
            
            # Get the corresponding image ID
            selected_img_id = image_ids[closest_idx]
            selected_samples.append(selected_img_id)
            
            print(f"Cluster {i}: Selected image ID {selected_img_id} (distance: {distances[closest_idx]:.4f})")
        
        selected_samples = np.array(selected_samples)
        print(f"\nSelected samples using SSRAE + K-means: {selected_samples}")
        
        # Plot the selected images
        # self.plot_selected_images(selected_samples)

        # Remove selected_samples do features_dict
        for img_id in selected_samples:
            if img_id in features_dict:
                del features_dict[img_id]

        return selected_samples
    
    def plot_selected_images(self, samples):
        """Plot the selected images in a grid layout"""
        n_samples = len(samples)
        n_cols = 5  # 5 columns
        n_rows = (n_samples + n_cols - 1) // n_cols  # Calculate needed rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        fig.suptitle(f'Selected Images for Active Learning ({n_samples} samples)', fontsize=16)
        
        # Flatten axes array for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, sample_idx in enumerate(samples):
            # Get the image and label
            image = self.dataset.X_train[sample_idx]
            label = self.dataset.Y_train[sample_idx]
            
            # Get class name if available
            class_name = self.dataset.get_class_name(label) if hasattr(self.dataset, 'get_class_name') else f'Class {label}'
            
            # Plot the image
            axes[i].imshow(image)
            axes[i].set_title(f'ID: {sample_idx}\n{class_name}', fontsize=10)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Also save the plot
        plt.savefig('selected_images_ssrae.png', dpi=300, bbox_inches='tight')
        print(f"Selected images plot saved as 'selected_images_ssrae.png'")
