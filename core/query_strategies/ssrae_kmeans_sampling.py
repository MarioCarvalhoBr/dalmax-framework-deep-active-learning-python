import numpy as np
import matplotlib.pyplot as plt
from .strategy import Strategy
from sklearn.cluster import KMeans
import time 

class SSRAEKmeansSampling(Strategy):
    def __init__(self, dataset, net, logger):
        super(SSRAEKmeansSampling, self).__init__(dataset, net, logger)

    def query(self, n):
        print(f"Initializing the DAL strategy with SSRAESampling query {n} samples")
        
        features_dict = self.dataset.features_dict

        # Convert features dictionary to matrix for K-means
        image_ids = list(features_dict.keys())
        features_matrix = np.vstack([features_dict[img_id].numpy() for img_id in image_ids])
        
        print(f"\nRunning K-means clustering with n={n} clusters...")
        
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
        

        # Remove selected_samples do features_dict
        for img_id in selected_samples:
            if img_id in features_dict:
                del features_dict[img_id]

        return selected_samples
    
