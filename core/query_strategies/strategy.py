import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Strategy:
    def __init__(self, dataset, net, logger):
        self.dataset = dataset
        self.net = net
        self.logger = logger

    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.logger.warning(f"Updating the DAL strategy with {len(pos_idxs)} positive samples and {(neg_idxs)} negative samples")
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def info(self):
        self.logger.warning("-------------------------------------------------------------------")
        self.logger.warning(f"Size of unlabeled pool: {self.dataset.get_size_pool_unlabeled()}")
        self.logger.warning(f"Size of labeled pool: {self.dataset.get_size_bucket_labeled()}")
        self.logger.warning(f"Size of training data: {self.dataset.get_size_train_data()}")
        self.logger.warning(f"Size of testing data: {self.dataset.get_size_test_data()}")
        self.logger.warning("-------------------------------------------------------------------")

    def train(self):
        self.logger.warning("Training the DAL strategy")
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def train_full(self):
        self.logger.warning("Training FULL DATASET the without DAL strategy")
        labeled_idxs, labeled_data = self.dataset.get_train_data()
        self.net.train(labeled_data)

    def predict(self, data):
        self.logger.warning("Predicting with the DAL strategy")
        preds = self.net.predict(data)
        return preds
    
    def predict_prob(self, data):
        self.logger.warning("Predicting probabilities with the DAL strategy")
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop):
        self.logger.warning(f"Predicting probabilities with the DAL strategy using dropout {n_drop}")
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop):
        self.logger.warning(f"Predicting probabilities with the DAL strategy using dropout {n_drop} and split")
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        self.logger.warning("Getting embeddings with the DAL strategy")
        embeddings = self.net.get_embeddings(data)
        return embeddings
    
    def save_model(self, dir_results):
        
        path = dir_results + "/saved_model.pth"
        self.net.save_model(path)
        self.logger.warning(f"Model saved in '{path}'.")


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
