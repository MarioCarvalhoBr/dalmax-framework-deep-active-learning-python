import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

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


