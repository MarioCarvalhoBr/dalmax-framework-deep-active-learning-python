from abc import abstractmethod
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

class SSLStrategy(Strategy):
    def __init__(self, dataset, net, logger):
        super(SSLStrategy, self).__init__(dataset, net, logger)

        self.index = 0  # To keep track of the next sample to select
        self.flag = True
        self.features_2d = None
        self.params = self.params if hasattr(self, 'params') else None  # Ensure params attribute exists

        self.logger.warning(f"\n\n---->INSTANCE> DAL strategy with {self.__class__.__name__}")

    def query(self, n):
        self.logger.warning(f"Initializing the DAL strategy with {self.__class__.__name__} query {n} samples")
        
        features_dict = self.dataset.features_dict
        # self.logger.warning shape of path_pkl
        self.logger.warning(f"Features dictionary contains {len(features_dict)} items.")
        self.logger.warning(f"Example feature vector shape: {next(iter(features_dict.values())).shape}")
        
        self.logger.warning(f'Type of features_dict: {type(features_dict)}')
        # Tipo de um item do vetor dicionario features_dict
        self.logger.warning(f'Type of an item in features_dict: {type(next(iter(features_dict.values())))}')

        # Converte features_dict de forma segura para usar o .numpy(): OS itens podem ser tensores ou arrays: <class 'numpy.ndarray'> ou <class 'torch.Tensor'>
        data = None
        if isinstance(next(iter(features_dict.values())), np.ndarray):
            data = np.vstack([features_dict[key] for key in features_dict])
        elif isinstance(next(iter(features_dict.values())), torch.Tensor):
            data = np.vstack([features_dict[key].numpy() for key in features_dict])
        else:
            raise ValueError("Unsupported data type in features_dict values.")


        TARGET_SIZE = n  # Number of samples to select
        NUMERO_GRUPOS_1_ITER = 500 # 300, 200
        NUMERO_GRUPOS_2_ITER = 200 # 100, 50
        NUMERO_GRUPOS_3_ITER = 50 # 50, 25
        
        N_LEVELS = 2
        
        sample_sizes = [15, 2] # 30, 15, 2
        
        # self.logger.warning build info
        self.logger.warning("\n\n-----------------------------------------------------")
        self.logger.warning(f"--->Starting {self.__class__.__name__} hierarchical K-means clustering with resampling...")
        self.logger.warning(f"--->Target size for sampling: {TARGET_SIZE}")
        self.logger.warning(f"--->Data shape for clustering: {data.shape}")
        self.logger.warning(f"--->Number of levels: {N_LEVELS}")
        self.logger.warning(f"--->Number of clusters per level: {[NUMERO_GRUPOS_1_ITER, NUMERO_GRUPOS_2_ITER]}")
        self.logger.warning(f"--->Sample sizes per level: {sample_sizes}")
        self.logger.warning("-----------------------------------------------------\n\n")

        
        clusters = hkmg.hierarchical_kmeans_with_resampling(
            data=torch.tensor(data, device="cuda", dtype=torch.float32),
            n_clusters=[NUMERO_GRUPOS_1_ITER, NUMERO_GRUPOS_2_ITER],
            n_levels=N_LEVELS,
            sample_sizes=sample_sizes,
            verbose=False
        )

        cl = HierarchicalCluster.from_dict(clusters)
        sampled_indices = hs.hierarchical_sampling(cl, target_size=TARGET_SIZE)

        selected_samples = np.array(sampled_indices)
        self.logger.warning(f"\nSelected samples using {self.__class__.__name__} + K-means HierarchicalCluster: {selected_samples}")
        

        # Remove selected_samples do features_dict
        for img_id in selected_samples:
            if img_id in features_dict:
                del features_dict[img_id]
                
        return selected_samples
                
class SSRAEKmeansHCSampling(SSLStrategy):
    def __init__(self, dataset, net, logger):
        super(SSRAEKmeansHCSampling, self).__init__(dataset, net, logger)

class VCTexKmeansHCSampling(SSLStrategy):
    def __init__(self, dataset, net, logger):
        super(VCTexKmeansHCSampling, self).__init__(dataset, net, logger)
