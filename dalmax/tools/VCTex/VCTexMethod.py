import torch
import numpy as np

from .extractor import VCTex

class VCTexMethod:
    def __init__(self, Q: list[int], device: torch.device):
        """
        VCTex: Color-Texture Representation Extractor.

        Args:
            Q (int | list[int]): 
                The number of neurons in the hidden layer(s), or more specifically,
                the latent space dimension(s) of the randomized autoencoder.
                Can be a single value or a list of values.
            device (torch.device):
                The device to run the randomized autoencoder.
        """
        # Convert single Q to list for uniform handling
        self._Q = [Q] if isinstance(Q, int) else Q
        self._device = device
        
        # Create an extractor for each Q value
        self._extractors = [VCTex(Q=q, device=device) for q in self._Q]

    def __call__(self, image: np.ndarray) -> list[float]:
        """
        Extract features from the input image using all configured Q values.
        If multiple Q values are used, the features are concatenated.

        Args:
            image (np.ndarray): The input image with shape (H, W, C)

        Returns:
            list[float]: Concatenated features from all extractors
        """
        # Extract features using each extractor
        all_features = []
        for extractor in self._extractors:
            features = extractor(image)
            all_features.extend(features)
            
        return all_features



