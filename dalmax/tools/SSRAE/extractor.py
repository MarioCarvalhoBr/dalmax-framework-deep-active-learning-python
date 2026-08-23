import torch
import numpy as np

from .rnn import RNN
from .splitter import WindowSplitter

class ColorFeatureExtractor():

    def __init__(self, Q: int) -> None:
        """
        A New Approach to Learn Spatio-Spectral Texture Representation with Randomized
        Networks: Application to Brazilian Plant Species Identification.
        
        Args:
            Q (int): The hidden neuron count. This parameter also specifies the
                dimension of the space where the features are projected.
        """
        # The number of hidden neurons in the hidden layer. This parameter also
        # specifies the dimension of the space where the features are projected.
        self._Q = Q
        
        # The method for splitting the image in windows.
        self._splitter = WindowSplitter()

    def extract(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract the features for the input image with dimensions (H, W, C),
        where H is the image height, W is the image width, and C is the number
        of channels.
        
        Args:
            image (np.ndarray): The image with dimensions (H, W, C), where
                C represents the number of channels.
                
        Raises:
            ValueError: Either if the image dimension is not equal to 3, or
                if the third dimension is not equal to 3.
                
        Returns:
            (torch.Tensor): The extracted features.
        """
        # Check if the dimension of the specified image is 3.
        if len(image.shape) != 3:
            raise ValueError('Unexpected image dimension. Expected: %d-D. Got: %d-D.' % (3, len(image.shape)))

        # Check if the third dimension is equal to 3, indicating the amount of channels.
        if image.shape[2] != 3:
            raise ValueError('Unexpected image channel amount. Expected: %d. Got: %d.' % (3, image.shape[2]))

        # The image split into multiple windows.
        windows_R = self._splitter.split(image[:, :, 0], window_size=3, padding=True)
        windows_G = self._splitter.split(image[:, :, 1], window_size=3, padding=True)
        windows_B = self._splitter.split(image[:, :, 2], window_size=3, padding=True)

        # The input matrix for the autoencoder consists of the image split into multiple windows.
        X_R = torch.from_numpy(windows_R.T).float()
        X_G = torch.from_numpy(windows_G.T).float()
        X_B = torch.from_numpy(windows_B.T).float()

        XX_R = X_R.clone()
        XX_G = X_G.clone()
        XX_B = X_B.clone()

        # Apply z-score in the input feature matrices.
        X_R = X_R.t_()
        X_R = torch.divide(torch.subtract(X_R, torch.mean(X_R, dim=0)), torch.std(X_R, dim=0) + 1e-3)
        X_R = X_R.t_()

        X_G = X_G.t_()
        X_G = torch.divide(torch.subtract(X_G, torch.mean(X_G, dim=0)), torch.std(X_G, dim=0) + 1e-3)
        X_G = X_G.t_()

        X_B = X_B.t_()
        X_B = torch.divide(torch.subtract(X_B, torch.mean(X_B, dim=0)), torch.std(X_B, dim=0) + 1e-3)
        X_B = X_B.t_()

        # Instantiate the randomized neural network.
        rnn = RNN(Q=self._Q, P=X_R.shape[0], tikhonov=True, lambda_=1e-3)

        # Fit the input matrix and fetch the beta matrix from the autoencoder.
        rnn.fit(X_R, XX_R)
        beta_R = rnn.beta
        
        rnn.fit(X_G, XX_G)
        beta_G = rnn.beta

        rnn.fit(X_B, XX_B)
        beta_B = rnn.beta

        rnn.fit(X_R, XX_G)
        beta_S_R = rnn.beta

        rnn.fit(X_G, XX_B)
        beta_S_G = rnn.beta

        rnn.fit(X_B, XX_R)
        beta_S_B = rnn.beta

        beta = torch.hstack([beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B]).reshape((1, -1))
        
        return beta.squeeze()