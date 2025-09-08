import torch
import numpy as np

from .rnn import RNN
from .split import split

class VCTex:

    def __init__(self, Q: int, device: torch.device):
        """
        VCTex: Color-Texture Representation Extractor.

        Args:
            Q (int):
                The number of neurons in the hidden layer, or more specifically,
                the latent space dimension of the randomized autoencoder.
        
            device (torch.device):
                The device to run the randomized autoencoder.
        """
        self._Q = Q             # The number of hidden neurons.
        self._device = device   # The device to run the network.
        
        # The randomized autoencoder will be instantiated later.
        self._rae = None

    def __call__(self, image: np.ndarray) -> list[float]:
        """
        Receives a image of expected shape (H, W, C), where `H` is the image's height,
        `W` is the image's width, and `C` is the number of channels, and returns the
        extracted features of the image.

        Args:
            image (np.ndarray) of expected shape (H, W, C):
                The image to have its features extracted, where `H` is the image's height,
                `W ` is the image's widht, and `C` is the number of channels.

        Returns:
            list[float]:
                A list storing the extracted features of the image.
        """
        # Check if the image has three dimensions, i.e. (H, W, C).
        if len(image.shape) != 3:
            raise ValueError('Unexpected image dimension. Expected: %d-D. Got: %d-D.' % (3, len(image.shape)))
        
        # Check if the image is a 3-channel image.
        if image.shape[2] != 3:
            raise ValueError('Unexpected image channel amount. Expected: %d. Got: %d.' % (3, image.shape[2]))

        # Split each color channel in multiple 3x3 windows.
        windows_R = split(image[:, :, 0], window_size=3, padding=True)  # Shape: (H * W, 3 * 3).
        windows_G = split(image[:, :, 1], window_size=3, padding=True)  # Shape: (H * W, 3 * 3).
        windows_B = split(image[:, :, 2], window_size=3, padding=True)  # Shape: (H * W, 3 * 3).

        # Build the partial input feature matrix consisting of the tranpose of the windows.
        X_R = torch.from_numpy(windows_R.T).float()  # Shape: (3 * 3, H * W).
        X_G = torch.from_numpy(windows_G.T).float()  # Shape: (3 * 3, H * W).
        X_B = torch.from_numpy(windows_B.T).float()  # Shape: (3 * 3, H * W).

        X = torch.vstack([
            X_R,
            X_G,
            X_B
        ])  # Shape: (3 * 3 * 3, H * W).

        Y = X.clone()  # Shape: (3 * 3 * 3, H * W).

        # Standardize the input feature matrix.
        X = X.t_()
        X = torch.divide(torch.subtract(X, torch.mean(X, dim=0)), torch.std(X, dim=0) + 1e-3)
        X = X.t_()

        # Instantiate the autoencoder.
        if self._rae is None:
            self._rae = RNN(Q=self._Q, p=X.shape[0], N=X.shape[1], lambda_=1e3, device=self._device)

        # Fit the randomized autoencoder and fetch the output layer's weights.
        self._rae.fit(X, Y)
        beta = self._rae.beta

        # Flatten the output layer's weights.
        return beta.flatten().tolist()
