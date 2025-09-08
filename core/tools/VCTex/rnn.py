import torch

class RNN:
    """
    Randomized Neural Network (RNN) implementation.

    Args:
        Q (int):
            The number of neurons in the hidden layers, or more specifically,
            the latent space dimension.
        
        P (int):
            The number of features in the input layer, or more specifically,
            the input space dimension.

        N (int):
            The number of samples.

        lambda_ (float):
            The Tikhonov's regularization parameter value.

        device (torch.device):
            The device to run the network.
    """

    def __init__(self, Q: int, p: int, N: int, lambda_: float, device: torch.device):
        self._Q = Q  # The hidden layer neuron count.
        self._p = p  # The input matrix rows count. (Number of attributes)
        self._N = N  # The input matrix columns count. (Number of samples)

        self._lambda = lambda_  # Tikhonov's regularization parameter.
        self._device = device  # The device to run the network.

        # Hidden layer's output activation function.
        self._activation = torch.sigmoid

        # Generate the random weight matrix and the bias matrix.
        self._weights = self._setup_weight_matrix().to(self._device)
        self._eye = lambda_ * torch.eye(Q + 1, dtype=torch.float).to(self._device)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Calculate the output layer's weights based on the specified input and output feature matrices.

        Args:
            X (torch.Tensor) of expected shape (p, N):
                The input feature matrix of shape (p, N), where `P` is the number
                of input features, and `N` is the number of samples.
            
            Y (torch.Tensor) of expected shape (r, N):
                The output feature matrix of shape (r, N), where `r` is the number
                of output features, and `N` is the number of samples.

        Returns:
            (torch.Tensor) of expected shape (r, Q):
                The output layer's weights calculated using the (regularized) least-squares methods.
        """
        # Send the input and output feature matrices to the specified device.
        X = X.to(self._device)
        Y = Y.to(self._device)
        
        # Add the bias to X.
        X = torch.vstack([torch.ones((X.shape[1], ), device=self._device) * -1., X])

        # Hidden layer's output matrix.
        Z = self._activation(torch.mm(self._weights, X))

        # Add the bias to Z.
        Z = torch.vstack([torch.ones((Z.shape[1], ), device=self._device) * -1., Z])

        # Calculates beta applying Tikhonov's regularization.
        beta = torch.mm(torch.mm(Y, Z.t()), torch.linalg.inv(
            torch.add(torch.mm(Z, Z.t()), self._eye)))

        self.beta = beta

    def _setup_weight_matrix(self):
        """
        Builds the random weight matrix.

        Returns:
            (torch.Tensor) of expected shape (Q, p + 1):
                The random weight matrix used to randomly project the input feature matrix.
        """
        L = self._Q * (self._p + 1)

        V = torch.zeros(L, dtype=torch.float)
        V[0] = L + 1

        a = L + 2
        b = L + 3
        c = L * L

        for x in range(1, L):
            V[x] = (a * V[x-1] + b) % c
        
        # Always keep the zscore normalization for our LCG weights
        V = torch.divide(torch.subtract(V, torch.mean(V)), torch.std(V))

        return V.reshape((self._Q, self._p + 1))