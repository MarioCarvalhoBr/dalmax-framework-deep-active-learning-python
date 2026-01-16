import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .strategy import Strategy


class AdversarialBIM(Strategy):
    def __init__(self, dataset, net, logger, eps=0.05, max_iter=10, batch_size=64, device=None):
        super(AdversarialBIM, self).__init__(dataset, net, logger)
        self.eps = eps
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _bim_distance_batch(self, x_batch):
        """Compute BIM perturbation L2 distance for a batch."""
        x_batch = x_batch.to(self.device)

        with torch.no_grad():
            out0, _ = self.net.clf(x_batch)
            y0 = out0.argmax(dim=1)

        eta = torch.zeros_like(x_batch, requires_grad=True)

        for _ in range(self.max_iter):
            out, _ = self.net.clf(x_batch + eta)
            loss = F.cross_entropy(out, y0, reduction="sum")
            grad = torch.autograd.grad(loss, eta, retain_graph=False, create_graph=False)[0]
            eta = (eta + self.eps * grad.sign()).detach().requires_grad_(True)

            with torch.no_grad():
                preds = out.argmax(dim=1)
                if (preds != y0).all():
                    break

        with torch.no_grad():
            dis = (eta * eta).flatten(start_dim=1).sum(dim=1)
        return dis.cpu().numpy()

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        self.net.clf.to(self.device)
        self.net.clf.eval()

        loader = DataLoader(
            unlabeled_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        dis = np.zeros(len(unlabeled_idxs), dtype=np.float32)
        offset = 0

        for batch in tqdm(loader, total=len(loader), ncols=100):
            x_batch, _, idx_batch = batch
            dis_batch = self._bim_distance_batch(x_batch)
            bsz = len(dis_batch)
            dis[offset : offset + bsz] = dis_batch
            offset += bsz

        return unlabeled_idxs[dis.argsort()[:n]]


