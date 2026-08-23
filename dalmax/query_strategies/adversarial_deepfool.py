import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import Strategy


class AdversarialDeepFool(Strategy):
    def __init__(self, dataset, net, logger, max_iter=10, batch_size=16, device=None, eps=1e-12):
        super().__init__(dataset, net, logger)
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps

    def _deepfool_distance_batch(self, x_batch):
        """Compute DeepFool perturbation L2 distance for a batch on GPU (top-2 approximation)."""
        x_batch = x_batch.to(self.device)

        with torch.no_grad():
            logits0, _ = self.net.clf(x_batch)
            y0 = logits0.argmax(dim=1)

        eta = torch.zeros_like(x_batch)

        for _ in range(self.max_iter):
            adv = (x_batch + eta).detach().requires_grad_(True)
            logits, _ = self.net.clf(adv)

            # competitor class: second-best logit per sample
            top2 = logits.topk(2, dim=1).indices
            comp = top2[:, 1]

            grad_y0 = torch.autograd.grad(
                logits.gather(1, y0.view(-1, 1)).sum(), adv, retain_graph=True, create_graph=False
            )[0]
            grad_comp = torch.autograd.grad(
                logits.gather(1, comp.view(-1, 1)).sum(), adv, retain_graph=True, create_graph=False
            )[0]

            wi = grad_comp - grad_y0
            wi_flat = wi.flatten(start_dim=1)
            fi = logits[torch.arange(logits.shape[0]), comp] - logits[torch.arange(logits.shape[0]), y0]

            wi_norm = wi_flat.norm(dim=1) + self.eps
            step = (torch.abs(fi) / wi_norm / wi_norm).view(-1, 1, 1, 1)
            eta = eta + step * wi

            with torch.no_grad():
                preds = logits.argmax(dim=1)
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
            dis_batch = self._deepfool_distance_batch(x_batch)
            bsz = len(dis_batch)
            dis[offset : offset + bsz] = dis_batch
            offset += bsz

        return unlabeled_idxs[dis.argsort()[:n]]


