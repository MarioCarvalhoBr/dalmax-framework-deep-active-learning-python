from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dalmax.models.checkpoint import load_checkpoint, save_checkpoint


class DeepLearning:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    def save_model(
        self,
        path: str,
        *,
        model_name: str,
        class_names: list[str],
        img_size: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save the trained model (`self.clf`) as a `dalmax-checkpoint`.

        Historical bug (see `dalmax.models.checkpoint` module docstring):
        this used to do `torch.save(self.net, path)`, saving the model
        *class* (`self.net`) instead of the trained instance (`self.clf`) —
        every checkpoint produced that way contains no weights.
        `model_name`/`class_names`/`img_size`/`extra` are required because
        `DeepLearning` itself only holds the legacy `params` dict, not the
        dataset/registry context needed to make the checkpoint
        self-describing for later inference (`dalmax/inference/predictor.py`).
        """
        save_checkpoint(
            path,
            self.clf,
            model_name=model_name,
            n_classes=self.params["n_classes"],
            class_names=class_names,
            img_size=img_size,
            extra=extra,
        )

    def load_model(self, path: str) -> dict[str, Any]:
        """Load a `dalmax-checkpoint` written by `save_model`, setting
        `self.clf` to the restored, `.eval()`'d model on `self.device`.

        Returns the checkpoint's metadata dict (see
        `dalmax.models.checkpoint.load_checkpoint`). Raises
        `dalmax.models.checkpoint.CheckpointError` on a legacy pre-fix
        class-pickle file (the historical bug this replaces) or any other
        file that is not a `dalmax-checkpoint`.
        """
        self.clf, meta = load_checkpoint(path, device=self.device)
        return meta

    def train(self, data):
        n_epoch = self.params['n_epoch']
        n_classes = self.params['n_classes']
        self.clf = self.net(n_classes).to(self.device)
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            all_acc = []
            all_loss = []
            for _batch_idx, (x, y, _idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                all_loss.append(loss.item())
                pred = out.max(1)[1]
                acc = 1.0 * (pred == y).sum().item() / len(y)
                all_acc.append(acc)
                loss.backward()
                optimizer.step()
            # Calcular a loss e a acurácia
            mean_loss = np.mean(all_loss)
            mean_acc = np.mean(all_acc)
            print(f" - Epoch {epoch}/{n_epoch} Loss: {mean_loss:.4f} Acc: {mean_acc:.4f}")


    def predict(self, data):
        self.clf.eval()
        # preds = torch.zeros(len(data), dtype=data.Y.dtype)
        if not isinstance(data.Y, torch.Tensor):
            data.Y = torch.tensor(data.Y)
        preds = torch.zeros(len(data), dtype=data.Y.dtype)


        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:

                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_prob_dropout(self, data, n_drop):
        n_drop = self.params['n_drop']
        print("predict_prob_dropout with n_drop", n_drop)
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for _i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data, n_drop):
        n_drop = self.params['n_drop']
        print("predict_prob_dropout_split with n_drop", n_drop)
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings

