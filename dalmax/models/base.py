from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dalmax.models.checkpoint import load_checkpoint, save_checkpoint
from dalmax.models.mc_dropout import mc_dropout_mode


class DeepLearning:
    def __init__(
        self,
        net: Callable[..., torch.nn.Module],
        params: dict[str, Any],
        device: torch.device,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """`net` is the model class, instantiated at every `train()` call as
        `net(params["n_classes"], **model_kwargs)`. `model_kwargs` carries the
        params JSON's `model_args` (e.g. `{"dropout_p": 0.5}`, KI-41, ADR 0014);
        empty (the default) builds the model exactly as before that option existed."""
        self.net = net
        self.params = params
        self.device = device
        self.model_kwargs: dict[str, Any] = dict(model_kwargs or {})

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
            model_kwargs=self.model_kwargs,
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

    def build_model(self) -> torch.nn.Module:
        """A fresh, untrained task model on `self.device`:
        `net(params["n_classes"], **model_kwargs)`.

        The single place a task model is constructed, shared by `train` and by the
        strategy-specific trainers of ADR 0015 (`dalmax.models.loss_prediction`,
        `dalmax.models.waal`), so all of them draw the same initial weights from the
        run's torch stream.
        """
        return self.net(self.params['n_classes'], **self.model_kwargs).to(self.device)

    def train(self, data):
        n_epoch = self.params['n_epoch']
        self.clf = self.build_model()
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
            # Mean loss and accuracy of the epoch.
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

    def predict_prob_dropout(self, data, n_drop: int) -> torch.Tensor:
        """Mean softmax over `n_drop` MC-dropout passes (dropout active, the rest
        of the model, BatchNorm included, in eval mode; KI-41, ADR 0014).

        Raises `dalmax.models.mc_dropout.MCDropoutError` if the model has no
        dropout module with `p > 0`, and `ValueError` if `n_drop < 1`.
        """
        _check_n_drop(n_drop)
        print("predict_prob_dropout with n_drop", n_drop)
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with mc_dropout_mode(self.clf):
            for _i in range(n_drop):
                with torch.no_grad():
                    for x, y, idxs in loader:
                        x, y = x.to(self.device), y.to(self.device)
                        out, e1 = self.clf(x)
                        prob = F.softmax(out, dim=1)
                        probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data, n_drop: int) -> torch.Tensor:
        """Softmax of each of `n_drop` MC-dropout passes, shape
        `[n_drop, len(data), n_classes]` (same mode handling and errors as
        `predict_prob_dropout`)."""
        _check_n_drop(n_drop)
        print("predict_prob_dropout_split with n_drop", n_drop)
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with mc_dropout_mode(self.clf):
            for i in range(n_drop):
                with torch.no_grad():
                    for x, y, idxs in loader:
                        x, y = x.to(self.device), y.to(self.device)
                        out, e1 = self.clf(x)
                        probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs

    def get_embeddings(self, data) -> torch.Tensor:
        """The model's `e1` for every sample, in eval mode (for the DANINHAS
        ResNet-50: the 2048-d pooled features that feed the classifier, KI-42)."""
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings


    def predict_logits_and_embeddings(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits and `e1` embeddings of every sample in one eval-mode pass, shapes
        `[len(data), n_classes]` and `[len(data), get_embedding_dim()]` (CPU tensors).

        For the strategies that need both from the same forward pass (`BadgeSampling`,
        ADR 0015); `e1` is the input of the final linear layer for both registered
        models (in eval mode their dropout is the identity).
        """
        self.clf.eval()
        logits = torch.zeros([len(data), self.params['n_classes']])
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, _y, idxs in loader:
                out, e1 = self.clf(x.to(self.device))
                logits[idxs] = out.cpu()
                embeddings[idxs] = e1.cpu()
        return logits, embeddings


def _check_n_drop(n_drop: int) -> None:
    """Fail fast on a pass count that would average nothing (division by zero)."""
    if isinstance(n_drop, bool) or not isinstance(n_drop, int) or n_drop < 1:
        raise ValueError(f"n_drop (MC-dropout passes) must be an int >= 1, got {n_drop!r}")
