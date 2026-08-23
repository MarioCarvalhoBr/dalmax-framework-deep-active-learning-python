import os
import pickle

import numpy as np
import torch


def cache_file_path(
    name: str,
    dataset_folder: str,
    q: int | list[int] | None = None,
    pool_hash: str | None = None,
) -> str:
    """Build a cache file path keyed by dataset folder name (and Q, and the
    unlabeled-pool identity, if given).

    Prevents the hazard described in `.claude/rules/reproducibility.md` and
    `.specs/quality/known-issues.md` KI-3: a cache file computed for one
    dataset/Q/pool silently being reused for another. Feature caches
    (SSRAE/VCTex) are extracted only for the current unlabeled pool, which
    depends on `--seed` and `--n_init_labeled`; without `pool_hash` in the key,
    two runs with different seeds/`n_init_labeled` (and therefore different
    unlabeled pools) could silently load each other's stale features. `Y_train`
    has no notion of a pool (it covers the whole training split) so it is
    called with `q=None` and never carries a `pool_hash` segment, regardless of
    what is passed in.

    Files live under `results/cache/` (created if missing) so they never
    collide with the legacy fixed-path caches
    (`results/features_dict_ssrae.pkl`, `results/features_dict_vctex.pkl`,
    `results/Y_train.pkl`), which are left in place untouched for backward
    compatibility.
    """
    cache_dir = "results/cache/"
    os.makedirs(cache_dir, exist_ok=True)
    if q is None:
        return f"{cache_dir}{name}_{dataset_folder}.pkl"
    q_str = "-".join(str(v) for v in q) if isinstance(q, (list, tuple)) else str(q)
    if pool_hash is None:
        return f"{cache_dir}{name}_{dataset_folder}_Q{q_str}.pkl"
    return f"{cache_dir}{name}_{dataset_folder}_Q{q_str}_pool{pool_hash}.pkl"


class Data:
    def __init__(self, X_train, Y_train,Z_train_paths, X_test, Y_test,Z_test_paths, handler, classes, class_to_idx, dataset_folder: str = "unknown"):
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.dataset_folder = dataset_folder
        self.X_train = X_train
        self.Y_train = Y_train
        # Save Y_train with pickle, keyed by dataset folder name so different
        # datasets never share this cache file (see cache_file_path above).
        path_pkl = cache_file_path("Y_train", self.dataset_folder)
        if not os.path.exists(path_pkl):
            with open(path_pkl, 'wb') as f:
                pickle.dump(self.Y_train, f)
            print(f"Y_train saved to {path_pkl}")
        self.Z_train_paths = Z_train_paths

        self.X_test = X_test
        self.Y_test = Y_test
        self.Z_test_paths = Z_test_paths
        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)


        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

        self.features_dict = {}
        self.strategy_name = ""

        self.create_indexes_path()

    def create_indexes_path(self):
        # Salva em um arquivo indices.txt o indice (id) da imagem, classe_nome, e seu caminho
        # Faça com cabeçalho e separado por ;
        with open("results/original_indices.txt", "w") as f:
            f.write("id;class_name;path\n")
            for i, path in enumerate(self.Z_train_paths):
                f.write(f"{i};{self.Y_train[i]};{path}\n")
            for i, path in enumerate(self.Z_test_paths):
                f.write(f"{i};{self.Y_test[i]};{path}\n")

    def get_classes_names(self):
        return self.classes

    def get_image_by_id(self, id):
        if 0 <= id < self.n_pool:
            return self.X_train[id], self.Y_train[id]
        else:
            raise IndexError("Image ID out of range")

    def get_class_name(self, idx):
        return self.classes[idx]
    def get_classes_to_idx(self):
        return self.class_to_idx

    def initialize_labels(self, n_init_labeled, strategy_name):
        """Shuffle the pool and mark the first `n_init_labeled` ids as labeled.

        The legacy SSRAE/VCTex feature-map extraction that used to run here
        for `strategy_name in {"SSRAEKmeansSampling", "SSRAEKmeansHCSampling",
        "VCTexKmeansSampling", "VCTexKmeansHCSampling"}` (behind a
        `compute_legacy_features` flag) was removed in refactor Phase 4: it
        was dead code from `dalmax.cli`'s point of view —
        `dalmax.experiment.runner.ExperimentRunner` never took that path,
        computing embeddings itself via
        `dalmax.query_strategies.representation.RepresentationStrategy` +
        `dalmax.embeddings.EmbeddingProvider`/`EmbeddingCache` instead. See
        `.specs/architecture/refactor-plan.md` Phase 4 and
        `.specs/architecture/current-state.md` §0/§11.
        """
        print(f"Initializing with {n_init_labeled} labeled samples using strategy: {strategy_name}")
        self.strategy_name = strategy_name

        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:n_init_labeled]] = True

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)

    def cal_test_acc(self, preds):
        # Converter self.Y_test para tensor caso seja um array NumPy
        if isinstance(self.Y_test, np.ndarray):
            self.Y_test = torch.from_numpy(self.Y_test).to(torch.int64)
        else:
            self.Y_test = self.Y_test.to(torch.int64)

        # Converter preds para tensor caso seja um array NumPy
        preds = torch.from_numpy(preds).to(torch.int64) if isinstance(preds, np.ndarray) else preds.to(torch.int64)

        # Calcular a precisão
        return 1.0 * (self.Y_test == preds).sum().item() / self.n_test

    # Calcular a precision, recall e f1-score
    def calc_metrics_manual(self, preds):
        # Converter self.Y_test para tensor caso seja um array NumPy
        if isinstance(self.Y_test, np.ndarray):
            self.Y_test = torch.from_numpy(self.Y_test).to(torch.int64)
        else:
            self.Y_test = self.Y_test.to(torch.int64)

        # Converter preds para tensor caso seja um array NumPy
        preds = torch.from_numpy(preds).to(torch.int64) if isinstance(preds, np.ndarray) else preds.to(torch.int64)

        # Calcular a precisão, recall e f1-score
        TP = (preds & self.Y_test).sum().item()
        _TN = ((~preds) & (~self.Y_test)).sum().item()
        FP = (preds & (~self.Y_test)).sum().item()
        FN = ((~preds) & self.Y_test).sum().item()

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return precision, recall, f1_score

    def calc_metrics_sklearn(self, preds):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        # Converter self.Y_test para tensor caso seja um array NumPy
        if isinstance(self.Y_test, np.ndarray):
            self.Y_test = torch.from_numpy(self.Y_test).to(torch.int64)
        else:
            self.Y_test = self.Y_test.to(torch.int64)

        # Calcular a precisão, recall e f1-score
        accuracy = accuracy_score(self.Y_test, preds)
        precision = precision_score(self.Y_test, preds, average='weighted', zero_division=0)
        recall = recall_score(self.Y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(self.Y_test, preds, average='weighted', zero_division=0)

        return accuracy, precision, recall, f1

    def calc_metrics(self, preds) -> dict:
        """Weighted AND macro-averaged accuracy/precision/recall/F1.

        Extends `calc_metrics_sklearn` (kept intact above, unused by new
        code but left for backward compatibility) with macro-averaged
        variants, per `.specs/architecture/refactor-plan.md` Phase 2's
        macro-F1 acceptance criterion — weighted metrics can look strong
        while a minority class is never predicted; macro metrics catch that.
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        # Converter self.Y_test para tensor caso seja um array NumPy
        if isinstance(self.Y_test, np.ndarray):
            self.Y_test = torch.from_numpy(self.Y_test).to(torch.int64)
        else:
            self.Y_test = self.Y_test.to(torch.int64)

        return {
            "acc": accuracy_score(self.Y_test, preds),
            "precision_weighted": precision_score(
                self.Y_test, preds, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                self.Y_test, preds, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(self.Y_test, preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(
                self.Y_test, preds, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(self.Y_test, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(self.Y_test, preds, average="macro", zero_division=0),
        }

    def get_size_pool_unlabeled(self):
        unlabeled_idxs, handler = self.get_unlabeled_data()
        return len(unlabeled_idxs)

    def get_size_bucket_labeled(self):
        labeled_idxs, handler = self.get_labeled_data()
        return len(labeled_idxs)

    def get_size_train_data(self):
        labeled_idxs, handler = self.get_train_data()
        return len(labeled_idxs)

    def get_size_test_data(self):
        handler = self.get_test_data()
        return len(handler)
