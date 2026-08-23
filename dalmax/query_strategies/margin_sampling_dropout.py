from .base import Strategy


class MarginSamplingDropout(Strategy):
    def __init__(self, dataset, net, logger):
        super().__init__(dataset, net, logger)
        self.n_drop = net.params['n_drop']

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data, n_drop=self.n_drop)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
