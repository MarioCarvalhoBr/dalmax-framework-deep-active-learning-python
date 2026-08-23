from .base import Strategy


class LeastConfidence(Strategy):
    def __init__(self, dataset, net, logger):
        super().__init__(dataset, net, logger)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
