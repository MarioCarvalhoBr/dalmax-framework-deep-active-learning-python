import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models

class DeepLearning:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    def save_model(self, path):
        torch.save(self.net, path)
    
    # TODO: FIX THIS
    def load_model(self, path):
        self.net = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.net.eval()
        n_classes = self.params['n_classes']    
        self.clf = self.net(n_classes).to(self.device)

        return self.net

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
            for batch_idx, (x, y, idxs) in enumerate(loader):
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
        for i in range(n_drop):
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
                    prob = F.softmax(out, dim=1)
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

