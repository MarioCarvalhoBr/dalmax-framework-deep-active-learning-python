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

class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device
        
    def train(self, data):
        n_epoch = self.params['n_epoch']
        self.clf = self.net().to(self.device)
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
        preds = torch.zeros(len(data), dtype=torch.tensor(data.Y).dtype)

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
    
    def predict_prob_dropout(self, data, n_drop=10):
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
    
    def predict_prob_dropout_split(self, data, n_drop=10):
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

class DaninhasModel(nn.Module):
    def __init__(self):
        super(DaninhasModel, self).__init__()
        
        # Carregar a ResNet-50 pré-treinada
        resnet = models.resnet18(pretrained=True)
        
        # Remover a camada totalmente conectada final
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Todas as camadas até o penúltimo bloco
        
        # Número de features da penúltima camada
        num_features = resnet.fc.in_features  
        
        # Embedding layer para reduzir a dimensionalidade
        self.embedding_dim = 50
        self.embedding_layer = nn.Linear(num_features, self.embedding_dim)
        
        # Camada final para classificação em 5 classes
        self.classifier = nn.Linear(num_features, 5)

    def forward(self, x):
        # Extrair features da backbone ResNet
        features = self.feature_extractor(x)  # Saída: [batch_size, num_features, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten para [batch_size, num_features]
        
        # Embedding: reduz para 50 dimensões
        e1 = self.embedding_layer(features)
        e1 = torch.relu(e1)
        
        # Saída final para classificação
        x = self.classifier(features)
        return x, e1

    def get_embedding_dim(self):
        return self.embedding_dim
