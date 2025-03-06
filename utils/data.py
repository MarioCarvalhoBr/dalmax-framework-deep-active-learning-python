import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import datasets
class Data:
    def __init__(self, X_train, Y_train,Z_train_paths, X_test, Y_test,Z_test_paths, handler, classes, class_to_idx):
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.X_train = X_train
        self.Y_train = Y_train
        self.Z_train_paths = Z_train_paths

        self.X_test = X_test
        self.Y_test = Y_test
        self.Z_test_paths = Z_test_paths
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

        
    def get_classes_names(self):
        return self.classes
    
    def get_class_name(self, idx):
        return self.classes[idx]
    def get_classes_to_idx(self):
        return self.class_to_idx
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
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
        TN = ((~preds) & (~self.Y_test)).sum().item()
        FP = (preds & (~self.Y_test)).sum().item()
        FN = ((~preds) & self.Y_test).sum().item()
        
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        return precision, recall, f1_score
    
    def calc_metrics_sklearn(self, preds):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    

def get_DANINHAS(handler, data_dir, img_size=128):
    """
    Carrega o dataset estruturado em pastas de classes para treino e teste.
    
    Args:
        handler: Classe manipuladora do dataset (e.g., `MyDataset_Handler`).
        data_dir: Diretório raiz do dataset.
        img_size: Tamanho das imagens (serão redimensionadas para img_size x img_size).
        
    Returns:
        Uma instância da classe `Data` configurada com os dados carregados.
    """
    print("Loading DANINHAS...")
    # Função para carregar imagens e rótulos
    def load_images_and_labels(path, classes, class_to_idx):
        images, labels = [], []
        images_path = []
        for class_name in classes:
            class_dir = os.path.join(path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Carregar e redimensionar imagem
                    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
                    images.append(np.array(img))  # Converter para array numpy
                    labels.append(class_to_idx[class_name])  # Obter índice da classe
                    images_path.append(img_path)
                except Exception as e:
                    print(f"Erro ao carregar a imagem {img_path}: {e}")
        return np.array(images), np.array(labels), images_path

    # Diretórios de treino e teste
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Identificar classes e mapear para índices
    classes = sorted(os.listdir(train_dir))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    print(f"Classes: {classes}")
    print(f"Class to index: {class_to_idx}")

    # Carregar dados de treino
    X_train, Y_train, Z_train_paths = load_images_and_labels(train_dir, classes, class_to_idx)

    # Carregar dados de teste
    X_test, Y_test, Z_test_paths = load_images_and_labels(test_dir, classes, class_to_idx)

    # Criar instância da classe `Data`
    return Data(X_train, Y_train,Z_train_paths, X_test, Y_test,Z_test_paths, handler, classes, class_to_idx)

def get_CIFAR10(handler, data_dir, img_size=32):
    """
    Carrega o dataset estruturado em pastas de classes para treino e teste.
    
    Args:
        handler: Classe manipuladora do dataset (e.g., `MyDataset_Handler`).
        data_dir: Diretório raiz do dataset.
        img_size: Tamanho das imagens (serão redimensionadas para img_size x img_size).
        
    Returns:
        Uma instância da classe `Data` configurada com os dados carregados.
    """
    print("Loading CIFAR10...")
    # Função para carregar imagens e rótulos
    def load_images_and_labels(path, classes, class_to_idx):
        images, labels = [], []
        for class_name in classes:
            class_dir = os.path.join(path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Carregar e redimensionar imagem
                    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
                    images.append(np.array(img))  # Converter para array numpy
                    labels.append(class_to_idx[class_name])  # Obter índice da classe
                except Exception as e:
                    print(f"Erro ao carregar a imagem {img_path}: {e}")
        return np.array(images), np.array(labels)

    # Diretórios de treino e teste
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Identificar classes e mapear para índices
    classes = sorted(os.listdir(train_dir))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    # Carregar dados de treino
    X_train, Y_train, Z_train_paths = load_images_and_labels(train_dir, classes, class_to_idx)

    # Carregar dados de teste
    X_test, Y_test, Z_test_paths = load_images_and_labels(test_dir, classes, class_to_idx)

    # Criar instância da classe `Data`
    return Data(X_train, Y_train,Z_train_paths, X_test, Y_test,Z_test_paths, handler)

def get_CIFAR10_Download(handler):
    data_train = datasets.CIFAR10('./DATA/NEW_CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./DATA/NEW_CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)