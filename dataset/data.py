import numpy as np
import os
from PIL import Image
import torch

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
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
        # return 1.0 * (self.Y_test==preds).sum().item() / self.n_test
        # Garantir que ambos sejam tensores PyTorch
        self.Y_test = torch.tensor(self.Y_test, dtype=torch.int64)
        preds = preds.to(torch.int64)  # Certifique-se de que os tipos combinam

        # Calcular a precisão
        return 1.0 * (self.Y_test == preds).sum().item() / self.n_test

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


def get_DaninhasDataset(handler, data_dir, img_size=128):
    """
    Carrega o dataset estruturado em pastas de classes para treino e teste.
    
    Args:
        handler: Classe manipuladora do dataset (e.g., `MyDataset_Handler`).
        data_dir: Diretório raiz do dataset.
        img_size: Tamanho das imagens (serão redimensionadas para img_size x img_size).
        
    Returns:
        Uma instância da classe `Data` configurada com os dados carregados.
    """
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
    X_train, Y_train = load_images_and_labels(train_dir, classes, class_to_idx)

    # Carregar dados de teste
    X_test, Y_test = load_images_and_labels(test_dir, classes, class_to_idx)

    # Criar instância da classe `Data`
    return Data(X_train, Y_train, X_test, Y_test, handler)