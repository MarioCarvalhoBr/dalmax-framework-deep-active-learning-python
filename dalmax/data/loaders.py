import os

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from dalmax.data.datasets import Data


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
    dataset_folder = os.path.basename(data_dir.rstrip("/"))
    return Data(X_train, Y_train,Z_train_paths, X_test, Y_test,Z_test_paths, handler, classes, class_to_idx, dataset_folder=dataset_folder)

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

    # Carregar dados de treino
    X_train, Y_train, Z_train_paths = load_images_and_labels(train_dir, classes, class_to_idx)

    # Carregar dados de teste
    X_test, Y_test, Z_test_paths = load_images_and_labels(test_dir, classes, class_to_idx)

    # Criar instância da classe `Data`
    dataset_folder = os.path.basename(data_dir.rstrip("/"))
    return Data(X_train, Y_train, Z_train_paths, X_test, Y_test, Z_test_paths, handler, classes, class_to_idx, dataset_folder=dataset_folder)

def get_CIFAR10_Download(handler):
    data_train = datasets.CIFAR10('./DATA/NEW_CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./DATA/NEW_CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)
