import os

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from dalmax.data.datasets import Data


def list_image_names(class_dir: str) -> list[str]:
    """File names of one class folder in the loaders' ordering contract (KI-47).

    Contract (`dalmax/data_order.py::ORDER_CONTRACT`): plain lexicographic sort of
    the names (`sorted()` on `str`, i.e. by Unicode code point), never the file
    system's listing order, which differs between machines and between Colab
    sessions. Pool and test indices are positions in the lists built from this
    order, so the same index must name the same image everywhere.
    """
    return sorted(os.listdir(class_dir))


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
            for img_name in list_image_names(class_dir):  # sorted: the order contract (KI-47)
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Carregar e redimensionar imagem
                    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
                    images.append(np.array(img))  # Converter para array numpy
                    labels.append(class_to_idx[class_name])  # Obter índice da classe
                    images_path.append(img_path)
                except Exception as e:
                    # Fail fast (KI-47): skipping an unreadable file would shift every
                    # later index, so the same index would name another image.
                    raise RuntimeError(f"cannot load image {img_path}: {e}") from e
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
            for img_name in list_image_names(class_dir):  # sorted: the order contract (KI-47)
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Carregar e redimensionar imagem
                    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
                    images.append(np.array(img))  # Converter para array numpy
                    labels.append(class_to_idx[class_name])  # Obter índice da classe
                    images_path.append(img_path)
                except Exception as e:
                    # Fail fast (KI-47): skipping an unreadable file would shift every
                    # later index, so the same index would name another image.
                    raise RuntimeError(f"cannot load image {img_path}: {e}") from e
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
