import pickle
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import datasets
from core.tools.SSRAE.extractor import ColorFeatureExtractor

from core.tools.VCTex.VCTexMethod import VCTexMethod



import time

class Data:
    def __init__(self, X_train, Y_train,Z_train_paths, X_test, Y_test,Z_test_paths, handler, classes, class_to_idx):
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.X_train = X_train
        self.Y_train = Y_train
        # Save Y_train with picke in results/Y_train.pkl
        path_pkl = 'results/Y_train.pkl'
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
        
    def create_feature_maps_vctex(self):
        # Verifica se o arquivo results/features_dict_vctex.pkl existe. Se sim, carrega e adiciona a features_dict
        path_pkl = 'results/features_dict_vctex.pkl'
        if os.path.exists(path_pkl):
            with open(path_pkl, 'rb') as f:
                self.features_dict = pickle.load(f)
            print(f"Features dictionary loaded from {path_pkl}")
        else:
            print(f"Features dictionary file {path_pkl} NOT FOUND. Please run the feature extraction first.")
                        
            # Imprime todos os ids samples do dataset inteiro que já estão anotados e que não estão anotados (pool)
            labeled_ids = np.where(self.labeled_idxs==1)[0]
            unlabeled_ids = np.where(self.labeled_idxs==0)[0]
            # Print len 
            print(f"---->Labeled IDs length: {len(labeled_ids)}")
            print(f"---->Unlabeled IDs length: {len(unlabeled_ids)}")
            print(f"---->Labeled IDs: {labeled_ids}")
            print(f"---->Unlabeled IDs: {unlabeled_ids}")
            
            
            # Device.
            device = torch.device("cuda:0")

            # Method's hyperparameters.
            Q = [5,17] #best parameters of the paper. You can test different vales 

            # Instantiate the color feature extractor.
            extractor = VCTexMethod(Q=Q, device=device)

            # Dictionary to store features for each unlabeled image
            features_dict = {}
            
            print(f"Extracting features for {len(unlabeled_ids)} unlabeled images...")
            start_time = time.time()
            
            # Extract features for each unlabeled image
            for i, img_id in enumerate(unlabeled_ids):
                # Get the image from the dataset
                image = self.X_train[img_id]

                # Extract features from the image
                features = extractor(image)
                
                # Store in dictionary
                features_dict[img_id] = np.array(features)

                # Print progress every 100 images
                if (i + 1) % 100 == 0 or i == 0:
                    print(f"Processed {i + 1}/{len(unlabeled_ids)} images...")
            
            end_time = time.time()
            print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
            print(f"Features dictionary contains {len(features_dict)} entries")
            print(f"Feature shape for each image: {list(features_dict.values())[0].shape}")
            
            # Show example of first few entries
            first_ids = list(features_dict.keys())[:3]
            print(f"\nExample features for first 3 images:")
            for img_id in first_ids:
                print(f"Image ID {img_id}: Features shape = {features_dict[img_id].shape}")

            self.features_dict = features_dict
            
            # Use t-SNE to convert each image features to 2D and plot with class colors
            # self.plot_features_tsne()
            
            
            ## SAVE PICKLE
            if not os.path.exists(path_pkl):
                with open(path_pkl, 'wb') as f:
                    pickle.dump(self.features_dict, f)
                print(f"Features dictionary saved to {path_pkl}")
        

    def create_feature_maps_ssrae(self):

        # Verifica se o arquivo results/features_dict_ssrae.pkl existe. Se sim, carrega e adiciona a features_dict
        path_pkl = 'results/features_dict_ssrae.pkl'
        if os.path.exists(path_pkl):
            with open(path_pkl, 'rb') as f:
                self.features_dict = pickle.load(f)
            print(f"Features dictionary loaded from {path_pkl}")
            
        else:
            print(f"Features dictionary file {path_pkl} NOT FOUND. Please run the feature extraction first.")

            # Imprime todos os ids samples do dataset inteiro que já estão anotados e que não estão anotados (pool)
            labeled_ids = np.where(self.labeled_idxs==1)[0]
            unlabeled_ids = np.where(self.labeled_idxs==0)[0]
            # Print len 
            print(f"---->Labeled IDs length: {len(labeled_ids)}")
            print(f"---->Unlabeled IDs length: {len(unlabeled_ids)}")
            print(f"---->Labeled IDs: {labeled_ids}")
            print(f"---->Unlabeled IDs: {unlabeled_ids}")

            # Method's hyperparameters.
            Q = 13  # The number of hidden neurons.

            # Instantiate the color feature extractor.
            extractor = ColorFeatureExtractor(Q=Q)

            # Dictionary to store features for each unlabeled image
            features_dict = {}
            
            print(f"Extracting features for {len(unlabeled_ids)} unlabeled images...")
            start_time = time.time()
            
            # Extract features for each unlabeled image
            for i, img_id in enumerate(unlabeled_ids):
                # Get the image from the dataset
                image = self.X_train[img_id]

                # Extract features from the image
                features = extractor.extract(image)
                
                # Store in dictionary
                features_dict[img_id] = features
                
                # Print progress every 100 images
                if (i + 1) % 100 == 0 or i == 0:
                    print(f"Processed {i + 1}/{len(unlabeled_ids)} images...")
            
            end_time = time.time()
            print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
            print(f"Features dictionary contains {len(features_dict)} entries")
            print(f"Feature shape for each image: {list(features_dict.values())[0].shape}")
            
            # Show example of first few entries
            first_ids = list(features_dict.keys())[:3]
            print(f"\nExample features for first 3 images:")
            for img_id in first_ids:
                print(f"Image ID {img_id}: Features shape = {features_dict[img_id].shape}")

            self.features_dict = features_dict
        
            
            
            ## SAVE PICKLE
            if not os.path.exists(path_pkl):
                with open(path_pkl, 'wb') as f:
                    pickle.dump(self.features_dict, f)
                print(f"Features dictionary saved to {path_pkl}")
                
        # Use t-SNE to convert each image features to 2D and plot with class colors
        # self.plot_features_tsne()
        

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
        print(f"Initializing with {n_init_labeled} labeled samples using strategy: {strategy_name}")
        self.strategy_name = strategy_name

        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:n_init_labeled]] = True

        if self.strategy_name == "SSRAEKmeansSampling" or self.strategy_name == "SSRAEKmeansHCSampling":
            print(f"Creating SSRAE feature maps...")
            self.create_feature_maps_ssrae()
        elif self.strategy_name == "VCTexKmeansSampling" or self.strategy_name == "VCTexKmeansHCSampling":
            print(f"Creating VCTex feature maps...")
            self.create_feature_maps_vctex()
            
        self.plot_features_tsne()
                
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
    
    def plot_features_tsne(self):
        """Use t-SNE to convert each image features to 2D and plot with class colors"""
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        if not self.features_dict:
            print("No features dictionary found. Skipping t-SNE plot.")
            return
        
        print("Creating t-SNE visualization of features...")
        
        # Get image IDs and convert features to matrix
        image_ids = list(self.features_dict.keys())
        features_matrix = np.vstack([self.features_dict[img_id] for img_id in image_ids])
        
        # Get corresponding labels for the unlabeled images
        labels = [self.Y_train[img_id] for img_id in image_ids]
        
        print(f"Running t-SNE on {features_matrix.shape[0]} samples with {features_matrix.shape[1]} features...")
        
        # Apply t-SNE
        perplexity = min(30, len(image_ids) - 1)  # Ensure perplexity is valid
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', 
                   perplexity=perplexity, random_state=42)
        features_2d = tsne.fit_transform(features_matrix)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Get unique classes and assign colors
        unique_classes = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
        
        # Plot each class with different color
        for i, class_idx in enumerate(unique_classes):
            mask = np.array(labels) == class_idx
            class_name = self.get_class_name(class_idx)
            
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=f'Class {class_idx}: {class_name}', 
                       alpha=0.7, s=50)
        
        plt.title('t-SNE Visualization of SSRAE Features by Class')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('tsne_features_visualization.png', dpi=300, bbox_inches='tight')
        print("t-SNE visualization saved as 'tsne_features_visualization.png'")
        plt.show()
    

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
    return Data(X_train, Y_train, Z_train_paths, X_test, Y_test, Z_test_paths, handler, classes, class_to_idx)

def get_CIFAR10_Download(handler):
    data_train = datasets.CIFAR10('./DATA/NEW_CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./DATA/NEW_CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)