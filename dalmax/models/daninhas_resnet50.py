import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class DaninhasModelResNet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        # Carregar a ResNet-18 pré-treinada com os pesos da ImageNet
        weights = ResNet50_Weights.IMAGENET1K_V1
        resnet = resnet50(weights=weights)

        # Remover a camada totalmente conectada final
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Todas as camadas até o penúltimo bloco

        # Número de features da penúltima camada
        num_features = resnet.fc.in_features

        # Embedding layer para reduzir a dimensionalidade
        self.embedding_dim = 50
        self.embedding_layer = nn.Linear(num_features, self.embedding_dim)

        # Camada final para classificação em n_classes classes
        self.classifier = nn.Linear(num_features, self.n_classes)

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
