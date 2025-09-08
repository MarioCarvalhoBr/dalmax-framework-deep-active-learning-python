import os
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import accuracy_score
from extractor import ColorFeatureExtractor
from PIL import Image
import time

def extract_features_from_folder(folder_path, Q):
    # Initialize ColorFeatureExtractor with Q
    extractor = ColorFeatureExtractor(Q=Q)
    
    features_list = []
    labels = []
    
    # Iterate through all images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Get class label from filename (before the underscore)
            label = filename.split('_')[0]
            
            # Load and process image
            image_path = os.path.join(folder_path, filename)
            image = np.array(Image.open(image_path))
            
            # Extract features
            features = extractor.extract(image)
            
            features_list.append(features)
            labels.append(label)
    
    return np.array(features_list), np.array(labels)

def main():
    # Path to your image folder
    folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outex")
    
    # Extract features and labels
    start_time = time.time()
    X, y = extract_features_from_folder(folder_path, Q=13)
    end_time = time.time()
    print(f"Features of Dataset Extracted in {end_time - start_time:.2f} seconds")

    # Initialize LDA
    lda = LinearDiscriminantAnalysis()
   
    print("Classification: LDA Initialized")
    #val = LeaveOneOut() #para reproduzir os resultados do paper (mais lento)
    val = KFold(n_splits=10, shuffle=True, random_state=42) #para testar mais rapido, resultados proximos com do paper
    
    # Perform cross validation and get scores
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(lda, X, y, cv=val)
    
    # Print results
    print(f"Average accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

if __name__ == "__main__":
    main()
