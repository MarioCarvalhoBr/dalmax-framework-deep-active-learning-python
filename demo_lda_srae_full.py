import os
import time
from PIL import Image


import numpy as np
import torch

from utils.LOGGER import get_logger, get_path_logger
from core.tools.SSRAE.extractor import ColorFeatureExtractor


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold

# Global logger
logger = get_logger()
path_logger = get_path_logger()
def extract_features_from_folder(folder_path, Q):
    print(f"\n\nExtracting features from images in folder: {folder_path}")
    extractor = ColorFeatureExtractor(Q=Q)
    
    features_list = []
    labels = []
    
    # Iterate through all images in the folder
    for folder in os.listdir(folder_path):
        print(f"\nProcessing folder: {folder}")
        images_files = os.listdir(os.path.join(folder_path, folder))
        for image_file in images_files:
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                

                # Load and process image
                image_path = os.path.join(folder_path, folder, image_file)
                image = np.array(Image.open(image_path))
                
                # Extract features
                features = extractor.extract(image)
                
                features_list.append(features)
                labels.append(folder)
    
    return np.array(features_list), np.array(labels)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path to your image folder
    # folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outex")
    folder_path = "/home/aroeira/Desktop/CARVALHO/doutorado/projeto/quali/dalmax-deep-active-learning-python/DATA/daninhas_full/train"
    # Extract features and labels
    start_time = time.time()
    X, y = extract_features_from_folder(folder_path, Q=13)
    end_time = time.time()
    print(f"Features of Dataset Extracted in {end_time - start_time:.2f} seconds")

    # folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outex")
    folder_path = "/home/aroeira/Desktop/CARVALHO/doutorado/projeto/quali/dalmax-deep-active-learning-python/DATA/daninhas_full/test"
    # Extract features and labels
    start_time = time.time()
    X_test, y_test = extract_features_from_folder(folder_path, Q=13)
    end_time = time.time()
    print(f"Features of Dataset Extracted in {end_time - start_time:.2f} seconds")
    
    # Extend X and y with test data
    X = np.vstack((X, X_test))
    y = np.concatenate((y, y_test))
    
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
