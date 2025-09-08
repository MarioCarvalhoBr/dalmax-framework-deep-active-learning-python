import os
import time
from PIL import Image


import numpy as np
import torch

from utils.LOGGER import get_logger, get_path_logger
from core.tools.VCTex.VCTexMethod import VCTexMethod

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold


# Global logger
logger = get_logger()
path_logger = get_path_logger()
def extract_features_from_folder(folder_path, device, Q):
    print(f"\n\nExtracting features from images in folder: {folder_path}")
    print(f"Using device: {device}")
    extractor = VCTexMethod(Q=Q, device=device)
    
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
                features = extractor(image)
                
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
    X, y = extract_features_from_folder(folder_path, device, Q=[5,17])
    end_time = time.time()
    print(f"Features of Dataset Extracted in {end_time - start_time:.2f} seconds")

    # Initialize LDA
    # Initialize LDA
    lda = LinearDiscriminantAnalysis()
    
    lda.fit(X, y)
   
    # folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outex")
    folder_path = "/home/aroeira/Desktop/CARVALHO/doutorado/projeto/quali/dalmax-deep-active-learning-python/DATA/daninhas_full/test"
    # Extract features and labels
    start_time = time.time()
    X_test, y_test = extract_features_from_folder(folder_path, device, Q=[5,17])
    end_time = time.time()
    print(f"Features of Dataset Extracted in {end_time - start_time:.2f} seconds")    

    # Perform classification
    y_pred = lda.predict(X_test)
    # Acuracy no teste
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()