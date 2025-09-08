import os
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import accuracy_score

from PIL import Image
import time

# Lista os arquivos da pasta atual e do contexto atual
files = os.listdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Files in current directory: {files}")

from VCTexMethod import VCTexMethod

def extract_features_from_folder(folder_path, device, Q):
    print(f"\n\nExtracting features from images in folder: {folder_path}")
    print(f"Using device: {device}")
    # Initialize VCTexMethod with Q=13 (best parameter from paper)
    extractor = VCTexMethod(Q=Q, device=device)
    
    features_list = []
    labels = []
    
    # Iterate through all images in the folder
    for folder in os.listdir(folder_path):
        print(f"\nProcessing folder: {folder}")
        images_files = os.listdir(os.path.join(folder_path, folder))
        for image_file in images_files:
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                print(f"Processing image: {image_file}")
                break

                # Load and process image
                image_path = os.path.join(folder_path, folder, image_file)
                image = np.array(Image.open(image_path))
                
                # Extract features
                features = extractor(image)
                
                features_list.append(features)
                labels.append(folder)
    
    
    exit()
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
