import torch
import numpy as np

from extractor import VCTex
from VCTexMethod import VCTexMethod

# Device.
device = torch.device("cpu")

Q = [2,4] #best parameters of the paper. You can test different vales 


# Create VCTex extractor with Q=9
ext = VCTexMethod(Q=Q, device=device)

# Create a sample image (128x128 pixels with 3 color channels)
image = np.random.randint(0, 255, size=(128, 128, 3))

# Extract features from the image
features = ext(image)

# Print information about the extracted features
print(f'Q: {Q}, \nFeatures Length: {len(features)}')
print("\nFeatures:", features)



