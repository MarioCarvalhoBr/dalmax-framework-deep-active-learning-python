import numpy as np

from extractor import ColorFeatureExtractor

# Method's hyperparameters.
Q=2            # The number of hidden neurons.

# Instantiate the color feature extractor.
extractor = ColorFeatureExtractor(Q=Q)

# A random generated image of size 128x128x3.
image = np.random.randint(low=0, high=256, size=(128, 128, 3))

# Extract the features from the given image.
features = extractor.extract(image)

print(f'Q: {Q}, \nFeatures Shape: {features.shape}.') 
print("\nFeatures:", features)