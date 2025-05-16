import os
import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

# Fetch the dataset
lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Create a directory to save images
os.makedirs("lfw_images", exist_ok=True)

# Save the images
for i, image in enumerate(lfw_dataset.images):
    plt.imsave(f'lfw_images/image_{i}.png', image, cmap='gray')

print(f"Saved {len(lfw_dataset.images)} images to the 'lfw_images' directory.")
