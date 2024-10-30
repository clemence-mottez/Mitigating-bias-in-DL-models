import os
import torch
import torchxrayvision as xrv
import numpy as np
from skimage.io import imread
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import csv
import pandas as pd
import skimage, torch, torchvision
import time
import ast

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def process_x_ray_densenet(image_folder, model):
    image_folder = '../../../lfay/CheXpert-v1.0-512/images/' + image_folder
    image_path = os.path.join(os.getcwd(), image_folder)

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return torch.empty(())  # Return array of 18 zeros if file not found
    else:
        try:
            img = skimage.io.imread(image_path)
            img = xrv.datasets.normalize(img, 255)  # Convert 8-bit image to [-1024, 1024] range
            img = img.mean(2)[None, ...]  # Make single color channel

            transform = torchvision.transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224),
            ])

            img = transform(img)
            img = torch.from_numpy(img).to(device)
            model.eval() 

            with torch.no_grad():  
                features = model.features(img)  
            features = features.detach().cpu().numpy()
            
            return features 

        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
            return torch.empty(())  
        

# Load your model and move it to GPU
model = xrv.baseline_models.chexpert.DenseNet(weights_zip="chexpert_weights.zip", num_models=3)
model.to(device)

# Load your CSV file
df = pd.read_csv('../data_processing/final_data/chexpert_plus_240401_test.csv')
# df = pd.read_csv('../data_processing/final_data/chexpert_plus_240401_cleaned_label_sample.csv')


start_time = time.time()  

# Apply the function to each image path
pred = df['path_to_image'].apply(lambda x: process_x_ray_densenet(x, model))

df_predictions = pd.DataFrame({'embeddings': pred.apply(lambda x: x.tolist())})

new_df = pd.concat([df.reset_index(drop=True), df_predictions.reset_index(drop=True)], axis=1)
new_df.to_csv('densenet_data/chexpert3_test_embeddings.csv', index=False)


elapsed_time = time.time() - start_time  
print(f"Total processing time: {elapsed_time:.2f} seconds")  # Print the elapsed time

