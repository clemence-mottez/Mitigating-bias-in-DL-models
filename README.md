# Mitigate Bias Project

## Overview
This project aims to address and mitigate bias in deep learning models using CheXpert database.  
It is recommended to run this project with a GPU. 


## Creating a Conda Environment

To create a Conda environment for this project, run the following command in your terminal:

```bash
conda create --name mitigate-bias python=3.8.19
```
To open the environment, run the following command in your terminal:

```bash
conda activate mitigate-bias
```

## Preprocess Data

First download the data from the Chexpert official website.  

Perform only once

1. Clean tabular data to remove outliers and subjects that will not be used
    ```bash
    /data/clean_tabular_data.ipynb
    ```
2. Add path to resized images (224,224) to table
    ```bash
    /data/preprocess_image_data.ipynb
    ```
3. Add text/tabular prompts to table
    * Prompt as sentence:  "The subject is a [age]-year-old [sex] identifying as [race] with [insurance_type] insurance coverage."
    * Prompt as dict (str): "Age: [age], [sex]: female, race: [race], insurance: [insurance_type]"
    ```bash
    /data/preprocess_tabular_data.ipynb
    ```
4. Create train (60%) /valid (10%) /test split (30%)
    ```bash
    /data/create_train_valid_test_split.ipynb
    ```


## Labels in the data
  
"sex": {"Male": 0, "Female": 1},  
"race": {"White": 0, "Asian": 1, "Black": 2},  
"insurance_type": {"Medicaid": 0, "Medicare": 1, "Private Insurance": 2}  

## Image data
Add info on how to download images and resize them

## Extract embeddings with torch x-ray vision
How to download the weights we used for Chexpert  
Where to find the models  
Copy right github  

## Study embeddings  




