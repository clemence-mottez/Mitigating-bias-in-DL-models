# INVESTIGATING AND MITIGATING BIAS IN DEEP LEARNING MODELS FOR CLINICAL APPLICATIONS

Machine learning (ML) and Deep learning (DL) models are transforming healthcare by improving
diagnostic accuracy, personalizing treatments, and enhancing patient outcomes.   
However, a critical concern is the potential for these models to exacerbate healthcare disparities, as demonstrated in previous studies. If a model
performs differently across patient subgroups (e.g., by sex, age, race or health insurance status), it may pose safety risks,
as it could unfairly disadvantage certain populations.  
Tackling these challenges is crucial to ensure the fair and safe implementation of ML and DL in healthcare.  
  
Our motivation is to identify, quantify, and mitigate biases in well-known deep-learning models used for disease predictions from x-ray images. 


# How to run the project

It is needed to run this project with a good GPU. 

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

First download the data from the Chexpert official website and put the files "train.csv", "test.csv" and "df_chexpert_plus_240401.csv" it in data/initial_data folder.  
  
Then run: 

1. Clean tabular data to remove outliers and subjects that will not be used
    ```bash
    /data/data_processing/clean_initial_tabular_data.ipynb
    ```
    This will create a file "processed_data/chexpert_plus_240401_cleaned.csv"


2. Label data and remove useless information
    ```bash
    /data/data_processing/label_and_reduce_data.ipynb
    ```
    This will create a file "final_data/chexpert_plus_240401_cleaned_label.csv"
      
    "sex": {"Male": 0, "Female": 1},  
    "race": {"White": 0, "Asian": 1, "Black": 2},  
    "insurance_type": {"Medicaid": 0, "Medicare": 1, "Private Insurance": 2}  
      
  
3. Create train (60%) / valid (10%) /test split (30%)
    ```bash
    /data/data_processing/create_train_valid_test_split.ipynb
    ```
    This will create files "final_data/chexpert_plus_240401_train.csv", "final_data/chexpert_plus_240401_test.csv", "final_data/chexpert_plus_240401_valid.csv"

4. Data can be explored in
    ```bash
    /data/data_exploration/explore_tabular_data.ipynb
    ```
    
## Image data

1. Download all the images from "huggingface.co/datasets/StanfordAIMI/CheXpert-v1.0-512"
    ```bash
    /data/image_data_processing/download_image_data.ipynb
    ```
    
2. Resize images in (224, 224), add path to resized images to table
    ```bash
    /data/image_data_processing/preprocess_image_data.ipynb
    ```

## Extract embeddings 
We are extracting the embeddings using the densenet and chexpert models in the torch x-ray vision github. 
You need to download a zip file containing the weights to use the chexpert model. 

1. Densenet embeddings
    ```bash
    /model_dev/pre_trained_densenet_GPU_embeddings.py
    ```
This will create "data_with_embeddings/densenet_test_embeddings.csv", "data_with_embeddings/densenet_train_embeddings.csv", "data_with_embeddings/densenet_valid_embeddings.csv"

2. Chexpert embeddings
    ```bash
    /model_dev/pre_trained_chexpert_GPU_embeddings.py
    ```
This will create "data_with_embeddings/chexpert3_test_embeddings.csv", "data_with_embeddings/chexpert3_train_embeddings.csv", "data_with_embeddings/chexpert3_valid_embeddings.csv"

## Disease predictions from Densenet embeddings, model evaluation, bias detection

1. Prediction using Multiclass Neural Network
    ```bash
    /model_eval/densenet_from_embeddings_to_pred_nn.ipynb
    ```

2. Prediction using Multiclass XGBoost
    ```bash
    /model_eval/densenet_from_embeddings_to_pred_xgboost.ipynb
    ```







