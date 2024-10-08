# Mitigate Bias Project

## Overview
This project aims to address and mitigate bias in deep learning models using CheXpert database.

## Creating a Conda Environment

To create a Conda environment for this project, run the following command in your terminal:

```bash
conda create --name mitigate-bias python=3.8.19
```
To open the environment, run the following command in your terminal:

```bash
conda activate mitigate-bias
```

## Labels in the data
  
"sex": {"Male": 0, "Female": 1},  
"race": {"White": 0, "Asian": 1, "Black": 2},  
"insurance_type": {"Medicaid": 0, "Medicare": 1, "Private Insurance": 2}  