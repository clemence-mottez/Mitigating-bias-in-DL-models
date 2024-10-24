#%%

#importing important libaries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


#%%
df = pd.read_csv("data/densenet_data_valid_predictions.csv")

#%%
# delete rows where predicitons = 0,0,0,0
disease_columns = df.columns[df.columns.get_loc('Atelectasis.1'):]
df_without_0_rows = df[(df[disease_columns] != 0).any(axis=1)]

#%%
#don't include all the diseases where all pred = 0.5

df_cleanded_rows_columns = df_without_0_rows.drop(["Infiltration",'Emphysema','Fibrosis','Pleural_Thickening', 'Nodule', 'Mass', 'Hernia'], axis = 1)

df_cleanded_rows_columns.columns

#%%
# Transform probability into 0 or 1 according to a threshold

threshold = 0.9

predictions_columns = df_cleanded_rows_columns.loc[:,'Atelectasis.1':]
df_cleanded_rows_columns.loc[:, 'Atelectasis.1':] = predictions_columns.applymap(lambda x: 1 if x >= threshold else 0)

#%%
# create a column no finding

df_cleanded_rows_columns['No Finding'] = (df_cleanded_rows_columns.loc[:, 'Atelectasis.1':] == 0).all(axis=1).astype(int)
no_finding_values = df_cleanded_rows_columns.pop("No Finding")
df_cleanded_rows_columns.insert(df_cleanded_rows_columns.columns.get_loc('Support Devices') + 1, "No Finding", no_finding_values)


# %%
# rename column Effusion to Effusion.1 and merge colum Pleural Effusion and Pleural Other and dropping these two colums

df_cleanded_rows_columns = df_cleanded_rows_columns.rename(columns = {'Effusion': "Effusion.1"})
df_cleanded_rows_columns['Effusion'] = df_cleanded_rows_columns.apply(lambda row: 1 if row['Pleural Effusion'] == 1 or row['Pleural Other'] == 1 else 0, axis=1)
cols = df_cleanded_rows_columns.columns.tolist()
pneumothorax_index = cols.index('Pneumothorax')
cols.insert(pneumothorax_index + 1, cols.pop(cols.index('Effusion')))
df_cleanded_rows_columns = df_cleanded_rows_columns[cols]
df_cleanded_rows_columns = df_cleanded_rows_columns.drop(columns=['Pleural Effusion', 'Pleural Other'])

#%% save table
df_cleanded_rows_columns.to_csv('table_processed.csv', index=False)
# %%


#%%
#Calculating the Accuracy and F1 (Recall Precision) for every disease

disease_columns = ['Effusion', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Pneumonia', 
                   'Fracture', 'Edema', 'Consolidation', 'Pneumothorax', 'Enlarged Cardiomediastinum']
accuracy_results = {}
metrics_results = []

for disease in disease_columns:
    true_labels = df_cleanded_rows_columns[disease]
    pred_labels = df_cleanded_rows_columns[f'{disease}.1']
    accuracy_results[disease] = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    metrics_results.append({
        'Disease': disease,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })


accuracy_df = pd.DataFrame(accuracy_results.items(), columns=["Disease", "Accuracy"])
metrics_df = pd.DataFrame(metrics_results)
print(accuracy_df)
print(metrics_df)

# %%

precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# %%

#Calculating the Accuracy and F1 (Recall Precision) for every disease in each category
def calculate_metrics_by_category(df_cleanded_rows_columns, category_column):
    category_metrics = []

    #Grouping of each category
    for category_value, group in df_cleanded_rows_columns.groupby(category_column):
        for disease in disease_columns:
            true_labels = group[disease]
            pred_labels = group[f'{disease}.1']
            
            # Calculating the metrics
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            recall = recall_score(true_labels, pred_labels, zero_division=0)
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            accuracy = accuracy_score(true_labels, pred_labels)
            
            # Saving the metrics
            category_metrics.append({
                'Category': category_column,
                'Category Value': category_value,
                'Disease': disease,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Accuracy': accuracy
            })
    
    return pd.DataFrame(category_metrics)

# Calculating the metrics by sex
metrics_by_sex = calculate_metrics_by_category(df_cleanded_rows_columns, 'sex')
print("Metrics by Sex:")
print(metrics_by_sex)

#%%
# Calculating the metrics by race
metrics_by_race = calculate_metrics_by_category(df_cleanded_rows_columns, 'race')

print("\nMetrics by Race:")
print(metrics_by_race)


# %%
# Calculating the metrics by insurance_type
metrics_by_insurance_type = calculate_metrics_by_category(df_cleanded_rows_columns, 'insurance_type')


print("\nMetrics by inusrance_type:")
print(metrics_by_insurance_type)


# %%
