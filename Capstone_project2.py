import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import textwrap

sns.set()

file_path = "/Users/laurensteppe/Desktop/python/obesity_data.csv"
data = pd.read_csv(file_path)

data.head()

data.describe(include = 'all')

data.drop(['YearEnd', 'Data_Value_Unit'], axis = 1, inplace = True)
data.head()

data.isnull().sum()

datanew = data.dropna(subset = ['GeoLocation', 'Sample_Size', 'Low_Confidence_Limit', 'High_Confidence_Limit ', 'Data_Value', 'Data_Value_Alt']).copy()
datanew.drop(['Class','Datasource','LocationAbbr','Data_Value_Footnote_Symbol', 'Data_Value_Footnote','Total','Age(years)', 'Education', 'Gender','Income','Race/Ethnicity'], axis = 1, inplace = True)
datanew.isnull().sum()

datanew.describe(include = 'all')

datanew.reset_index(drop = True, inplace= True)
selected_columns = datanew[['Question', 'Data_Value', 'StratificationCategory1', 'Stratification1']]
filtered_df = selected_columns[selected_columns['Question'].str.contains('activit', case=False, regex=True)]
filter2 = filtered_df[filtered_df['StratificationCategory1'] == 'Income']
filter3 = filter2[filter2['Stratification1'] != 'Data not reported']
grouped_data = filter3.groupby(['Question', 'Stratification1'])['Data_Value'].mean().reset_index()
grouped_data.sort_values('Stratification1', inplace=True)
print(grouped_data)
print("Total number of rows in the dataset:", len(filter3))

#Visualizing the data
# Get a list of unique questions
unique_questions = grouped_data['Question'].unique()

# Consider using fewer rows per figure if there are many unique questions
n_rows = 2
n_cols = 3

# Slightly larger figure size
plt.figure(figsize=(12, 8))  # Adjusted for better spacing

for i, question in enumerate(unique_questions):
    ax = plt.subplot(n_rows, n_cols, i + 1)
    data_to_plot = grouped_data[grouped_data['Question'] == question]
    sns.barplot(x='Stratification1', y='Data_Value', hue='Stratification1', data=data_to_plot, ax=ax, palette='Set3', legend=False)
    wrapped_title = textwrap.fill(question, width=50)
    ax.set_title(wrapped_title, fontsize=9, pad=10)
    ax.set_xlabel('Income Group', fontsize=9)
    ax.set_ylabel('Average Data Value as %', fontsize=9)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.tick_params(axis='x', rotation=75, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
plt.show()
