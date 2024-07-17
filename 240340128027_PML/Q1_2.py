import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# df_titanic 

df_titanic=pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\MACHINE LEARNING\06 Machine Learning_Sanjay Sane\Repository\Titanic-Dataset.csv")
 

df_titanic['Sex_Encoded'] = df_titanic['Sex'].map({'male': 0, 'female': 1})

print(df_titanic.isna().sum())

# calculate Median
print("after remove all null values ")
median_age = df_titanic['Age'].median()
df_titanic['Age'] = df_titanic['Age'].fillna(median_age)
print(df_titanic.isna().sum())



 