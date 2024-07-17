
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

from category_encoders import BinaryEncoder
df_titanic=pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\MACHINE LEARNING\06 Machine Learning_Sanjay Sane\Repository\Titanic-Dataset.csv")

# eligible columns for Binary Encoding
eligible_cols = ['Sex', 'Survived']

# Create a Binary Encoder
encoder = BinaryEncoder(cols=eligible_cols)

# Fit and transform the eligible columns
titanic_encoded = encoder.fit_transform(df_titanic)

# Print rows of the encoded dataset
titanic_encoded.head()