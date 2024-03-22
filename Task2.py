#Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading data
df_train = pd.read_csv("train.csv")
df_train.info()
print(df_train.isnull().sum())									#checking for any null values in the dataset
print(df_train)


df_train.dropna(subset=["Embarked"], inplace=True)				#removing rows with NaN values from Embarked cloumn
df_train["Cabin"].fillna("Unknown", inplace=True)				#replacing NaN with Unknown
df_train["Age"].fillna(df_train["Age"].mean(), inplace=True)	#replacing NaN with mean value of Age 

print(df_train)
print(df_train.isnull().sum())
print(df_train.duplicated().sum())

#plotting Histogram
plt.figure(figsize=(8, 4))
sns.histplot(df_train["Age"], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

#plotting Bar Graph 
plt.figure(figsize=(8, 4))
sns.countplot(data=df_train, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", loc="upper right")
plt.show()

#plotting Scatter plot 
plt.figure(figsize=(8, 4))
sns.scatterplot(data=df_train, x="Age", y="Fare", hue="Survived")
plt.title("Scatter Plot of Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Survived")
plt.show()
