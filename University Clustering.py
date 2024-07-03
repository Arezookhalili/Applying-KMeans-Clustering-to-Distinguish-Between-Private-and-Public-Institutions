# Import required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


## Get the Data

df = pd.read_csv('College_Data',index_col=0)

### Check the dataset

df.head()

df.info()

df.describe()

## EDA

### Creating a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column

sns.set_style('whitegrid')
sns.lmplot(x='Room.Board',y='Grad.Rate',data=df, hue='Private',
           palette='coolwarm',height=6,aspect=1,fit_reg=False)


## Creating a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column

sns.set_style('whitegrid')
sns.lmplot(x='Outstate',y='F.Undergrad',data=df, hue='Private',
           palette='coolwarm',height=6,aspect=1,fit_reg=False)

## Finding if there is a school with a graduation rate of higher than 100%. What is the name of that school?

df[df['Grad.Rate']>100]

## Setting that school's graduation rate to 100 so it makes sense.

df['Grad.Rate']['Cazenovia College'] = 100

df.loc['Cazenovia College']

df[df['Grad.Rate']>100]

## K Means Cluster Creation

### Creating an instance of a K Means model with 2 clusters

kmeans = KMeans(n_clusters=2)

### Creating input variable

X = df.drop('Private', axis = 1)

### Fitting the model to all the data except for the Private label

kmeans.fit(X)

### What are the cluster center vectors?

kmeans.cluster_centers_

## Evaluation

### Creating a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)

df.head()

### Creating a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))