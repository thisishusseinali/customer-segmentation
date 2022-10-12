import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans

def model_creation():
  customers_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Customer Segmentation/customers_data.csv')
  Gender_Values = [customers_data['Gender'].value_counts()[1],customers_data['Gender'].value_counts()[0]]
  X = customers_data.drop(['CustomerID','Gender'],axis=1)
  kmeans = KMeans(3)
  kmeans.fit(X)
  identified_clusters = kmeans.fit_predict(X)
  data_with_clusters = X.copy()
  data_with_clusters['Clusters'] = identified_clusters 
  plt.scatter(X['Annual Income (k$)'],X['Spending Score (1-100)'],c=X['Age'],cmap='rainbow')
  from joblib import dump, load
  dump(kmeans,'kmeans')




