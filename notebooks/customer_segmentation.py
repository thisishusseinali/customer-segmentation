import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans

customers_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Customer Segmentation/customers_data.csv')

"""# Data Exploration And Analysis"""

print(f'DATA SHAPE : [{customers_data.shape[0]} ROWS ,{customers_data.shape[1]} COLUMNS ]')

customers_data.head()

customers_data.info()

customers_data.isnull().sum()

customers_data.describe()

customers_data['Gender'].value_counts()

Gender_Values = [customers_data['Gender'].value_counts()[1],customers_data['Gender'].value_counts()[0]]
Gender_Labels = ['Male','Female']

plt.figure(figsize=(10, 10))
plt.title("What's Our Customers Gender")

plt.pie(Gender_Values,labels =Gender_Labels,autopct='%.2f')
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0,0,1,1])
Gender_Values = [customers_data['Gender'].value_counts()[1],customers_data['Gender'].value_counts()[0]]
Gender_Labels = ['Male','Female']
ax.bar(Gender_Labels,Gender_Values)
plt.title('Using BarPlot to display Gender Comparision')
plt.legend()
plt.show()

print(f"Customer's Age Stats :\n(1) Mean : {customers_data['Age'].mean()}\n(2) Median : {customers_data['Age'].median()}\n(3) Mode : {customers_data['Age'].mode()}")

fig = plt.figure(figsize=(10,10))
plt.hist(customers_data['Age'])

plt.axvline(customers_data['Age'].mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(customers_data['Age'].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(customers_data['Age'].mean()))

plt.title('Using Hist  to display Customers Age')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,10))
sns.distplot(customers_data['Age'],hist=False)
plt.axvline(customers_data['Age'].mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(customers_data['Age'].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(customers_data['Age'].mean()))

plt.show()

fig = plt.figure(figsize=(10,10))
plt.hist(customers_data['Annual Income (k$)'])

plt.axvline(customers_data['Annual Income (k$)'].mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(customers_data['Annual Income (k$)'].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(customers_data['Annual Income (k$)'].mean()))

plt.title('Using Hist  to display Customers Annual Income')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,10))
sns.distplot(customers_data['Annual Income (k$)'],hist=False)
plt.axvline(customers_data['Annual Income (k$)'].mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(customers_data['Annual Income (k$)'].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(customers_data['Annual Income (k$)'].mean()))

plt.show()

fig = plt.figure(figsize=(10,10))
plt.hist(customers_data['Spending Score (1-100)'])

plt.axvline(customers_data['Spending Score (1-100)'].mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(customers_data['Spending Score (1-100)'].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(customers_data['Spending Score (1-100)'].mean()))

plt.title('Using Hist  to display Customers Spending Score')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,10))
sns.distplot(customers_data['Spending Score (1-100)'],hist=False)
plt.axvline(customers_data['Spending Score (1-100)'].mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(customers_data['Spending Score (1-100)'].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(customers_data['Spending Score (1-100)'].mean()))

plt.show()





Age = customers_data['Age']
Annual_Income = customers_data['Annual Income (k$)']
Spending_Score = customers_data['Spending Score (1-100)']
data = [Age, Annual_Income,Spending_Score]
 
fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
 
# Creating axes instance
bp = ax.boxplot(data, patch_artist = True,
                notch ='True', vert = 0)
 
colors = ['#0000FF', '#00FF00',
          '#FFFF00', '#FF00FF']
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)
 
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
ax.set_yticklabels(['Age','Annual_Income','Spending_Score'])
 
# Adding title
plt.title("Customized box plot")
 
# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
     
# show plot
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(customers_data['Age'],customers_data['Spending Score (1-100)'])
plt.scatter(customers_data['Age'],customers_data['Annual Income (k$)'])
plt.xlim(0,150)
plt.ylim(0,150)
plt.show()



X = customers_data.drop(['CustomerID','Gender'],axis=1)

kmeans = KMeans(3)
kmeans.fit(X)

identified_clusters = kmeans.fit_predict(X)
identified_clusters

data_with_clusters = X.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(X['Annual Income (k$)'],X['Spending Score (1-100)'],c=X['Age'],cmap='rainbow')

from joblib import dump, load
dump(kmeans,'kmeans')





