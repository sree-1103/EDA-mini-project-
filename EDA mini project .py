#!/usr/bin/env python
# coding: utf-8

# In[ ]:


PART - B


# In[ ]:


5


# In[19]:


data = pd.read_csv('~/Desktop/PGP-DSE/Mini Projects/EDA/project 2/project2/creditcard.csv')
print(data.head(10))


# In[ ]:


6


# In[11]:


print(data.describe())


# In[ ]:


7


# In[ ]:





# In[ ]:


8


# In[12]:


import matplotlib.pyplot as plt

# Count frauds
frauds = data['Class'].value_counts()
print(f'Number of Frauds: {frauds[1]}')

# Pie chart
labels = ['Legit', 'Fraud']
sizes = [frauds[0], frauds[1]]
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Fraud vs Legit Transactions')
plt.show()

# Bar plot
frauds.plot(kind='bar', legend=False, figsize=(10,5))
plt.title('Distribution of Fraudulent Transactions')
plt.ylabel('Count')
plt.show()


# In[13]:


fraud_transactions = data[data['Class'] == 1]
print(f'Minimum Fraudulent Amount: {fraud_transactions["Amount"].min()}')
print(f'Maximum Fraudulent Amount: {fraud_transactions["Amount"].max()}')


# In[ ]:


PART - A


# In[38]:


import pandas as pd
import numpy as np
from scipy import stats 
# Create a pandas dataframe
df = pd.DataFrame({'CEO Ages': [57, 61, 57, 57, 58, 57, 61, 54, 68, 51, 49, 64, 50, 48, 65, 52, 56, 46, 54, 49, 50, 47, 55, 55, 54, 42, 51, 56, 55, 54, 51, 60, 62, 43, 55, 56, 61, 52, 69, 64, 46, 54]})

# Mean
mean = df['CEO Ages'].mean()

# Median
median = df['CEO Ages'].median()


# Range
range_data = df['CEO Ages'].max() - df['CEO Ages'].min()

# Variance
variance = df['CEO Ages'].var()

# Standard Deviation
std_dev = df['CEO Ages'].std()


print(f"Mean: {mean}, Median: {median}")
print(f"Range: {range_data}, Variance: {variance}, Standard Deviation: {std_dev}")


# In[ ]:


5


# In[35]:


# Count number of values within two standard deviations
count = df['CEO Ages'].between(mean - 2*std_dev, mean + 2*std_dev).sum()

print(f"Number of values within two standard deviations: {count}")


# In[ ]:


6


# In[36]:


# Quartiles
q1 = df['CEO Ages'].quantile(0.25)
q2 = df['CEO Ages'].quantile(0.5)
q3 = df['CEO Ages'].quantile(0.75)

# Interquartile Range
iqr = q3 - q1

print(f"Q1: {q1}, Q2 (median): {q2}, Q3: {q3}. Interquartile Range (IQR): {iqr}")


# In[ ]:


7


# In[46]:


import matplotlib.pyplot as plt
import numpy as np

ages = [57, 61, 57, 57, 58, 57, 61, 54, 68, 51, 49, 64, 50, 48, 65, 52, 56, 46, 54, 49, 50, 47, 55, 55, 54, 42, 51, 56, 55, 54, 51, 60, 62, 43, 55, 56, 61, 52, 69, 64, 46, 54]

Q1 = np.percentile(ages, 25)
Q3 = np.percentile(ages, 75)
IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

outliers = []
for age in ages:
    if age < lower_bound or age > upper_bound:
        outliers.append(age)

print("Outliers:", outliers)

# Create a boxplot to visualize the outliers
plt.boxplot(ages)
plt.show()


# In[ ]:


8


# In[45]:


import matplotlib.pyplot as plt

# Replace these with the actual data you have
data = [57, 61, 57, 57, 58, 57, 61, 54, 68, 51, 49, 64, 50, 48, 65, 52, 56, 46, 54, 49, 50, 47, 55, 55, 54, 42, 51, 56, 55, 54, 51, 60, 62, 43, 55, 56, 61, 52, 69, 64, 46, 54]

plt.boxplot(data)
plt.title('Boxplot of Ages of CEOs when they took over')
plt.xlabel('Dataset')
plt.ylabel('Age')
plt.show()


# In[ ]:




