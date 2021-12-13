#!/usr/bin/env python
# coding: utf-8

# # **SMART PHONES COST RECOGNITION AND PREDICTION MODEL**

# Objective
# 1) To predict price range of the mobile for test data
# 2) To check the accuracy of the classifiers Decision tree,Logistic Regression classifier,K- Nearest Neighbor, Random forest model, Decision tree and K nearest neighbor.

# # Importing Libraries

# 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# # Importing Datasets
# 
# 

# In[2]:


url = 'https://drive.google.com/file/d/1-h4tZ9r2JIdHZscuhipjgxFeuWoXsZ1f/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
data_train = pd.read_csv(path)


# In[3]:


url = 'https://drive.google.com/file/d/1lSuI1kNXZo7pzx3JJ6X5NvTCEgnBBLz5/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
data_test = pd.read_csv(path)


# # Descripation of Datasets:
# Dataset as 21 features and 2000 entries. The meanings of the features are given below.
# 
# battery_power: Total energy a battery can store in one time measured in mAh
# blue: Has bluetooth or not
# clock_speed: speed at which microprocessor executes instructions
# dual_sim: Has dual sim support or not
# fc: Front Camera mega pixels
# four_g: Has 4G or not
# int_memory: Internal Memory in Gigabytes
# m_dep: Mobile Depth in cm
# mobile_wt: Weight of mobile phone
# n_cores: Number of cores of processor
# pc: Primary Camera mega pixels
# px_height: Pixel Resolution Height
# px_width: Pixel Resolution Width
# ram: Random Access Memory in Mega Byte
# sc_h: Screen Height of mobile in cm
# sc_w: Screen Width of mobile in cm
# talk_time: longest time that a single battery charge will last when you are
# three_g: Has 3G or not
# touch_screen: Has touch screen or not
# wifi: Has wifi or not
# price_range: This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).

# Dataset with Price Range (Target Variable) 

# In[ ]:


data_test.head()


# # Exploratory Data Analysis

# In[ ]:


data_train.shape


# In[ ]:


data_test.shape


# In[ ]:


data_train.info()


# In[ ]:


data_test.info()


# In[ ]:


for col in data_train.columns:
    print("{} have {} unique values: ".format(col, data_train[col].nunique()))
print("*" * 35)
for col in data_train.columns:
    if data_train[col].nunique() <= 16:
        print("{}: {}".format(col, data_train[col].unique()))


# Checking Datasets to be Cleaned (checking null values in data)

# In[ ]:


data_train.isnull().sum()


# In[ ]:


data_test.isnull().sum()


# In[ ]:


data_train.describe()


# # Data Visualization

# 1) Correlation between features
# 
# 

# In[ ]:


plt.figure(figsize=(8,8))
sb.heatmap(data_train.corr())


# In[ ]:


corr = data_train.corr()
g = sb.heatmap(corr, vmax=.3, center=0,
square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
sb.despine()
g.figure.set_size_inches(25,9)
plt.show()


# As we can see that target price range has highly positive correlation between ram. And also battery power have good realtion,4G and 3G have good relation,px_height and px_width have good relation.

# In[ ]:


a = ((data_train['price_range'] == 0).sum() / data_train['price_range'].count() * 100), ((data_train['price_range'] == 1).sum() / data_train['price_range'].count() * 100), ((data_train['price_range'] == 2).sum() / data_train['price_range'].count() * 100), ((data_train['price_range'] == 3).sum() / data_train['price_range'].count() * 100)
font1 = {'family':'serif','color':'black','size':20}
# Creating plot
fig = plt.figure(figsize =(10, 7))
plt.pie(a, labels = data_train['price_range'].unique(),explode = [0, 0, 0, 0], autopct='%1.0f%%')
plt.title("Distribution of Mobile Price Range", fontdict = font1)
plt.show()


# # Visualizing Correlated Data

# In[ ]:


plt.figure(figsize=[16, 6])
sb.lineplot(x="price_range", y="ram", data=data_train, hue='battery_power', marker = '*', ms=13, ci=None)
plt.show()


# In[4]:


import plotly.express as px
fig = px.scatter_3d(data_train.head(1000), x='ram', y='battery_power', z='px_width', color='price_range')
fig.show()


# In[ ]:


plt.figure(figsize=(17,6))
sb.set_style("whitegrid")
plt.title('ram is the best featture to separate the price ranges')
sb.kdeplot(data=data_train, x='ram',hue='price_range', shade=True)


# 2) Histograms

# In[5]:


col=["battery_power","clock_speed","int_memory","mobile_wt"]
plt.figure(figsize=(10,10))
i=1
k=0
for j in col:
    if i!=5:
        plt.subplot(2,2,i)
        sb.histplot(data_train[col[k]])
        k=k+1
    else:
        break
    i=i+1


# 3) CountPlot

# In[ ]:


sb.countplot(data_train['price_range'])


# In[6]:


col=['blue',"dual_sim","three_g","four_g","touch_screen",'wifi']
plt.figure(figsize=(16,8))
i=1
k=0
for j in col:
    if i!=7:
        plt.subplot(2,3,i)
        sb.countplot(data_train[col[k]])
        k=k+1
    else:
        break
    i=i+1


# 4) Scatter plot

# In[ ]:


data_train.plot(x='price_range',y='ram',kind='scatter')
plt.show()


# In[ ]:


data_train.plot(x='price_range',y='mobile_wt',kind='scatter')
plt.show()


# In[ ]:


data_train.plot(x='price_range',y='fc',kind='scatter')
plt.show()


# In[ ]:


data_train.plot(x='price_range',y='n_cores',kind='scatter')
plt.show()


# 5) Pie Plot for 4G and 3G phones

# In[ ]:


plt.subplot=(1,2,1)
values=data_train['three_g'].value_counts().values
labels=["3G","Non 3G"]
col=["pink","blue"]
plt.pie(values,labels=labels,colors=col,autopct='%1.1f%%')
plt.show()

plt.subplot=(1,2,2)
values=data_train['four_g'].value_counts().values
labels=["4G","Non 4G"]
col=["pink","blue"]
plt.pie(values,labels=labels,colors=col,autopct='%1.1f%%')
plt.show()


# In[ ]:


n_cores = data_train['n_cores'].value_counts()
plt.title('Number of cores in mobile phones\n\n', weight='bold')
n_cores.plot.pie(autopct="%.1f%%", radius=1.5)
plt.show()


# 6) Bar Plot

# In[ ]:


sb.barplot(data_train['dual_sim'],data_train['price_range'])


# In[ ]:


sb.barplot(data_train['wifi'],data_train['price_range'])


# In[ ]:


sb.barplot(x='price_range',y='px_height',data=data_train,palette="Reds")
plt.show()


# In[ ]:


sb.barplot(x="price_range",y='px_width',data=data_train,palette='Blues')


# 7) Box Plot

# In[ ]:


sb.boxplot(data_train['price_range'],data_train['battery_power'])


# In[ ]:


data_train.plot(kind='box',figsize=(20,10))
plt.show()


# In[ ]:


pip install plotly


# In[5]:


import plotly.express as px
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.offline as py
import plotly.figure_factory as ff


# In[6]:


px.box(data_train,color="price_range")


# In[7]:


px.box(data_train,x ="ram" ,color="price_range")


# In[ ]:


matrix =np.triu(data_train.corr())
fig,ax =plt.subplots(figsize=(12,6),dpi=150)
sb.heatmap(data_train.corr(),vmax=1,vmin=-1,center=0,annot=True,fmt=".2f",mask=matrix,ax=ax,cmap="rainbow");


#  Configuring Data

# In[8]:


X=data_train.drop('price_range',axis=1)


# In[14]:


X


# In[ ]:


data_test.shape


# In[15]:


data_test=data_test.drop('id',axis=1)


# In[16]:


data_test.head()


# In[ ]:


data_test.shape


# In[9]:


Y=data_train['price_range']


# In[18]:


Y.unique()


# # **Training The Model**

# Splitting data into dependent variable and independent variable

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[12]:


X_train


# In[ ]:


Y_train


# In[ ]:


X_test


# In[ ]:


Y_test


# # 1) Decision Tree

# In[13]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[14]:


dt.fit(X_train,Y_train)


# In[15]:


Y_pred=dt.predict(X_test)


# In[16]:


Y_pred


# In[18]:


from sklearn.metrics import accuracy_score


# In[19]:


dt_ac=accuracy_score(Y_test,Y_pred)


# In[20]:


dt_ac


# In[21]:


from sklearn.metrics import confusion_matrix
confusion_dt=confusion_matrix(Y_test,Y_pred)

#confusion matrix plot
plot=sb.heatmap(confusion_dt,square=True,annot=True)
class_lables=['0','1','2','3']
plot.set_xlabel('Actual values')
plot.set_ylabel('Predicted values')


# In[22]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, hamming_loss, balanced_accuracy_score, cohen_kappa_score, explained_variance_score, r2_score, mean_absolute_error
Accuracy_dt=accuracy_score(Y_test,Y_pred)
Precision_dt=precision_score(Y_test,Y_pred,average='weighted')
Recall_dt=recall_score(Y_test,Y_pred,average='weighted')
f1_score_dt=f1_score(Y_test,Y_pred, average='weighted')
fbeta_score_dt= fbeta_score(Y_test,Y_pred, average='weighted', beta=0.5)
hamming_loss_dt= hamming_loss(Y_test,Y_pred)
balanced_accuracy_score_dt= balanced_accuracy_score(Y_test,Y_pred)
cohen_kappa_score_dt= cohen_kappa_score(Y_test,Y_pred)
explained_variance_score_dt= explained_variance_score(Y_test,Y_pred, multioutput='variance_weighted')
r2_score_dt= r2_score(Y_test,Y_pred, multioutput='variance_weighted')
mean_absolute_error_dt= mean_absolute_error(Y_test,Y_pred)


# In[32]:


from sklearn.model_selection import cross_val_score


# In[33]:


print(f'Cross Validation Scores: ' + str(cross_val_score(dt, X_train,Y_train, cv=5)))

print(f'Cross Validation Score (Mean): ' + str(np.mean(cross_val_score(dt, X_train,Y_train, cv=5))))


# In[24]:


from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1', 'class 2', 'class 3']
print(classification_report(Y_test,Y_pred, target_names=target_names))


# In[35]:


from sklearn.datasets import make_classification


# In[36]:


feats = pd.DataFrame(index=X.columns,data=dt.feature_importances_,columns=['Importance'])
imp_feats = feats.sort_values("Importance")
plt.figure(figsize=(12,6))
sb.barplot(data=imp_feats.sort_values('Importance'), x=imp_feats.sort_values('Importance').index, y='Importance')

plt.xticks(rotation=90);


# # 2) KNN

# In[39]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[40]:


knn.fit(X_train,Y_train)


# In[41]:


Y_pred_knn=knn.predict(X_test)


# In[42]:


Y_pred_knn


# In[43]:


knn_ac=accuracy_score(Y_test,Y_pred_knn)


# In[44]:


knn_ac


# In[45]:


from sklearn.metrics import confusion_matrix
confusion_knn=confusion_matrix(Y_test,Y_pred_knn)
#confusion matrix plot
plot=sb.heatmap(confusion_knn,square=True,annot=True)
class_lables=['0','1','2','3']
plot.set_xlabel('Actual values')
plot.set_ylabel('Predicted values')


# In[46]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, hamming_loss, balanced_accuracy_score, cohen_kappa_score, explained_variance_score, r2_score, mean_absolute_error
Accuracy_knn=accuracy_score(Y_test,Y_pred_knn)
Precision_knn=precision_score(Y_test,Y_pred_knn,average='weighted')
Recall_knn=recall_score(Y_test,Y_pred_knn,average='weighted')
f1_score_knn=f1_score(Y_test,Y_pred, average='weighted')
fbeta_score_knn= fbeta_score(Y_test,Y_pred, average='weighted', beta=0.5)
hamming_loss_knn= hamming_loss(Y_test,Y_pred)
balanced_accuracy_score_knn= balanced_accuracy_score(Y_test,Y_pred)
cohen_kappa_score_knn= cohen_kappa_score(Y_test,Y_pred)
explained_variance_score_knn= explained_variance_score(Y_test,Y_pred, multioutput='variance_weighted')
r2_score_knn= r2_score(Y_test,Y_pred, multioutput='variance_weighted')
mean_absolute_error_knn= mean_absolute_error(Y_test,Y_pred)


# In[ ]:


print(f'Cross Validation Scores: ' + str(cross_val_score(knn, X_train,Y_train, cv=5)))

print(f'Cross Validation Score (Mean): ' + str(np.mean(cross_val_score(knn, X_train,Y_train, cv=5))))


# # 3) Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[ ]:


lr.fit(X_train,Y_train)


# In[ ]:


Y_pred_lr=lr.predict(X_test)


# In[ ]:


Y_pred_lr


# In[ ]:


lr_ac=accuracy_score(Y_test,Y_pred_lr)


# In[ ]:


lr_ac


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_lr=confusion_matrix(Y_test,Y_pred_lr)
#confusion matrix plot
plot=sb.heatmap(confusion_lr,square=True,annot=True)
class_lables=['0','1','2','3']
plot.set_xlabel('Actual values')
plot.set_ylabel('Predicted values')


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, hamming_loss, balanced_accuracy_score, cohen_kappa_score, explained_variance_score, r2_score, mean_absolute_error
Accuracy_lr=accuracy_score(Y_test,Y_pred_lr)
Precision_lr=precision_score(Y_test,Y_pred_lr,average='weighted')
Recall_lr=recall_score(Y_test,Y_pred_lr,average='weighted')
f1_score_lr=f1_score(Y_test,Y_pred, average='weighted')
fbeta_score_lr= fbeta_score(Y_test,Y_pred, average='weighted', beta=0.5)
hamming_loss_lr= hamming_loss(Y_test,Y_pred)
balanced_accuracy_score_lr= balanced_accuracy_score(Y_test,Y_pred)
cohen_kappa_score_lr= cohen_kappa_score(Y_test,Y_pred)
explained_variance_score_lr= explained_variance_score(Y_test,Y_pred, multioutput='variance_weighted')
r2_score_lr= r2_score(Y_test,Y_pred, multioutput='variance_weighted')
mean_absolute_error_lr= mean_absolute_error(Y_test,Y_pred)


# In[ ]:


print(f'Cross Validation Scores: ' + str(cross_val_score(lr, X_train,Y_train, cv=5)))

print(f'Cross Validation Score (Mean): ' + str(np.mean(cross_val_score(lr, X_train,Y_train, cv=5))))


# # 4) Random forest model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
model = RandomForestRegressor() 
hyp = RandomizedSearchCV(estimator = model, param_distributions=grid, n_iter=10, scoring= 'neg_mean_squared_error', 
                         cv=5,verbose = 2, random_state = 42 ,n_jobs = 1) 
hyp.fit(X_train,Y_train)


# In[ ]:


y_pred = hyp.predict(X_test)
y_pred


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_rf=confusion_matrix(Y_test,y_pred_rf)
#confusion matrix plot
plot=sb.heatmap(confusion_rf,square=True,annot=True)
class_lables=['0','1','2','3']
plot.set_xlabel('Actual values')
plot.set_ylabel('Predicted values')


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, hamming_loss, balanced_accuracy_score, cohen_kappa_score, explained_variance_score, r2_score, mean_absolute_error
Accuracy_rf=accuracy_score(Y_test,y_pred_rf)
Precision_rf=precision_score(Y_test,y_pred_rf,average='weighted')
Recall_rf=recall_score(Y_test,y_pred_rf,average='weighted')
f1_score_rf=f1_score(Y_test,Y_pred, average='weighted')
fbeta_score_rf= fbeta_score(Y_test,Y_pred, average='weighted', beta=0.5)
hamming_loss_rf= hamming_loss(Y_test,Y_pred)
balanced_accuracy_score_rf= balanced_accuracy_score(Y_test,Y_pred)
cohen_kappa_score_rf= cohen_kappa_score(Y_test,Y_pred)
explained_variance_score_rf= explained_variance_score(Y_test,Y_pred, multioutput='variance_weighted')
r2_score_rf= r2_score(Y_test,Y_pred, multioutput='variance_weighted')
mean_absolute_error_rf= mean_absolute_error(Y_test,Y_pred)


# In[ ]:


rf_ac=accuracy_score(Y_test,y_pred_rf)


# In[ ]:


rf_ac


# In[ ]:


print(f'Cross Validation Scores: ' + str(cross_val_score(hyp, X_train,Y_train, cv=5)))

print(f'Cross Validation Score (Mean): ' + str(np.mean(cross_val_score(hyp, X_train,Y_train, cv=5))))


# # Graphical View (comparision of model)

# In[ ]:


models=pd.DataFrame({'Model':["Decision Tree","KNN","Logistic Regression","Random forest"],
                     'Accuracy':[Accuracy_dt,Accuracy_knn,Accuracy_lr,Accuracy_rf],
                    "precision":[Precision_dt,Precision_knn,Precision_lr,Precision_rf],
                    'Recall':[Recall_dt,Recall_knn,Recall_lr,Recall_rf],
                    "f1_score":[f1_score_dt,f1_score_knn,f1_score_lr,f1_score_rf],
                    "fbeta_score":[fbeta_score_dt, fbeta_score_knn, fbeta_score_lr, fbeta_score_rf],
                    "hamming_loss":[hamming_loss_dt,hamming_loss_knn,hamming_loss_lr,hamming_loss_rf],
                    "r2_score":[r2_score_dt,r2_score_knn,r2_score_lr,r2_score_rf],
                    "balanced_accuracy_score":[balanced_accuracy_score_dt,balanced_accuracy_score_knn,balanced_accuracy_score_lr,balanced_accuracy_score_rf],
                    "cohen_kappa_score":[cohen_kappa_score_dt,cohen_kappa_score_knn,cohen_kappa_score_lr,cohen_kappa_score_rf],
                    "explained_variance_score":[explained_variance_score_dt,explained_variance_score_knn,explained_variance_score_lr,explained_variance_score_rf],
                    "mean_absolute_error":[mean_absolute_error_dt,mean_absolute_error_knn,mean_absolute_error_lr,mean_absolute_error_rf]})
                   
models


# In[ ]:


plt.bar(x=['dt','knn','lr','rf'],height=[dt_ac,knn_ac,lr_ac,rf_ac])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.show()


# In[ ]:


accuracy=models['Accuracy'].values
precision=models['precision'].values
Recall=models['Recall'].values
model=["Decision Tree","KNN","Logistic Regression","Random forest"]

x_axis=np.arange(len(model))
plt.figure(figsize=(6,9))
plt.bar(x_axis-0.2,accuracy,width=0.15,label="Accuracy")
plt.bar(x_axis-0.05,precision,width=0.15,label="Precision")
plt.bar(x_axis+0.1,Recall,width=0.15,label="Recall")
plt.xlabel("Models")
plt.ylabel("Accuracy/precision/Recall")
plt.legend()
plt.show()


# # KNN classifier got highest accuracy
# # Logistic Regression classifier got lowest accuracy
