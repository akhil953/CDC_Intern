#!/usr/bin/env python
# coding: utf-8

# In[163]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cherrypy


# In[164]:


df=pd.read_csv("/home/seethalprince/Documents/Data1.csv",index_col=0)


# In[165]:


df=pd.read_csv("/home/seethalprince/Documents/Data1.csv")


# In[166]:


df.head


# In[167]:


df=pd.read_csv("/home/seethalprince/Documents/Data1.csv")


# In[168]:


df.head()


# In[169]:


df


# In[170]:


df.describe()


# In[171]:


sns.set_style('whitegrid')
sns.lmplot('Age','Heartbeat',data=df, hue='pressure',palette='coolwarm',size=6,aspect=1,fit_reg=False)


# In[172]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)


# In[173]:


#kmeans.fit(df)


# In[174]:


df.loc[(df['Heartbeat'] >=70) & (df['pressure'] >90 ), 'class'] = "yes"
df.loc[(df['Heartbeat'] >=60) & (df['Heartbeat'] <80 )&(df['pressure']>100)&(df['pressure']<=90), 'class'] = "No"
df.loc[(df['Heartbeat'] >=20) &(df['Heartbeat'] <60 )&(df['pressure']>90), 'class'] = "maybe"

                                  


# In[175]:


df


# In[176]:


x=df[['Name', 'Age', 'Heartbeat','Body temp' ,'blood level','pressure']].values


# In[177]:


x


# In[178]:


def heart(attack):
    if attack=='yes':
        return 0
    elif attack == 'maybe':
        return 1
    elif attack=='No':
        return 2


# In[179]:


df['class'] = df['class'].apply(heart)
print(df['class'])


# In[180]:


plt.scatter(df['Heartbeat'], df['Body temp'])
plt.show()


# In[181]:


sns.factorplot('Heartbeat', data=df, hue='class', size=5, kind='count')
plt.show()


# In[182]:


sns.pairplot(df, hue='class', size=2)
plt.show()
sns.factorplot('Heartbeat', data=df, hue='class', size=5, kind='count')
plt.show()


# In[183]:


from sklearn.preprocessing  import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[184]:


label_encoder=LabelEncoder()
df['Name']=label_encoder.fit_transform(df['Name'].astype(str))
df['Age']=label_encoder.fit_transform(df['Age'].astype(str))
df['Heartbeat']=label_encoder.fit_transform(df['Heartbeat'].astype(str))
df['blood level']=label_encoder.fit_transform(df['blood level'].astype(str))
df['pressure']=label_encoder.fit_transform(df['pressure'].astype(str))

#Age', 'Heartbeat','Body temp' ,'blood level','pressure'


# In[185]:


df.dtypes


# In[186]:


print (df.dtypes)


# In[187]:


df.head()


# In[188]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[189]:


data=pd.read_csv("/home/seethalprince/Documents/Data1.csv")
data.head()
df=pd.DataFrame(data)
print(df)
print (df.shape)
print(df.size)
print(df.values)
#x=df[[0,1,2]]
#x=df.colums
#print(x)
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
#x,y=df.data,df.target
#x,y=np.arrange(10).reshape((5,2)),range(5)
#print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[190]:


print(x_train)
print(x_test)


# In[191]:


print(y_train)
print(y_test)


# In[4]:


from sklearn.svm import SVR
model=SVR()
model.fit(x_train,y_train)
pred=model.predict(x_test)
acc=model.score(x_test,y_test)
print(acc)


# In[5]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
# Train classifier
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)
acc=gnb.score(x_test,y_test)
print(acc)


# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[7]:


df=pd.read_csv("/home/seethalprince/Documents/Data1.csv")

label_encoder=LabelEncoder()
df['Name']=label_encoder.fit_transform(df['Name'].astype(str))
df['Age']=label_encoder.fit_transform(df['Age'].astype(str))
df['Heartbeat']=label_encoder.fit_transform(df['Heartbeat'].astype(str))
df['blood level']=label_encoder.fit_transform(df['blood level'].astype(str))
df['pressure']=label_encoder.fit_transform(df['pressure'].astype(str))


# In[8]:


x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(x_test)


# In[9]:


x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#x_train.head()
#data.colums["obs","BodyTemp","Gender","HeartRate"]
print(x_train)
#from sklearn.svm import svr
from sklearn.svm import SVR
model=SVR()
model.fit(x_train,y_train)
pred=model.predict(x_test)
acc=model.score(x_test,y_test)
print(acc)


# In[10]:


#implementdecisssion tree#


# In[11]:


import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


# In[12]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[13]:


import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
#from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split


# In[14]:


data=pd.read_csv("/home/seethalprince/Documents/Data1.csv")
data.head()
df=pd.DataFrame(data)
print(df)
print (df.shape)
print(df.size)
print(df.values)
#x=df[[0,1,2]]
#x=df.colums
#print(x)
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
#x,y=df.data,df.target
#x,y=np.arrange(10).reshape((5,2)),range(5)
#print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
def importdata(): 
    balance_data = pd.read_csv("/home/seethalprince/Documents/Data1.csv")
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
    print ("Dataset: ",balance_data.head()) 
return balance_data 
def splitdataset(balance_data): 
  
    # Seperating the target variable 
    X = balance_data.values[:, 1:5] 
    Y = balance_data.values[:, 0] 
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 
      
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
  
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 
  
# Driver code 
def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = train_using_entropy(X_train, X_test, y_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
      
if __name__=="__main__": 
    main() 


# In[ ]:


#svm


# In[ ]:


data=pd.read_csv("/home/seethalprince/Documents/Data1.csv")
data.head()
df=pd.DataFrame(data)
print(df)
print (df.shape)
print(df.size)
print(df.values)
#x=df[[0,1,2]]
#x=df.colums
#print(x)
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
#x,y=df.data,df.target
#x,y=np.arrange(10).reshape((5,2)),range(5)
#print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.svm import SVR
model=SVR()
model.fit(x_train,y_train)
pred=model.predict(x_test)
acc=model.score(x_test,y_test)
print(acc)


# In[ ]:


#Random Forest#


# In[ ]:


import pandas as pd
import numpy as np
import random
import seaborn as sns
#from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn import preprocessing


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
clf.predict(x_test)
acc=clf.score(x_test,y_test)
print(acc)


# In[ ]:


#kmean#


# In[ ]:


from sklearn import preprocessing
#from sklearn.preprocessing import standardScaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(df)
X_std=sc.transform(df)


# In[ ]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(X_std)
kmeans.labels_


# In[ ]:





# In[ ]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(X_std)
kmeans.labels_
print("\ncluster centers:")
#print(kmeans.cluster_center_)
print(kmeans.cluster_centers_)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# define one new instance
Xnew = [[-0.79415228, 89.10495117]]
# make a prediction
ynew = model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
# make a prediction
ynew = model.predict_proba(Xnew)
# show the inputs and predicted probabilities
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))


# In[15]:


#creating model
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

dataframe = pandas.read_csv("/home/seethalprince/cdc/CDC_Intern/Dataset/Data1.csv")
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'model.sav'
joblib.dump(model, filename)
 
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)


# In[17]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




