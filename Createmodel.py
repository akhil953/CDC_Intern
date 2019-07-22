import pandas as pd
import numpy as np
df=pd.read_csv("/home/seethalprince/Documents/Data1.csv")
df.head()
from sklearn import linear_modelx = data[['Heartbeat','pressure']]
y = data.Age.values
clf=LogisticRegression()
clf.fit(x,y)
clf.score(x,y)
joblib.dump(clf,'/home/seethalprince/Desktop/cdcproject/model.pk1')
clf_load=joblib.load('/home/seethalprince/Desktop/cdcproject/model.pk1')
assert clf.score(x,y)==clf_load.score(x,y)
print(clf_load)
clf_load.predict(x)
