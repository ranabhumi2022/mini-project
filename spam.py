import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  #count vectorizer uses to convert text to number
import pandas as pd

#sample dataset
data={'message':['win money now',
                 'congratulation 1 crore',
                 'claim your amount',
                 'free enter in contest',
                 'are you coming today'],
'label':[0,1,0,1,0]} #1=spam 0=not spam
df=pd.DataFrame(data)

#convert text to numerical
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['message'])
Y=df['label']

#split dataset
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#create model
model=LogisticRegression()
model.fit(X_train,Y_train)  

prediction=model.predict(X_test)
print(prediction)
print("accuracy",accuracy_score(Y_test,prediction))

# test for new message
new_message=["freee money offer "]
new_message_vec=vectorizer.transform(new_message)

prediction = model.predict(new_message_vec)
if prediction[0]==1:
    print("spam message")
else:
    print("not spam")