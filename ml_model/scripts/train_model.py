import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('../data/ticket_data_vectorized.csv')
print(f'dataset for model training: {df.head()}')

#######################################################################

## Encoding the categorical columns ##
# ML models work with numerical data, but columns like Ticket type are categorical. So we need to convert them into numbers. 
#the LabelEncoder takes the distinct values in Ticket Type and assigns each one a numerical value. for example, network=0.
le = LabelEncoder()
df['Ticket Type'] = le.fit_transform(df['Ticket Type'])

#######################################################################

## Defining the features and the target + spliting to training and test sets ##
X = df.drop(columns=['Ticket ID', 'Approved']) #features
y = df['Approved'] #target 
y = y.map({'yes':1, 'no':0})

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=5)

#######################################################################
#Lets trainnn#
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

#######################################################################
#making predictions and evaluating the model

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy is: {accuracy*100}%')

#######################################################################
#Saving the trained model

with open('../model/ticket_approval_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)