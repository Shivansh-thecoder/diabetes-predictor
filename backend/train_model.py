import pandas as pd
import pickle
import os

data_path=os.path.join(os.path.dirname(__file__),'..','data','diabetes.csv')
df=pd.read_csv(data_path)
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df[col]=df[col].replace(0,pd.NA)
    df[col]=df[col].fillna(df[col].median())

X=df.drop('Outcome',axis=True)
y=df['Outcome']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
treeclassifier=DecisionTreeClassifier(max_depth=4,random_state=42)
treeclassifier.fit(X_train,y_train)

y_pred=treeclassifier.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("Accuray score:",accuracy_score(y_test,y_pred))
print("Confusion matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))

model_path = os.path.join(os.path.dirname(__file__), 'models', 'dt_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path,'wb') as f:
    pickle.dump(treeclassifier,f)

print(f" Model saved to {model_path}")

