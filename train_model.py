import os, re, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
DATA_PATH='mental_health_data.csv'; MODEL_DIR='model'; MODEL_PATH=os.path.join(MODEL_DIR,'mental_health_model.pkl')
def clean_text(text):
    text=str(text).lower(); text=re.sub(r'http\S+','',text); text=re.sub(r'[^a-zA-Z\s]','',text); text=re.sub(r'\s+',' ',text).strip(); return text
def train_and_save_model():
    os.makedirs(MODEL_DIR, exist_ok=True); df=pd.read_csv(DATA_PATH); df['clean_text']=df['text'].apply(clean_text)
    X_train,X_test,y_train,y_test=train_test_split(df['clean_text'],df['label'],test_size=.25,random_state=42,stratify=df['label'])
    model=Pipeline([('tfidf',TfidfVectorizer(stop_words='english',max_features=5000)),('classifier',LogisticRegression(max_iter=1000))])
    model.fit(X_train,y_train); y_pred=model.predict(X_test)
    print('Model Training Completed'); print('Accuracy:', round(accuracy_score(y_test,y_pred)*100,2), '%'); print(confusion_matrix(y_test,y_pred)); print(classification_report(y_test,y_pred))
    joblib.dump(model, MODEL_PATH); print('Model saved at:', MODEL_PATH); return model
if __name__=='__main__': train_and_save_model()
