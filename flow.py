from joblib import Memory
import pandas as pd
import re
import mlflow
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# import feature extraction methods from sklearn
from sklearn.feature_extraction.text import CountVectorizer

from prefect import task, flow

@task
def load_data(path):
    df = pd.read_csv(path)

    df = df.dropna(subset=['Review text'])

    df['sentiment'] = df['Ratings'].apply(lambda x: 1 if x >= 3 else 0) # 1 is positive and 0 is negative

    return df

@task
def split_inputs_output(data, inputs, output):
    X = data[inputs]
    y = data[output]
    return X, y

@task
def split_train_test(X, y, test_size=0.25, random_state=0):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)




@task
def preprocess_data(X_train, X_test):

    def clean(doc): # doc is a string of text
        lemmatizer = WordNetLemmatizer()

        # This text contains a lot of READ MORE tags.
        doc = doc.replace("READ MORE", " ")
        
        # Remove punctuation and numbers.
        doc = re.sub(r'[^a-zA-Z\s]', '', doc)
    #     doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])

        # Converting to lower case
        doc = doc.lower()
        
        # Tokenization
        tokens = nltk.word_tokenize(doc)

        # Lemmatize
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Stop word removal
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
        
        # Join and return
        return " ".join(filtered_tokens)

    X_train_clean = X_train.apply(lambda doc: clean(doc))
    X_test_clean = X_test.apply(lambda doc: clean(doc))

    # scaler = MinMaxScaler()
    vectorizer = TfidfVectorizer(max_features=1000)

    X_train_scaled = vectorizer.fit_transform(X_train_clean)
    X_test_scaled = vectorizer.transform(X_test_clean)

    return X_train_scaled, X_test_scaled


@task
def train_model(X_train_scaled, y_train, hyperparameters):

    svc = SVC(**hyperparameters)
    svc.fit(X_train_scaled, y_train)

    return svc

@task
def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_score = metrics.f1_score(y_train, y_train_pred)
    test_score = metrics.f1_score(y_test, y_test_pred)
    
    return train_score, test_score


@flow(name='SVC-Flow')
def workflow():
    path = 'reviews_badminton/data.csv'
    inputs = 'Review text'
    output = 'sentiment'
    param_grid = {'C':1,'kernel':'rbf'}
    df = load_data(path)

    X, y = split_inputs_output(df,inputs,output)

    X_train, X_test, y_train, y_test = split_train_test(X,y)

    X_train_scaled, X_test_scaled = preprocess_data(X_train,X_test)

    model = train_model(X_train_scaled,y_train,param_grid)

    train_score, test_score = evaluate_model(model,X_train_scaled,y_train, X_test_scaled, y_test)

    print("Train Score is ",train_score ,"\n Test Score is ",test_score)

if __name__ == "__main__":
    workflow.serve(
        name="my-first-deployment",
        cron="* * * * *"
    )