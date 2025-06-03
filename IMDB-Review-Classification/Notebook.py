#%% md
# ### In this task we will use the IMDB movie review dataset to perform a binary classification task. Our goal is to classify the reviews as positive or negative
# ---
# [Dataset Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
#%%
# Load libraries and data

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

df = pd.read_csv('data/IMDB_Dataset.csv')

df.head()
#%%
# Preprocess Data

train_data = df["review"]
test_data = df["sentiment"]

encoder = LabelEncoder()

test_data_encoded = encoder.fit_transform(test_data)

def clean_html(text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', text)

train_data = train_data.apply(clean_html)

X_train, X_test, y_train, y_test = train_test_split(train_data, test_data_encoded, test_size=0.1, random_state=42)

train_data
#%%
# Build and fit pipeline

pipeline = Pipeline([
    ("preprocessing", TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9, sublinear_tf=True)),
    ("model", LinearSVC(random_state=42))
])

pipeline.fit(X_train, y_train)
#%%
# Evaluate Model Performance

y_pred = pipeline.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Report:")
print(classification_report(y_test, y_pred))
#%%
score = abs(cross_val_score(X=train_data, y=test_data_encoded, estimator=pipeline, scoring="accuracy", cv=6)).mean()

print("Score: ", score)
#%% md
# ### Note: Ignore this cell this is to organize projects in the README.MD
# ```json
# {
# "Tags": ["Binary Classification", "SVC", "NLP"]
# }
# ```