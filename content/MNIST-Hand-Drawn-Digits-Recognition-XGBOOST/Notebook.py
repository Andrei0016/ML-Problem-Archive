#%% md
# ### In this task we will classify 28x28 images of handwritten digits using xgboost
# ---
#%%
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
#%%
# Function for loading dataset
def load_dataset(path):
    encoder = LabelEncoder()
    labels = []
    images = []

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            labels.append(folder)
            img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
            img = img.flatten() / 255.0
            images.append(img)

    labels = encoder.fit_transform(labels)
    return np.array(images), np.array(labels)
#%%
# Load data
X_train, y_train = load_dataset(path="data/train")
X_test, y_test = load_dataset(path="data/test")

# Shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)
#%%
# Create pipeline and train it
from xgboost import XGBClassifier

pipeline = Pipeline([
    ("dim-red", PCA(n_components=0.95, svd_solver="full")),
    ("model", XGBClassifier()),
])

pipeline.fit(X_train, y_train)
#%%
# Evaluate Model on unseen data
y_pred = pipeline.predict(X_test)
print("Accuracy:" , accuracy_score(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
#%% md
# ## Notes:
# ---
# During this project I made the following observations:
# 1. PCA even with a high `n_component` of `0.95` (keeps 95% relevance) the model performs well and the run time is drastically reduced
# 2. Xgboost is a good alternative to simple image classification tasks
# 3. Shuffling data is important and unshuffled data can compromise the model accuracy on some classes or make the model overfit
#%% md
# ### Note: Ignore this cell. This is to organize projects in the README.MD
# ```json
# {
# "Tags": ["Image Classification", "XGBoost", "CV", "Dimensionality Reduction", "PCA"]
# }
# ```