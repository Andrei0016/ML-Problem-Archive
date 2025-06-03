#%% md
# ### In this task we have 2 subtasks:
# 1. Classify human or ai text
# 2. Group texts into categories (crime, science, religion business)
# ---
# [Task Link](https://judge.nitro-ai.org/roai-2025/onia/problems/1/task)
#%%
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import math

df_train = pd.read_csv("data/train_data.csv")
df_test = pd.read_csv("data/test_data.csv")
#%%
X = df_train["text"]
y = df_train["label"]
X_task1 = df_test[df_test["subtaskID"] == 1]
X_task2 = df_test[df_test["subtaskID"] == 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#%% md
# ## Task 1 (Binary Classification)
#%%
task1_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('model', LinearSVC())
])

task1_pipeline.fit(X_train, y_train)
#%%
y_pred = task1_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
#%%
task1_pred = task1_pipeline.predict(X_task1["text"])

task1 = pd.DataFrame({
    "subtaskID": [1] * len(task1_pred),
    "datapointID": X_task1["ID"].values,
    "answer": task1_pred
})
#%% md
# ## Task 2 (Clustering)
#%%
task2_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('model', KMeans(n_clusters=4, random_state=42, n_init=500))
])

task2_pipeline.fit(X_task2["text"])
#%%
clusters = task2_pipeline.predict(X_task2["text"])

label_map = {0: "SCIENCE", 1: "CRIME", 2: "BUSINESS", 3: "RELIGION"}
mapped_clusters = pd.Series(clusters).map(label_map)
task2 = pd.DataFrame({
    "subtaskID": [2] * len(mapped_clusters),
    "datapointID": X_task2["ID"].values,
    "answer": mapped_clusters
})
#%%
submission = pd.concat([task1, task2])
submission.to_csv("result.csv", index=False)
#%% md
# ## (Optional) Cluster visualization
#%%
texts = X_task2["text"].reset_index(drop=True)
df_clusters = pd.DataFrame({'cluster': clusters, 'text': texts})
unique_clusters = sorted(df_clusters['cluster'].unique())
cols = 2
rows = math.ceil(len(unique_clusters) / cols)
plt.figure(figsize=(cols * 6, rows * 4))

for idx, cluster in enumerate(unique_clusters):
    cluster_text = " ".join(df_clusters[df_clusters['cluster'] == cluster]['text'])
    wc = WordCloud(width=800, height=400, background_color="white").generate(cluster_text)
    ax = plt.subplot(rows, cols, idx + 1)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Cluster {cluster} – {label_map.get(cluster, '')}")

plt.tight_layout()
plt.show()
#%% md
# ### Note: Ignore this cell this is to organize projects in the README.MD
# ```json
# {
# "Tags": ["Clustering", "Binary Classification", "SVC", "NLP"]
# }
# ```