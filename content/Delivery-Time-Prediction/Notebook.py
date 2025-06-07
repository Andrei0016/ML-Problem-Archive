#%% md
# ### In this task we will predict the delivery time of packages
# ---
# [Task Link](https://judge.nitro-ai.org/roai-2025/ojia-9-10/problems/2/task)
#%%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
#%%
# Load data
df_train = pd.read_csv('data/train_data.csv')
df_test = pd.read_csv('data/test_data.csv')

X = df_train[["Distance", "Time of Day", "Weather", "Traffic", "Road Quality", "Driver Experience"]]
y = df_train["deliver_time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Create column transformer
cat_features = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', Pipeline([
            ('scaler', StandardScaler())
        ]), num_features)
    ],
    remainder='passthrough'
)
#%%
# Create and fit pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
#%%
# Evaluate model performance
y_pred = pipeline.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred)

cv_mae = abs(cross_val_score(pipeline, X, y, cv=6, scoring='neg_mean_absolute_error')).mean()

print("MAE (test set):", mae_test)
print("Cross-validated MAE:", cv_mae)
#%%
# Make final submissions
X_pred = df_test[["Distance", "Time of Day", "Weather", "Traffic", "Road Quality", "Driver Experience"]]

cityMask = df_test['City A'] == 'Barlad'
fogMask = df_test['Weather'] == 'Fog'

st1 = df_test[cityMask & fogMask].shape[0]
subtask1 = pd.DataFrame({'subtaskID':[1], 'datapointID':1, 'answer':st1})

predictions = pipeline.predict(X_pred)
subtask2 = pd.DataFrame({'subtaskID':2, 'datapointID':df_test['ID'], 'answer':predictions})

output = pd.concat((subtask1, subtask2))

output.to_csv('submission.csv')
#%% md
# ### Note: Ignore this cell. This is to organize projects in the README.MD
# ```json
# {
# "Tags": ["Regression", "Feature Engineering"]
# }
# ```