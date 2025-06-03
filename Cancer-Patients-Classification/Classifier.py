#%%
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data/train_data.csv")
df_test = pd.read_csv("data/test_data.csv")

features = ['Age','T Stage','N Stage','Grade',
            'Tumor Size','Estrogen Status','Progesterone Status','Regional Node Examined','Reginol Node Positive',
            'Reginol Node Negative','Blood Pressure','Diastolic Pressure','Cholesterol',
            'Body Temperature','Oxygen Saturation','Respiratory Rate','Blood Glucose','BMI','Heart Rate',
            'Serum Creatinine','Uric Acid','Hemoglobin','GFR','Serum Sodium','Serum Potassium',
            'Serum Albumin','Lactate']

X = df[features]
y = df["Status"]
#%%
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
categorical_cols = X.select_dtypes(exclude=['float', 'int']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float', 'int']).columns.tolist()

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])
#%%
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
#%%
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (Dead):", precision_score(y_test, y_pred, pos_label=le.transform(['Dead'])[0]))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
#%%
test_data = df_test[features]
output = pipeline.predict(test_data)

result = []
for (_, row), answer in zip(df_test.iterrows(), output):
    result.append({
        "subtaskID": 5,
        "datapointID": row["ID"],
        "answer": le.inverse_transform([answer])[0]
    })

result = pd.DataFrame(result)
result.to_csv("result.csv", index=False)
#%%

#%%
