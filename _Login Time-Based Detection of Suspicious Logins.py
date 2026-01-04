#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_curve
)

import joblib

np.random.seed(42)


# In[4]:


users = [f"user_{i}" for i in range(1, 26)]  # 25 users
records = []

for user in users:
    base_hour = np.random.randint(8, 20)  # normal login time per user
    
    for _ in range(300):
        day = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 180))
        hour = int(np.clip(np.random.normal(base_hour, 1.8), 0, 23))
        minute = np.random.randint(0, 60)
        timestamp = day.replace(hour=hour, minute=minute)
        
        label = 0
        
        # Inject suspicious logins (~10%)
        if np.random.rand() < 0.10:
            timestamp = timestamp.replace(hour=np.random.randint(0, 24))
            label = 1
        
        records.append([
            user,
            timestamp,
            timestamp.hour,
            timestamp.weekday(),
            label
        ])

df = pd.DataFrame(
    records,
    columns=["user_id", "timestamp", "hour", "weekday", "label"]
)

df.head()


# In[5]:


print("Dataset Shape:", df.shape)
print("\nLabel Distribution:")
print(df['label'].value_counts())


# In[6]:


df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

user_avg = df.groupby('user_id')['hour'].mean().reset_index()
user_avg.columns = ['user_id', 'avg_hour']

df = df.merge(user_avg, on='user_id', how='left')

df['hour_deviation'] = abs(df['hour'] - df['avg_hour'])

df.head()


# In[7]:


X = df[['hour', 'weekday', 'is_weekend', 'hour_deviation']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# In[8]:


dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Decision Tree Performance")
print(classification_report(y_test, dt_pred))


# In[9]:


rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=7,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest Performance")
print(classification_report(y_test, rf_pred))


# In[10]:


iso = IsolationForest(contamination=0.10, random_state=42)
iso.fit(X_train)

iso_pred = iso.predict(X_test)
iso_pred = np.where(iso_pred == -1, 1, 0)

print("Isolation Forest Performance")
print(classification_report(y_test, iso_pred))


# In[11]:


def metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    ]

results = pd.DataFrame(
    [
        metrics(y_test, dt_pred),
        metrics(y_test, rf_pred),
        metrics(y_test, iso_pred)
    ],
    columns=["Accuracy", "Precision", "Recall", "F1-Score"],
    index=["Decision Tree", "Random Forest", "Isolation Forest"]
)

results


# In[12]:


def metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    ]

results = pd.DataFrame(
    [
        metrics(y_test, dt_pred),
        metrics(y_test, rf_pred),
        metrics(y_test, iso_pred)
    ],
    columns=["Accuracy", "Precision", "Recall", "F1-Score"],
    index=["Decision Tree", "Random Forest", "Isolation Forest"]
)

results


# In[13]:


dt_prob = dt.predict_proba(X_test)[:, 1]
rf_prob = rf.predict_proba(X_test)[:, 1]

fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

plt.plot(fpr_dt, tpr_dt, label="Decision Tree")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[14]:


feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

feat_imp


feat_imp.plot(kind='bar', x='Feature', y='Importance', title='Feature Importance')
plt.show()


# In[15]:


normal = df[df['label'] == 0]
suspicious = df[df['label'] == 1]

plt.hist(normal['hour'], bins=24, alpha=0.6, label='Normal')
plt.hist(suspicious['hour'], bins=24, alpha=0.6, label='Suspicious')
plt.xlabel("Hour of Day")
plt.ylabel("Count")
plt.title("Login Time Distribution")
plt.legend()
plt.show()


# In[16]:


def detect_login(hour, weekday):
    avg_hour = df['hour'].mean()
    deviation = abs(hour - avg_hour)
    is_weekend = 1 if weekday >= 5 else 0

    sample = pd.DataFrame(
        [[hour, weekday, is_weekend, deviation]],
        columns=X.columns
    )
    
    return "🚨 Suspicious Login" if rf.predict(sample)[0] == 1 else "✅ Normal Login"

detect_login(1, 2)


# In[17]:


df.to_csv("login_dataset_25_users.csv", index=False)
results.to_csv("model_results.csv")
joblib.dump(rf, "random_forest_model.pkl")


# In[ ]:




