import pandas as pd

# Load dataset
df = pd.read_csv('project_folder/StudentsPerformance.csv')

# Check for null values
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
df['parental level of education'] = le.fit_transform(df['parental level of education'])
df['lunch'] = le.fit_transform(df['lunch'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


import matplotlib.pyplot as plt
import seaborn as sns

# Plot histograms of scores
df[['math score', 'reading score', 'writing score']].hist(figsize=(10, 5))
plt.show()

# Boxplot: Gender vs Math Score
sns.boxplot(x='gender', y='math score', data=df)
plt.title("Math Scores by Gender")
plt.show()

# Create average score
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Barplot: Test Preparation vs Average Score
sns.barplot(x='test preparation course', y='average_score', data=df)
plt.title("Average Score vs Test Preparation")
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Calculate average score (already done in EDA)
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Pass/Fail result
df['result'] = df['average_score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')

# Binary label encoding
df['result_binary'] = df['result'].map({'Fail': 0, 'Pass': 1})



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Prepare features and target
X = df.drop(columns=['average_score', 'result', 'result_binary'])
y = df['result_binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Evaluation
print("Logistic Regression:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

print("\nK-Nearest Neighbors:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from sklearn.cluster import KMeans

# Select features for clustering
X_cluster = df[['math score', 'reading score', 'writing score']]

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_cluster)

# Visualization
sns.scatterplot(x='math score', y='reading score', hue='cluster', data=df, palette='Set1')
plt.title("Cluster of Students based on Math and Reading Scores")
plt.show()



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Test preparation vs average score
sns.boxplot(x='test preparation course', y='average_score', data=df)
plt.title("Test Preparation vs Average Score")
plt.show()

# Lunch type vs average score
sns.boxplot(x='lunch', y='average_score', data=df)
plt.title("Lunch Type vs Average Score")
plt.show()

# Parental education vs average score
sns.barplot(x='parental level of education', y='average_score', data=df)
plt.title("Parental Education vs Average Score")
plt.xticks(rotation=45)
plt.show()



