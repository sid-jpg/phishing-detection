import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import pickle  # Import pickle for saving/loading the model

# Load the dataset with low_memory=False to avoid mixed type warnings
data = pd.read_csv(r'\Phishing\urlset\urlset.csv', encoding='ISO-8859-1', on_bad_lines='skip', low_memory=False)

# Check the first few rows of the dataset
print(data.head())

# Check the data types of the columns
print(data.dtypes)

# Clean and convert data types
# Convert relevant columns to numeric types where applicable
numeric_columns = ['ranking', 'mld_res', 'mld.ps_res', 'card_rem',
                   'ratio_Rrem', 'ratio_Arem', 'jaccard_RR',
                   'jaccard_RA', 'jaccard_AR', 'jaccard_AA',
                   'jaccard_ARrd', 'jaccard_ARrem', 'label']

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Drop rows with NaN values that resulted from conversion errors
data = data.dropna()

# Use the correct column names based on the dataset description
# 'domain' for the URL and 'label' for the phishing status
df = data[['domain', 'label']]

# 3. Converting Text into Vectors using TfidfVectorizer
tf = TfidfVectorizer(stop_words="english", max_features=5000)  # Reduce the number of features
feature_x = tf.fit_transform(df["domain"])  # Keep it as a sparse matrix
y_tf = np.array(df['label'])  # Convert labels into a numpy array

# 4. Splitting into Train and Test Sets
x_train, x_test, y_train, y_test = train_test_split(feature_x, y_tf, train_size=0.8, random_state=0)

# 5. Instantiate and Train the Gradient Boosting Classifier
gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)
gbc.fit(x_train, y_train)

# Predict the target value from the model for the samples
y_train_gbc = gbc.predict(x_train)
y_test_gbc = gbc.predict(x_test)

# 6. Compute Accuracy, F1 Score, Recall, and Precision
acc_train_gbc = metrics.accuracy_score(y_train, y_train_gbc)
acc_test_gbc = metrics.accuracy_score(y_test, y_test_gbc)

print("Gradient Boosting Classifier : Accuracy on training Data: {:.3f}".format(acc_train_gbc))
print("Gradient Boosting Classifier : Accuracy on test Data: {:.3f}".format(acc_test_gbc))
print()

f1_score_train_gbc = metrics.f1_score(y_train, y_train_gbc, average='weighted')  # Use 'weighted' for multiclass
f1_score_test_gbc = metrics.f1_score(y_test, y_test_gbc, average='weighted')

print("Gradient Boosting Classifier : f1_score on training Data: {:.3f}".format(f1_score_train_gbc))
print("Gradient Boosting Classifier : f1_score on test Data: {:.3f}".format(f1_score_test_gbc))
print()

recall_score_train_gbc = metrics.recall_score(y_train, y_train_gbc, average='weighted')
recall_score_test_gbc = metrics.recall_score(y_test, y_test_gbc, average='weighted')
print("Gradient Boosting Classifier : Recall on training Data: {:.3f}".format(recall_score_train_gbc))
print("Gradient Boosting Classifier : Recall on test Data: {:.3f}".format(recall_score_test_gbc))
print()

precision_score_train_gbc = metrics.precision_score(y_train, y_train_gbc, average='weighted')
precision_score_test_gbc = metrics.precision_score(y_test, y_test_gbc, average='weighted')
print("Gradient Boosting Classifier : Precision on training Data: {:.3f}".format(precision_score_train_gbc))
print("Gradient Boosting Classifier : Precision on test Data: {:.3f}".format(precision_score_test_gbc))

# Compute the classification report of the model
print(metrics.classification_report(y_test, y_test_gbc))

# Optional: Hyperparameter Tuning for Learning Rate
training_accuracy = []
test_accuracy = []
depth = range(1, 10)

for n in depth:
    forest_test = GradientBoostingClassifier(learning_rate=n * 0.1)
    forest_test.fit(x_train, y_train)
    # Record training set accuracy
    training_accuracy.append(forest_test.score(x_train, y_train))
    # Record generalization accuracy
    test_accuracy.append(forest_test.score(x_test, y_test))

# Plotting the training & testing accuracy for learning rates
plt.figure(figsize=(10, 6))
plt.plot(depth, training_accuracy, label="Training Accuracy")
plt.plot(depth, test_accuracy, label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Learning Rate (x0.1)")
plt.title("Training and Testing Accuracy vs Learning Rate")
plt.legend()
plt.grid()
plt.show()

# Optional: Hyperparameter Tuning for Max Depth
training_accuracy = []
test_accuracy = []
depth = range(1, 10, 1)

for n in depth:
    forest_test = GradientBoostingClassifier(max_depth=n, learning_rate=0.7)
    forest_test.fit(x_train, y_train)
    # Record training set accuracy
    training_accuracy.append(forest_test.score(x_train, y_train))
    # Record generalization accuracy
    test_accuracy.append(forest_test.score(x_test, y_test))

# Plotting the training & testing accuracy for max_depth
plt.figure(figsize=(10, 6))
plt.plot(depth, training_accuracy, label="Training Accuracy")
plt.plot(depth, test_accuracy, label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Max Depth")
plt.title("Training and Testing Accuracy vs Max Depth")
plt.legend()
plt.grid()
plt.show()

# Save the model and vectorizer
model_path = r"Phishing\models\phishingGradient.pkl"
vectorizer_path = r"Phishing\models\phishingTfidfVectorizer.pkl"

with open(model_path, 'wb') as model_file:
    pickle.dump(gbc, model_file)

with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(tf, vectorizer_file)

print("Model and vectorizer saved successfully!")