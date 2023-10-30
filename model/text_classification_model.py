import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle

# read in the csv containing the sample data used to train the classification model
df = pd.read_csv("data/sample_questions_training_data.csv")
# splitting the data into a training set and a test set of data
train, test = train_test_split(df, test_size=0.33, random_state=42)
# identifying the training data and the labels associated with that data for both the training and testing set
train_x, train_y = train['Question'], train['Label']
test_x, test_y = test['Question'], test['Label']
# Instantiating the Term Frequency Inverse Document Frequency algorithm to transform text into a meaningful
# representation of numbers
tfidf = TfidfVectorizer(stop_words='english')
# Learning the vocabulary and inverse document frequency and returning a document-term matrix
train_x_vector = tfidf.fit_transform(train_x)
# transforming documents to document-term matrix
test_x_vector = tfidf.transform(test_x)
# instantiating the model - specifically a support vector machine
model = SVC(kernel='linear')
# fitting the model based on the training data we have vectorized and the training labels associated
model.fit(train_x_vector, train_y)

# TODO: Input a sample question to see what the classification model will classify it as
# answer = model.predict(tfidf.transform(["How many products did we sell over the last quarter?"]))
# print(answer[0])

# exporting the model, so we can use it in other files
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
# exporting the vectorizer, so we can use it in other files
with open("vectorizer.pkl", "wb") as fa:
    pickle.dump(tfidf, fa)
