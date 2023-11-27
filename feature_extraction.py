from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

"""1- Feature extraction ( apply all 3 algorithms with the classifier and choose the best according to the model's 
accuracy) 
2- ML classifier ( apply any ML classifier SVM, NB, DT, RF, etc.) and evaluation metrics ( including 
model's accuracy, confusion matrix )"""


def extract_features(data):
    vectorized = TfidfVectorizer()
    X = vectorized.fit_transform(data['text'])
    y = data['text']
    return X, y


def train_and_evaluate_model(X_train, X_test, y_train, y_test, classifier):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report
