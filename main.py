from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, drop_columns, apply_preprocessing, save_to_file
from feature_extraction import extract_features, train_and_evaluate_model
from ngrams_analysis import analyze_ngrams, save_ngram_probabilities
import warnings

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

fake_csv = 'data/Fake.csv'
true_csv = 'data/True.csv'
output_file_before = 'data/Before.txt'
output_file_after = 'data/output_file.txt'
ngram_probabilities_file = 'data/ngram_probabilities.csv'
sample_size = 1000


data = load_data(fake_csv, true_csv)
data = drop_columns(data)
data = apply_preprocessing(data)
# print(data.columns)

# """

save_to_file(data, output_file_before)


data = analyze_ngrams(data, n=3)
save_ngram_probabilities(data, sample_size, ngram_probabilities_file)


features, labels = extract_features(data)


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


nb_classifier = MultinomialNB()
svm_classifier = SVC()
rf_classifier = RandomForestClassifier()
dt_classifier = DecisionTreeClassifier()


nb_accuracy, nb_report = train_and_evaluate_model(X_train, X_test, y_train, y_test, nb_classifier)
svm_accuracy, svm_report = train_and_evaluate_model(X_train, X_test, y_train, y_test, svm_classifier)
rf_accuracy, rf_report = train_and_evaluate_model(X_train, X_test, y_train, y_test, rf_classifier)

'''

# Print results
print("Naive Bayes:")
print(f"Accuracy: {nb_accuracy}")
# print("Classification Report:")
# print(nb_report)

print("\nSVM:")
print(f"Accuracy: {svm_accuracy}")
# print("Classification Report:")
# print(svm_report)

print("\nRandom Forest:")
print(f"Accuracy: {rf_accuracy}")
# print("Classification Report:")
# print(rf_report)
'''
best_model = max([('Naive Bayes', nb_accuracy), ('SVM', svm_accuracy), ('Random Forest', rf_accuracy)],
                 key=lambda x: x[1])

classifiers = [('Naive Bayes', nb_classifier), ('SVM', svm_classifier), ('Random Forest', rf_classifier),
               ('Decision Tree', dt_classifier)]

for classifier_name, classifier in classifiers:
    accuracy, report = train_and_evaluate_model(X_train, X_test, y_train, y_test, classifier)
    print(f"\n{classifier_name}:")
    print(f"Accuracy: {accuracy}")
    # print("Classification Report:")
    # print(report)

print(f"\nBest Model: {best_model[0]}")
