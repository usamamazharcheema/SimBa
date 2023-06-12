import pathlib

import pandas as pd
import _pickle as cPickle

from sklearn import naive_bayes, preprocessing, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.learning import DATA_PATH

#classifier = LogisticRegression(max_iter=1200000)
#classifier = DecisionTreeClassifier(random_state=2)
#classifier = svm.SVC(probability=True)
#classifier = svm.NuSVC()
#classifier = svm.SVC()
#classifier = svm.LinearSVC(max_iter=1200000)
#classifier = naive_bayes.MultinomialNB()
#scaler = preprocessing.MinMaxScaler()
scaler = StandardScaler(with_mean=False)

STORAGE_PATH_CLASSIFIER = str(pathlib.Path(__file__).parent.resolve()) + "/models/var_detection_classifier.pkl"
STORAGE_PATH_SCALER = str(pathlib.Path(__file__).parent.resolve()) + "/models/var_detection_scaler.pkl"
COUNT_VECTORIZER_PATH = str(pathlib.Path(__file__).parent.resolve()) + "/models/count_vectorizer.pkl"
DF_PATH = DATA_PATH + "variable_detection/sv_ident_trial_train_and_val_variable_detection/variable_detection_df.tsv"


def text_column_to_bow_column(text_data, n=2000):
    count_vectorizer = CountVectorizer(max_features=n, lowercase=False)#, stop_words=['english', 'german'])
    bow_features = count_vectorizer.fit_transform(text_data).toarray()
    with open(COUNT_VECTORIZER_PATH , 'wb') as fid:
        cPickle.dump(count_vectorizer, fid)
    return bow_features


def test_classifier(df_path):
    classifier = svm.SVC()
    training_df = pd.read_csv(df_path, sep='\t', header=None)
    text_data = training_df.iloc[:, 1:2].values.ravel()
    X = text_column_to_bow_column(text_data)
    y = training_df.iloc[:, 2:]
    y = y.values.ravel()
    X = scaler.fit_transform(X, y)

    scores = cross_val_score(classifier, X, y, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))



def train_and_store_detection_classifier(n_features=2000):

    classifier = svm.SVC()
    training_df = pd.read_csv(DF_PATH, sep='\t')
    text_data = training_df.iloc[:, 1:2].values.ravel()
    X = text_column_to_bow_column(text_data, n=n_features)
    y = training_df.iloc[:, 2:]
    y = y.values.ravel()
    X = scaler.fit_transform(X, y)
    classifier.fit(X, y)

    with open(STORAGE_PATH_CLASSIFIER , 'wb') as fid:
        cPickle.dump(classifier, fid)

    with open(STORAGE_PATH_SCALER , 'wb') as fid:
        cPickle.dump(scaler, fid)


def load_classifier_and_predict_variables(queries_file_path):

    with open(STORAGE_PATH_CLASSIFIER, 'rb') as fid:
        this_classifier = cPickle.load(fid)

    with open(STORAGE_PATH_SCALER, 'rb') as fid:
        this_scaler = cPickle.load(fid)

    with open(COUNT_VECTORIZER_PATH , 'rb') as fid:
        this_vectorizer = cPickle.load(fid)

    test_df = pd.read_csv(queries_file_path, sep='\t')
    text_data = test_df.iloc[:, 1:2].values.ravel()
    X = this_vectorizer.transform(text_data).toarray()
    X = this_scaler.transform(X)
    y_pred = this_classifier.predict(X)
    test_df['is_variable'] = y_pred
    output_df = test_df.loc[test_df['is_variable'] != 0]
    output_df = output_df.iloc[:, :2]
    output_df.to_csv(queries_file_path, sep='\t', header=False, index=False)



# df_path = DATA_PATH + "variable_detection/sv_ident_trial_train_and_val_variable_detection/variable_detection_df.tsv"
# test_classifier(df_path)
# train_and_store_detection_classifier()

