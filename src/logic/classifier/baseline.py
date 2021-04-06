from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import datetime
from sklearn.svm import SVC
import nltk
from sklearn.preprocessing import LabelEncoder
from src.logic.classifier.load_data import LoadDataClassifier
from root import DIR_MODELS, DIR_OUTPUT
import csv
import os
import joblib
from nltk.stem.snowball import SpanishStemmer
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import logging.config
from root import DIR_CONF
logging.config.fileConfig(DIR_CONF + 'logging.conf')


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        stemmer = SpanishStemmer()
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        stemmer = SpanishStemmer()
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


class ErrorBaselineClassifier(Exception):
    """Base class for LoadDataClassifier exceptions"""
    pass


class BaselineClassifier:

    def __init__(self):
        self.encoder = LabelEncoder()
        self.data_classifier = LoadDataClassifier()
        self.stemmer = SpanishStemmer()
        self.vectorizer_data = None
        self.y_train = None
        self.y_test = None
        self.x_test = None
        self.x_train = None
        self.y_train_label = None
        self.y_test_label = None
        self.docs_train = None
        self.docs_test = None
        self.stop_words = None
        self.data_name = None
        self.current_collection_name = None
        self.current_file_all = None
        self.current_file_train = None
        self.current_file_test = None
        self.lang_support = {'es': 'spanish', 'en': 'english'}
        self.lang_current = ""
        self.user_features = False
        self.collection_type = ["folder", "excel", "csv", "json"]
        self.log_file = logging.getLogger('datasetBuild')
        self.log_file.info("Start Baseline Classifier")

    def stemmed_words(self, doc):
        return (self.stemmer.stem(w) for w in doc)

    def process_collection(self, configuration_folder=None, configuration_process=None, ngram=None, collection_name=None,
                           tfidf=None, lang=None, repeat_word=3, steam_spanish=False, oversampling=True):
        if configuration_folder is not None:
            self.data_classifier.configuration = configuration_folder
            self.data_classifier.processor_text.configuration = configuration_process
            self.data_classifier.read_source()
            self.lang_current = lang
            self.stop_words = nltk.corpus.stopwords.words(self.lang_support[lang])  # 'spanish', 'english'
            self.data_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

            self.docs_train, self.docs_test, self.y_train, self.y_test = \
                self.data_classifier.get_data_collection()

            self.current_collection_name = collection_name

            print('\t** DATA process_collection')
            print('\t** x_train: ', len(self.docs_train), 'x_test: ', len(self.docs_test),
                  'y_train', len(self.y_train), 'y_test', len(self.y_test))
            print('\t** class: ', self.y_train[0], self.y_train[0], self.docs_train[0][:50])
            print('\t** class: ', self.y_train[len(self.docs_train) - 1],
                  self.y_train[len(self.docs_train) - 1], self.docs_train[len(self.docs_train) - 1][:50])

            ngram = (1, 2) if ngram is None else ngram
            vectorizer = None
            if tfidf is None and steam_spanish is False:
                vectorizer = CountVectorizer(token_pattern='[a-zA-Z][a-zA-Z]{2,}', stop_words=self.stop_words,
                                             lowercase=True,  ngram_range=ngram, min_df=repeat_word)
            elif tfidf is not None and steam_spanish is False:
                vectorizer = TfidfVectorizer(token_pattern='[a-zA-Z][a-zA-Z]{2,}', stop_words=self.stop_words,
                                             lowercase=True, ngram_range=ngram, min_df=repeat_word)
            elif tfidf is None and steam_spanish is True:
                vectorizer = StemmedCountVectorizer(
                    token_pattern='[a-zA-Z][a-zA-Z]{2,}', stop_words=self.stop_words,
                    lowercase=True,  ngram_range=ngram, min_df=repeat_word, analyzer="word")
            elif tfidf is not None and steam_spanish is True:
                vectorizer = StemmedTfidfVectorizer(
                    token_pattern='[a-zA-Z][a-zA-Z]{2,}', stop_words=self.stop_words,
                    lowercase=True, ngram_range=ngram, min_df=repeat_word, analyzer="word")

            self.x_train = vectorizer.fit_transform(self.docs_train)
            self.x_test = vectorizer.transform(self.docs_test)
            print('\t ', 'Old: ', 'x_train.shape', self.x_train.shape, 'self.y_train.shape', self.y_train.shape)
            print('\t ', 'Old: ', sorted(Counter(self.y_train).items()))
            if oversampling:
                print('Over Sampling : ')
                ros = RandomOverSampler(random_state=1000)
                self.x_train, self.y_train = ros.fit_resample(self.x_train, self.y_train)
                print('\t ', 'New: ', sorted(Counter(self.y_train).items()))
                print('\t ', 'New: ', 'x_train.shape', self.x_train.shape, 'self.y_train.shape', self.y_train.shape)
            print('\t ', 'x_test.shape', self.x_test.shape, 'self.x_test.shape', self.y_test.shape)

            self.vectorizer_data = vectorizer
        else:
            raise ErrorBaselineClassifier("File and source settings are required. {}".format(configuration_folder))

    def classifier(self, type_classifier=None):
        if type_classifier is None or "LogisticRegression" in type_classifier:
            try:
                classifier_lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
                classifier_lr.fit(self.x_train, self.y_train)
                # score = classifier_lr.score(self.x_test, self.y_test)
                predicted_classes_lr = classifier_lr.predict(self.x_test)
                self.print_result_classifier_multilabel(
                    "LogisticRegression", self.data_name, predicted_classes_lr, classifier_lr)
            except Exception as e:
                self.log_file.error('\t ERROR Classifier LogisticRegression : ', str(e))

        if type_classifier is None or "MultinomialNB" in type_classifier:
            try:
                classifier_nb = MultinomialNB()
                classifier_nb.fit(self.x_train, self.y_train)
                predicted_classes_train_nb = classifier_nb.predict(self.x_test)
                self.print_result_classifier_multilabel(
                    "MultinomialNB", self.data_name, predicted_classes_train_nb, classifier_nb)
            except Exception as e:
                self.log_file.error('\t ERROR Classifier MultinomialNB : ', str(e))

        if type_classifier is None or "ComplementNB" in type_classifier:
            try:
                classifier_cnb = ComplementNB()
                classifier_cnb.fit(self.x_train, self.y_train)
                predicted_classes_train_cnb = classifier_cnb.predict(self.x_test)
                self.print_result_classifier_multilabel(
                    "ComplementNB", self.data_name, predicted_classes_train_cnb, classifier_cnb)
            except Exception as e:
                self.log_file.error('\t ERROR Classifier ComplementNB : ', str(e))

        if type_classifier is None or "GaussianNB" in type_classifier:
            try:
                classifier_gnb = GaussianNB()
                classifier_gnb.fit(self.x_train, self.y_train)
                predicted_classes_train_gnb = classifier_gnb.predict(self.x_test)
                self.print_result_classifier_multilabel(
                    "GaussianNB", self.data_name, predicted_classes_train_gnb, classifier_gnb)
            except Exception as e:
                self.log_file.error('\t ERROR Classifier GaussianNB : ', str(e))

        if type_classifier is None or "RandomForestClassifier" in type_classifier:
            try:
                classifier_rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
                classifier_rf.fit(self.x_train, self.y_train)
                predicted_classes_train_rf = classifier_rf.predict(self.x_test)
                self.print_result_classifier_multilabel(
                    "RandomForestClassifier", self.data_name, predicted_classes_train_rf, classifier_rf)
            except Exception as e:
                self.log_file.error('\t ERROR Classifier RandomForestClassifier : ', str(e))

        if type_classifier is None or "SVM" in type_classifier:
            try:
                classifier_svm = SVC(kernel='linear', C=0.5)
                classifier_svm.fit(self.x_train, self.y_train)
                predicted_classes_train_svm = classifier_svm.predict(self.x_test)
                self.print_result_classifier_multilabel(
                    "SVM", self.data_name, predicted_classes_train_svm, classifier_svm)
            except Exception as e:
                self.log_file.error('\t ERROR Classifier SVM : ', str(e))

    def print_result_classifier_multilabel(self, model_name, data_name, predicted_classes, classifier_sav):
        print("-" * 20, 'Start', model_name, "-" * 20)
        print("Test Accuracy on train {0}: {1:.2f}%".format(model_name, np.mean(predicted_classes == self.y_test) * 100))
        accuracy_score_model = accuracy_score(self.y_test, predicted_classes)
        print('Accuracy:', accuracy_score_model)
        f1_score_model = f1_score(self.y_test, predicted_classes, average='micro')
        print('F1 score:', f1_score_model)
        recall_score_model = recall_score(self.y_test, predicted_classes, average='micro')
        print('Recall:', recall_score_model)
        precision_score_model = precision_score(self.y_test, predicted_classes, average='micro')
        print('Precision:', precision_score_model)
        classification_report_model = classification_report(self.y_test, predicted_classes)
        print('\n Classification report:\n', classification_report_model)
        confusion_matrix_model = confusion_matrix(self.y_test, predicted_classes)
        print('\n Confusion matrix:\n', confusion_matrix_model)
        print("-" * 20, 'End', model_name, "-" * 20)
        self.save_to_csv_classifier(self.current_collection_name, self.lang_current, model_name, data_name,
                                    accuracy_score_model, recall_score_model, precision_score_model,
                                    classification_report_model, confusion_matrix_model, self.current_file_all,
                                    self.current_file_train, self.current_file_test, classifier_sav,
                                    self.vectorizer_data)

    @staticmethod
    def save_to_csv_dataset(path_file, values, keys, keys_name):
        with open(path_file, 'w') as out_csv:
            writer = csv.writer(out_csv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(["value", "key", "key_name"])
            for i in range(len(values)):
                if keys_name is None:
                    writer.writerow([LoadDataClassifier.delete_special_characters(LoadDataClassifier.proper_encoding(
                        values[i])), keys[i]])
                else:
                    writer.writerow([LoadDataClassifier.delete_special_characters(LoadDataClassifier.proper_encoding(
                        values[i])), keys[i], keys_name[i]])

    @staticmethod
    def save_to_csv_classifier(collection_name, lang, model_name, data_name, accuracy_score_model, recall_score_model,
                               precision_score_model, classification_report_model, confusion_matrix_model,
                               current_file_all, current_file_train, current_file_test, classifier_sav, vectorizer):
        # print(collection_name, lang, model_name, data_name, accuracy_score_model, recall_score_model,
        #                        precision_score_model, classification_report_model, confusion_matrix_model,
        #                        current_file_all, current_file_train, current_file_test, classifier_sav, vectorizer)
        date_file = datetime.datetime.now().strftime("%Y-%m-%d")
        date_model_file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        date_row = str(datetime.datetime.now())  # .strftime("%Y-%m-%d-%H-%M")
        file_path_sav = "{}classification/model_{}_{}_{}_{}_{}AC_{}.sav.z".format(
            DIR_OUTPUT, collection_name, lang, model_name, data_name, int(accuracy_score_model * 100), date_model_file)
        file_path_vec = "{}classification/model_{}_{}_{}_{}_{}.vec.z".format(
            DIR_OUTPUT, collection_name, lang, model_name, data_name, date_model_file)
        file_path_csv = "{}classification/report/result_{}.csv".format(DIR_OUTPUT, date_file)
        file_path_txt = "{}classification/report/summary_{}.txt".format(DIR_OUTPUT, date_file)
        type_file = 'a' if os.path.isfile(file_path_csv) else 'w'
        with open(file_path_csv, type_file) as out_csv:
            writer = csv.writer(out_csv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            if type_file == 'w':
                writer.writerow(['date', 'collection_name', 'lang', 'data_name', 'model_name',
                                 'accuracy_score', 'recall_score', 'precision_score',
                                 'current_file_all', 'current_file_train', 'current_file_test'])
            writer.writerow([date_row, collection_name, lang, data_name, model_name,
                             accuracy_score_model, recall_score_model, precision_score_model,
                             current_file_all, current_file_train, current_file_test])
        type_file = 'a' if os.path.isfile(file_path_txt) else 'w'
        with open(file_path_txt, type_file) as out_txt:
            out_txt.write("#" * 20 + ' START : ' + collection_name + " - " + model_name +
                          " - " + data_name + " - " + date_row + "#" * 20)
            out_txt.write("\n* classification_report_model \n")
            out_txt.write(classification_report_model)
            out_txt.write("\n* confusion_matrix_model \n")
            out_txt.write(str(confusion_matrix_model)+"\n")
            out_txt.write(
                "#" * 20 + ' END : ' + collection_name + " - " + model_name +
                " - " + data_name + " - " + date_row + "#" * 20)
            out_txt.write('\n')
        # save the model to disk
        joblib.dump(classifier_sav, file_path_sav, compress=True)
        joblib.dump(vectorizer, file_path_vec, compress=True)


if __name__ == '__main__':
    pass
    # app = Application()
    # app.run()
