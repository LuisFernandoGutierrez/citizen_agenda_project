from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
import os
import re
import unicodedata
import time
import sys
import spacy
from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger
import nltk
from nltk.corpus import sentiwordnet as swn
# Do this first, that'll do something eval()
# to "materialize" the LazyCorpusLoader
import queue
import threading
# next(swn.all_senti_synsets())
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging.config
import docx2txt
from root import DIR_CONF, DIR_INPUT
logging.config.fileConfig(DIR_CONF + 'logging.conf')


class ErrorLoadDataClassifier(Exception):
    """Base class for LoadDataClassifier exceptions"""
    pass


class LoadDataClassifier:
    configuration = {'type': '', 'source': '', 'filed_text': '', 'field_class': ''}

    def __init__(self):
        self.docs_raw = {}
        self.docs_raw_array = []
        self.x_docs = []
        self.y_labels = []
        self.y_labels_encodes = []
        self.label_encoder = LabelEncoder()
        self.key_test = []
        self.key_train = []
        self.processor_text = GetTextPreProcess()
        self.log_file = logging.getLogger('datasetBuild')
        self.log_file.info("Start process LoadDataClassifier")

    def read_folder(self):
        count_file = 0
        path_folder = "{}classification/{}".format(DIR_INPUT, self.configuration['source'])
        if os.path.exists(path_folder):
            self.log_file.info("\t** #:READ in path {0}.".format(path_folder))
            for subdir, dirs, files in os.walk(path_folder):
                for file in files:
                    count_file += 1
                    file_path = subdir + os.path.sep + file
                    list_dir = subdir.split(os.sep)
                    name_class = list_dir[len(list_dir) - 1].replace(' ', '_')
                    type_class = list_dir[len(list_dir) - 2].replace(' ', '_')
                    if name_class not in self.docs_raw:
                        self.docs_raw[name_class] = []
                    try:
                        if count_file % 200 == 0:
                            self.log_file.info("\t** #: Files Loads : {0}".format(count_file))

                        if '.txt' in file_path:
                            txt_file = ''
                            with open(file_path, newline='') as file_text_buffer:  # , encoding='UTF-8'
                                txt_file = file_text_buffer.read()
                            self.docs_raw[name_class].append(
                                {'class': name_class, 'type': type_class, 'text': txt_file, 'file_name': file_path})
                            self.docs_raw_array.append(
                                {'class': name_class, 'type': type_class, 'text': txt_file, 'file_name': file_path})
                            self.x_docs.append(self.processor_text.process(txt_file))
                            self.y_labels.append(name_class)
                            self.log_file.info('\t** # Class: {0} - Type: {1} - File {2} - Count {3}'.format(
                                name_class, type_class, file_path, len(self.docs_raw[name_class])))
                        if '.docx' in file_path:
                            txt_file = docx2txt.process(file_path)
                            self.docs_raw[name_class].append(
                                {'class': name_class, 'type': type_class, 'text': txt_file, 'file_name': file_path})
                            self.docs_raw_array.append(
                                {'class': name_class, 'type': type_class, 'text': txt_file, 'file_name': file_path})
                            self.x_docs.append(self.processor_text.process(txt_file))
                            self.y_labels.append(name_class)
                            self.log_file.info('\t** # Class: {0} - Type: {1} - File {2} - Count {3}'.format(
                                name_class, type_class, file_path, len(self.docs_raw[name_class])))
                    except Exception as e:
                        self.log_file.error('\t ERROR : {}'.format(str(e)))
            self.log_file.info("\t** read_folder docs_raw : {} ".format(len(self.docs_raw_array)))
        else:
            self.log_file.info("\t** #:ERROR Source not found!. {0}".format(path_folder))

    def read_file(self):
        df_file = None
        file_path = "{}classification/{}".format(DIR_INPUT, self.configuration['source'])
        type_file = self.configuration['type']
        if type_file == 'excel':
            df_file = pd.read_excel(file_path)
        elif type_file == 'csv':
            df_file = pd.read_csv(file_path)
        elif type_file == 'json':
            df_file = pd.read_json(file_path)
        else:
            raise ErrorLoadDataClassifier("Type files does not support.")

        for index, row in df_file.iterrows():
            text = row[self.configuration['field_text']]
            new_text = self.processor_text.process(text)
            label = row[self.configuration['field_class']]
            if index % 100 == 0:
                self.log_file.info('\t Example #{} read file class: {} - value: {} new_value: {}'.format(
                    index, label, text[:50], new_text[:50]))
            self.docs_raw_array.append(
                {'class': label, 'type': type_file, 'text': text, 'file_name': file_path})
            self.x_docs.append(self.processor_text.process(text))
            self.y_labels.append(label)

    def artificial_rows(self, times):
        for i in range(times):
            self.x_docs += self.x_docs
            self.y_labels += self.y_labels
            self.log_file.info('\t** # artificial_rows: {} - X: {} - Y {} '.format(times, len(self.x_docs),
                                                                                   len(self.y_labels)))

    def read_source(self):
        type_file = self.configuration['type']
        if type_file == 'folder':
            self.read_folder()
        else:
            self.read_file()
        self.artificial_rows(self.configuration.get('artificial_rows', 0))
        self.y_labels_encodes = self.label_encoder.fit_transform(self.y_labels)

    def get_data_collection(self):
        # TODO: Verify https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html

        print('\t** DATA get_data_collection')
        print('\t** docs_train: {} y_labels_encodes: {}'.format(len(self.x_docs), len(self.y_labels_encodes)))
        print('\t** self.y_labels: ', sorted(Counter(self.y_labels).items()))
        print('\t** self.y_labels_encodes: ', sorted(Counter(self.y_labels_encodes).items()))
        docs_train, docs_test, y_train, y_test = \
            train_test_split(self.x_docs, self.y_labels_encodes,  test_size=0.4, random_state=1000)
        return docs_train, docs_test, y_train, y_test


class GetTextPreProcess:
    configuration = {}

    def __init__(self):
        self.text_users = []
        self.high = 0
        self.current = 0
        self.nlp = spacy.load('es_core_news_sm')
        self.log_file = logging.getLogger('datasetBuild')

    def proper_encoding(self, text):
        try:
            text = unicodedata.normalize('NFD', text)
            text = text.encode('ascii', 'ignore')
            text = text.decode("utf-8")
        except Exception as e:
            self.log_file.error("*** Error proper_encoding: ",  e)
        return text

    def delete_special_characters(self, text):
        try:
            text = re.sub(r'\/|\\|\\.|\,|\;|\:|\n|\?|\)|\(|\!|\¡|\¿|\'|\t', ' ', text)
            text = re.sub(r"\s+\w\s+", " ", text)
            text = re.sub("", "", text)
            text = re.sub("|", "", text)
            text = re.sub("@", "", text)
        except Exception as e:
            self.log_file.error("*** Error delete_special_characters: ",  e)
        return text.lower()

    def remove_punctuation(self, text):
        try:
            punctuation = {'/', '"', '(', ')', '.', ',', '%', ';', '?', '¿', '!', '¡',
                           ':', '#', '$', '&', '>', '<', '-', '_', '°', '|', '¬', '\\', '*', '+',
                           '[', ']', '{', '}', '=', "'", '@'}
            for sign in punctuation:
                text = text.replace(sign, '')
        except Exception as e:
            self.log_file.error("*** Error remove_punctuation: ",  e)
        return text

    def clean_text(self, text):
        try:
            text = re.sub(r'[\U00010000-\U0010ffff]', ' labelemoji ', text)
            text = self.proper_encoding(text)
            text = text.lower()  # converts to lowercase
            text = re.sub(
                r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|' +
                '(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                'labelurl', text)  # removes URLs
            text = re.sub("@([A-Za-z0-9_]{1,40})", "labelmention", text)  # removes mentions
            text = re.sub("#([A-Za-z0-9_]{1,40})", "labelhashtag", text)  # removes hastags
            text = self.remove_punctuation(text)  # removes punctuation
            text = self.delete_special_characters(text)  # removes special char
        except Exception as e:
            self.log_file.error("*** Error clean_tweet: ",  e)
        return text

    def __str__(self):
        return 'GetTextPreProcess'

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def process(self, text_process):
        result_text = text_process
        # self.log_file.info('result_text', result_text)
        # self.log_file.info('self.configuration', self.configuration)
        if self.configuration.get('clean_text', False) is True:
            result_text = self.clean_text(result_text)
        if GetTextPreProcess.configuration.get('type_nlp', '') == 'tagger':
            result_text = self.tagger_spacy(result_text)
            if GetTextPreProcess.configuration['label'] is None:
                return result_text
            else:
                result_text = " ".join([item[GetTextPreProcess.configuration['label']] for item in result_text])
                return result_text
        elif GetTextPreProcess.configuration.get('type_nlp', '') == 'dependency':
            result_text = self.dependency_spacy(result_text)
            # self.log_file.info('result', result)
            if GetTextPreProcess.configuration['label'] is None:
                return result_text
            else:
                result_text = " ".join(["" if GetTextPreProcess.configuration['label'] not in item else
                                        item[GetTextPreProcess.configuration['label']] for item in result_text])
                # self.log_file.info(return_string)
                return result_text

        if GetTextPreProcess.configuration.get('type_nlp_blob', '') == 'tagger':
            blob = TextBlob(result_text)
            words_tagger = blob.tags
            result_text = " ".join([pos_tag for word, pos_tag in words_tagger])
            return result_text
        elif GetTextPreProcess.configuration.get('type_nlp_blob', '') == 'dependency':
            blob = TextBlob(result_text)
            words_nouns = blob.noun_phrases
            result_text = " ".join(words_nouns)
            return result_text
        return result_text

    def tagger_spacy(self, text):
        result = []
        try:
            doc = self.nlp(text)  # []  # self.nlp(text)
            for token in doc:
                item = {}
                item['text'] = token.text
                item['lemma'] = token.lemma_
                item['pos'] = token.pos_
                item['tag'] = token.tag_
                item['dep'] = token.dep_
                item['shape'] = token.shape_
                item['is_alpha'] = token.is_alpha
                item['is_stop'] = token.is_stop
                item['is_digit'] = token.is_digit
                item['is_punct'] = token.is_punct
                result.append(item)
                # self.log_file.info(item)
        except Exception as e:
            self.log_file.error("\t** Error tagger: ", str(e))
        return result

    def dependency_spacy(self, text):
        result = []
        try:
            doc = self.nlp(text)
            for chunk in doc.noun_chunks:
                item = {}
                item['chunk'] = chunk
                item['text'] = chunk.root.text
                item['pos_'] = chunk.root.pos_
                item['dep_'] = chunk.root.dep_
                item['tag_'] = chunk.root.tag_
                item['head_text'] = chunk.root.head.text
                item['head_pos'] = chunk.root.head.pos_
                item['children'] = [{'child': child, 'pos_': child.pos_, 'dep_': child.dep_,
                                     'tag_': child.tag_, 'head.text': child.head.text,
                                     'head.pos_': child.head.pos_} for child in chunk.root.children]
                result.append(item)
        except Exception as e:
            self.log_file.error("\t** Error dependency", str(e))
        return result


if __name__ == '__main__':
    configuration_process = {'name_method': "docs_clear", 'clean_text': True, 'tracer': True}
    text_1 = 'ADQUIRIR ELEMENTOS DE PROTECCIÓN PERSONAL Y SEGURIDAD'
    pre_process = GetTextPreProcess()
    pre_process.configuration = configuration_process
    text_2 = pre_process.process(text_1)
    pre_process.log_file.info('\t Value old: {} new: {}'.format(text_1, text_2))
    # data = LoadDataClassifier()
    # data.read_source()
    configuration_folder = {'type': 'excel', 'source': 'DatsetCazador.xlsx', 'field_text': 'text',
                            'field_class': 'class_2'}
    data = LoadDataClassifier()
    data.configuration = configuration_folder
    # data.processor_text = GetTextPreProcess()
    data.processor_text.configuration = configuration_process
    data.read_source()
