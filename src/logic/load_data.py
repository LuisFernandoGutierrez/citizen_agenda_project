from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import datetime
import csv
from xml.dom import minidom
import os
import re
import unicodedata
import random
import time
import spacy
from textblob import TextBlob, Word, Blobber
import nltk
from nltk.stem import SnowballStemmer
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger
import nltk
from nltk.corpus import sentiwordnet as swn
# Do this first, that'll do something eval()
# to "materialize" the LazyCorpusLoader
import queue
import threading
# next(swn.all_senti_synsets())


class DataClassifier:

    def __init__(self):
        print("Start process classification")
        self.result_text = []
        self.result_emotions = []
        self.result_interest = []
        self.result_gender = []
        self.result_age_range = []
        self.result_year_birth = []
        self.result_evidence = []
        self.docs_raw = {}
        self.docs_raw_array = []
        self.x_docs = []
        self.y_labels = []
        self.stemmer = SnowballStemmer('spanish')
        self.stop_words_spanish = set(nltk.corpus.stopwords.words('spanish'))
        self.nlp = spacy.load('es_core_news_sm')

    def stemmer_tokens(self, tokens):
        stemmed = []
        for item in tokens:
            stemmed.append(self.stemmer.stem(item))
        return stemmed

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [w for w in tokens if not w.lower() in self.stop_words_spanish]
        stems = self.stemmer_tokens(filtered_tokens)
        return ' '.join(stems)

    def lemmatization(self, text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        doc = self.nlp(text)
        texts_out = " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                              token.pos_ in allowed_postags])
        return texts_out

    @staticmethod
    def proper_encoding(text):
        # print('text: ', text)
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return text

    @staticmethod
    def delete_special_characters(text):
        text = re.sub('\/|\\|\\.|\,|\;|\:|\n|\?|\)|\(|\!|\¡|\¿|\'|\t', ' ', text)
        text = re.sub("\s+\w\s+", " ", text)
        text = re.sub("\.", "", text)
        text = re.sub("|", "", text)
        text = re.sub("@", "", text)
        return text.lower()

    @staticmethod
    def remove_punctuation(text):
        try:
            punctuation = {'/', '"', '(', ')', '.', ',', '%', ';', '?', '¿', '!', '¡',
                           ':', '#', '$', '&', '>', '<', '-', '_', '°', '|', '¬', '\\', '*', '+',
                           '[', ']', '{', '}', '=', "'", '@'}
            for sign in punctuation:
                text = text.replace(sign, '')
            return text
        except:
            # Logging.write_standard_error(sys.exc_info())
            return None

    @staticmethod
    def clean_text(text):
        try:
            text_out = re.sub(r'[\U00010000-\U0010ffff]', ' labelemoji ', text)
            text_out = DataClassifier.proper_encoding(text_out)
            text_out = text_out.lower()  # converts to lowercase
            text_out = re.sub(
                r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                'labelurl', text_out)  # removes URLs
            text_out = re.sub("@([A-Za-z0-9_]{1,40})", "labelmention", text_out)  # removes mentions
            text_out = re.sub("#([A-Za-z0-9_]{1,40})", "labelhashtag", text_out)  # removes hastags

            text_out = DataClassifier.remove_punctuation(text_out)  # removes punctuation
            text_out = DataClassifier.delete_special_characters(text_out)  # removes special char
            #text_out = ' '.join(
            #    [word for word in text_out.split() if word not in stopwords.words(lang)])  # removes stopwords
            return text_out
        except:
            # Logging.write_standard_error(sys.exc_info())
            return None

    def read_file(self, name_file):

        path_folder = DIR_CLASSIFICATION_TRAIN
        if os.path.exists(path_folder):

            if 'gender' in name_file:
                file_process = path_folder + 'names_users_with_gender_birth.csv'
                df_filter = pd.read_csv(file_process, encoding='UTF-8', sep=';', quotechar='"', na_values=['.', '??'])
                self.result_text = df_filter['name'].tolist()
                self.result_gender = df_filter['gender'].tolist()
                self.result_year_birth = df_filter['year_decade'].tolist()
                print("\t** read_file : text:{0} class:{1}".format(len(self.result_text), len(self.result_gender)))
            if 'age_range' in name_file:
                file_process = path_folder + 'tweets_with_age/tweets_with_age.csv'
                print(file_process)
                df_filter = pd.read_csv(file_process, encoding='UTF-8', sep='\t')
                df_filter_null = df_filter[df_filter['text'].notnull()]
                df_filter_null = df_filter_null[df_filter_null['age_range'].notnull()]
                # local_result_text = df_filter_null['text'].tolist()
                self.result_text = df_filter_null['text'].tolist()
                # count = 0
                # for item_text in local_result_text:
                #     count += 1
                #     text_age_lema = self.tokenize(item_text) # self.lemmatization(item_text)
                #     self.result_text.append(' '.join([item_text, text_age_lema]))
                #     print('\t lemma {} of {}'.format(count, len(local_result_text)))
                self.result_age_range = df_filter_null['age_range'].tolist()
                print("\t** read_file : text:{0} class:{1}".format(len(self.result_text), len(self.result_age_range)))
            if 'evidence' in name_file:
                file_process = path_folder + 'evidence_class.csv'
                df_filter = pd.read_csv(file_process, encoding='UTF-8', sep=';', quotechar='"', na_values=['.', '??'])
                df_filter_null = df_filter[df_filter['clasif'].notnull()]
                df_filter_null_null = df_filter_null[df_filter_null['text'].notnull()]
                self.result_text = df_filter_null_null['text'].tolist()
                self.result_evidence = df_filter_null_null['clasif'].tolist()
                print("\t** read_file : text:{0} class:{1}".format(len(self.result_text), len(self.result_evidence)))
            # print("\t** read_file : text:{0} class:{1}".format(len(self.result_text), len(self.result_gender)))
            if 'emotions' in name_file:
                file_process = path_folder + 'l_emotions_utf8.tsv'
                df = pd.read_csv(file_process, sep='\t', encoding='utf-8')
                emotions_key = ['Alegria', 'Enojo', 'Miedo', 'Repulsion', 'Sorpresa', 'Tristeza']
                dict_emotions = {}
                text_emotions = []
                label_emotions = []
                for item in emotions_key:
                    df_filter_1 = df[df['type1'] == item]
                    df_filter_2 = df[df['type2'] == item]
                    df_filter_3 = df[df['type3'] == item]
                    words_list_1 = list(df_filter_1['word'].values)
                    words_list_2 = list(df_filter_2['word'].values)
                    words_list_3 = list(df_filter_3['word'].values)
                    words_list_all = words_list_1 + words_list_2 + words_list_3
                    dict_emotions[item] = ' '.join(words_list_all)
                    text_emotions.append(' '.join(words_list_all))
                    label_emotions.append(item)
                    for i in range(1, 2000):
                        text_emotion_raw = ' '.join(random.sample(words_list_all, 100))
                        text_emotion_lema = self.lemmatization(text_emotion_raw)
                        text_emotions.append(' '.join([text_emotion_raw, text_emotion_lema]))
                        label_emotions.append(item)
                self.result_text = text_emotions
                self.result_emotions = label_emotions
            if 'interest' in name_file:
                file_process = path_folder + 'l_general_interest.tsv'
                df = pd.read_csv(file_process, sep='\t', encoding='utf-8')
                interest_key = ['astrologia', 'deporte', 'entretenimiento', 'familia', 'moda', 'musica', 'noticias',
                                'politica', 'religion', 'salud', 'tecnologia']
                dict_interest = {}
                text_interest = []
                label_interest = []
                for item in interest_key:
                    df_filter_1 = df[df['topic1'] == item]
                    df_filter_2 = df[df['topic2'] == item]
                    words_list_1 = list(df_filter_1['word'].values)
                    words_list_2 = list(df_filter_2['word'].values)
                    words_list_all = words_list_1 + words_list_2
                    dict_interest[item] = ' '.join(words_list_all)
                    text_interest.append(' '.join(words_list_all))
                    label_interest.append(item)
                    for i in range(1, 2000):
                        text_emotion_raw = ' '.join(random.sample(words_list_all, 20))
                        text_emotion_lema = self.lemmatization(text_emotion_raw)
                        text_interest.append(' '.join([text_emotion_raw, text_emotion_lema]))
                        label_interest.append(item)
                self.result_text = text_interest
                self.result_interest = label_interest
        else:
            print("\t** #:ERROR Folder {0} not found!".format(path_folder))

    def read_folder(self):
        # print(DIR_KNOWLEDGE_BASE)
        sources_files = [DIR_KNOWLEDGE_BASE]
        count_file = 0
        for path_folder_lang in sources_files:
            if os.path.exists(path_folder_lang):
                print("\t** #:READ in path {0}.".format(path_folder_lang))
                for subdir, dirs, files in os.walk(path_folder_lang):
                    for file in files:
                        count_file += 1
                        file_path = subdir + os.path.sep + file
                        list_dir = subdir.split(os.sep)
                        name_class = list_dir[len(list_dir) - 2].replace(' ', '_')
                        if name_class not in self.docs_raw:
                            self.docs_raw[name_class] = []

                        try:
                            if count_file % 200 == 0:
                                print("\t** #: Files Loads : {0}".format(count_file))

                            if '.txt' in file_path:
                                txt_file = ''
                                # print('name_class', name_class, 'file_path', file_path)
                                with open(file_path, newline='', encoding='UTF-8') as file_text_buffer:
                                    txt_file = file_text_buffer.read()
                                self.docs_raw[name_class].append(
                                    {'class': name_class, 'text': txt_file, 'file_name': file_path})
                                self.docs_raw_array.append(
                                    {'class': name_class, 'text': txt_file, 'file_name': file_path})
                                self.x_docs.append(txt_file)
                                self.y_labels.append(name_class)
                                print('\t** # Class: {} - File {} - Count {}'.format(
                                    name_class, file_path, len(self.docs_raw[name_class])))
                        except Exception as e:
                            print('\t ERROR : ', e)
                print("\t** read_folder docs_raw : ", len(self.docs_raw_array))
            else:
                print("\t** #:ERROR Source not found!. {0}".format(path_folder_lang))

    def read_file_sampling(self):

        path_folder = DIR_CLASSIFICATION_TRAIN
        if os.path.exists(path_folder):

            file_process = path_folder + 'sugar_train.csv'
            df = pd.read_csv(file_process, encoding='UTF-8', sep=';', quotechar='"', na_values=['.', '??'])
            df_filter_null = df[df['case_documentation'].notnull()]

            b_reason = df_filter_null['reason'].value_counts()
            df_cut_reason = pd.DataFrame(columns=['item', 'value', 'q'])
            df_cut_reason['item'] = b_reason.index
            df_cut_reason['value'] = b_reason.values
            df_cut_reason['q'] = list(pd.qcut(df_filter_null['reason'].value_counts(), duplicates='drop',
                                              q=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], labels=False))
            # print(df_cut_reason)
            label_encoder_reason = LabelEncoder()
            print('\t\t -- encoder reason: ', label_encoder_reason.fit_transform(b_reason.index))
            print('\t\t -- label reason: ', list(label_encoder_reason.classes_))

            b_process = df_filter_null['process'].value_counts()
            df_cut_process = pd.DataFrame(columns=['item', 'value', 'q'])
            df_cut_process['item'] = b_process.index
            df_cut_process['value'] = b_process.values
            df_cut_process['q'] = list(pd.qcut(df_filter_null['process'].value_counts(), duplicates='drop',
                                               q=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], labels=False))
            # print(df_cut_process)
            label_encoder_process = LabelEncoder()
            print('\t\t -- encoder process: ', label_encoder_process.fit_transform(b_process.index))
            print('\t\t -- label process: ', list(label_encoder_process.classes_))

            for item in b_reason.index:
                self.result_subcategory_by_reason[item.lower()] = {'stats': None, 'text': [], 'label': []}
                df_subcategory = df_filter_null[df_filter_null['reason'] == item]
                # print(df_subcategory.count(), df_filter_null.count())
                b_subcategory_motive = df_subcategory['subcategory_motive'].value_counts()
                # print(b_subcategory_motive)
                # print(df_subcategory['subcategory_motive'].unique())
                df_cut_subcategory_motive = pd.DataFrame(columns=['item', 'value', 'q'])
                df_cut_subcategory_motive['item'] = b_subcategory_motive.index
                df_cut_subcategory_motive['value'] = b_subcategory_motive.values
                df_cut_subcategory_motive['q'] = list(pd.qcut(df_subcategory['subcategory_motive'].value_counts(),
                                                              duplicates='drop',
                                                      q=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], labels=False))
                # print(df_cut_subcategory_motive)
                for item_loc, value_loc, q_loc in df_cut_subcategory_motive.values.tolist():
                    # item_loc, value_loc, q_loc
                    if q_loc < 2:
                        df_subcategory.loc[df_subcategory['subcategory_motive'] == item_loc, 'subcategory_motive'] = 'Z_Other'
                if len(df_subcategory['case_documentation'].tolist()) > 100 and len(b_subcategory_motive.values) > 2:
                    self.result_subcategory_by_reason[item.lower()]['text'] = df_subcategory['case_documentation'].tolist()
                    self.result_subcategory_by_reason[item.lower()]['label'] = df_subcategory['subcategory_motive'].tolist()
                    print('\t\t *** subcategory_motive by reason :', item,
                          len(self.result_subcategory_by_reason[item.lower()]['text']),
                          len(self.result_subcategory_by_reason[item.lower()]['label']))
                    b_subcategory_motive = df_subcategory['subcategory_motive'].value_counts()
                    label_encoder_subcategory_motive = LabelEncoder()
                    print('\t\t -- encoder subcategory_motive: ', item.lower(), label_encoder_subcategory_motive.fit_transform(b_subcategory_motive.index))
                    print('\t\t -- label subcategory_motive: ', item.lower(), list(label_encoder_subcategory_motive.classes_))
                # print(df_subcategory['subcategory_motive'].value_counts())

            self.result_text = df_filter_null['case_documentation'].tolist()
            self.result_reason = df_filter_null['reason'].tolist()
            self.result_process = df_filter_null['process'].tolist()
            self.result_subcategory = df_filter_null['subcategory_motive'].tolist()

            print("\t** read_file : text:{0} reason:{1} process:{2} subcategory:{3}".format(len(self.result_text),
                                                                                            len(self.result_reason),
                                                                                            len(self.result_process),
                                                                                            len(
                                                                                                self.result_subcategory)))
        else:
            print("\t** #:ERROR Folder {0} not found!".format(path_folder))

    def get_data_collection(self, collection, test_size=0.4):
        current_coll = None
        if collection is 'reason':
            current_coll = self.result_reason
        elif collection is 'process':
            current_coll = self.result_process
        elif collection is 'subcategory':
            current_coll = self.result_subcategory
        else:
            raise Exception('This collection not found: {}'.format(collection))

        text_train, text_test, label_train, label_test = \
            train_test_split(self.result_text, current_coll, test_size=test_size, random_state=1000)

        print('\t** get_data_collection text_train:{0} text_test:{1} label_train:{2} label_test:{3}'.format(
            len(text_train), len(text_test), len(label_train), len(label_test)))
        return text_train, text_test, label_train, label_test

    def get_data_collection_sampling(self, collection, sub_collection=None, over_sampling=False, test_size=0.4):
        current_label = []
        if collection is 'gender':
            current_text = self.result_text
            current_label = self.result_gender
        elif collection is 'age_range':
            current_text = self.result_text
            current_label = self.result_age_range
        elif collection is 'year':
            current_text = self.result_text
            current_label = self.result_year_birth
        elif collection is 'evidence':
            current_text = self.result_text
            current_label = self.result_evidence
        elif collection is 'emotions':
            current_text = self.result_text
            current_label = self.result_emotions
        elif collection is 'interest':
            current_text = self.result_text
            current_label = self.result_interest
        elif collection is 'knowledge_base':
            current_text = self.x_docs
            current_label = self.y_labels
        else:
            raise Exception('This collection not found: {}'.format(collection))
        print(current_text[:5], current_label[:5])
        text_train, text_test, label_train, label_test = \
            train_test_split(current_text, current_label, test_size=test_size, random_state=1000)

        print('\t** get_data_collection {0}-{1} text_train:{2} text_test:{3} label_train:{4} label_test:{5}'.format(
            collection, sub_collection, len(text_train), len(text_test), len(label_train), len(label_test)))
        return text_train, text_test, label_train, label_test


class GetTextPreProcess:
    # https://stackoverflow.com/questions/19151/build-a-basic-python-iterator
    data_pan = None
    configuration = {}

    def __init__(self, text_users):
        self.text_users = text_users
        self.high = len(text_users)
        self.current = 0
        # self.nlp = spacy.load('en_core_web_sm')

        print('\t- Configuration Process:  name_method: {0} | type_nlp: {1} | type_nlp_blob: {2} | label:{3} | '
              'clean_text: {4}'.format(GetTextPreProcess.configuration['name_method'],
              "" if 'type_nlp' not in GetTextPreProcess.configuration else GetTextPreProcess.configuration['type_nlp'],
              "" if 'type_nlp_blob' not in GetTextPreProcess.configuration else GetTextPreProcess.configuration['type_nlp_blob'],
              "" if 'label' not in GetTextPreProcess.configuration else GetTextPreProcess.configuration['label'],
              "" if 'clean_text' not in GetTextPreProcess.configuration else GetTextPreProcess.configuration['clean_text'],
              "" if 'tracer' not in GetTextPreProcess.configuration else GetTextPreProcess.configuration['tracer']))

    def __str__(self):
        return 'GetTextPreProcess'

    @staticmethod
    def get_name():
        return GetTextPreProcess.configuration['name_method'] \
            if GetTextPreProcess.configuration['name_method'] is not None else 'user_pre_process'

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.high:
            text_old = self.text_users[self.current]
            text_local = self.process(text_old)
            self.current += 1
            if 'tracer' in GetTextPreProcess.configuration:
                if GetTextPreProcess.configuration['tracer']:
                    if (self.current % 10) == 0 or self.current == self.high:
                        print('\t- Configuration Process:  name_method: {0} | type_nlp: {1} | type_nlp_blob: {2} | '
                              'label:{3} | clean_text: {4}'.format(
                                GetTextPreProcess.configuration['name_method'],
                                "" if 'type_nlp' not in GetTextPreProcess.configuration else
                                GetTextPreProcess.configuration['type_nlp'],
                                "" if 'type_nlp_blob' not in GetTextPreProcess.configuration else
                                GetTextPreProcess.configuration['type_nlp_blob'],
                                "" if 'label' not in GetTextPreProcess.configuration else
                                GetTextPreProcess.configuration['label'],
                                "" if 'clean_text' not in GetTextPreProcess.configuration else
                                GetTextPreProcess.configuration['clean_text'],
                                "" if 'tracer' not in GetTextPreProcess.configuration else
                                GetTextPreProcess.configuration['tracer']))
                        print('\t- old item ', self.current, ' of ', self.high, ' text ', text_old[:150] + '...')
                        print('\t- new item ', self.current, ' of ', self.high, ' text ', text_local[:150] + '...')
            return text_local
        else:
            raise StopIteration

    def process(self, text_process):
        result_text = text_process
        if 'clean_text' in GetTextPreProcess.configuration:
            if GetTextPreProcess.configuration['clean_text'] is True:
                result_text = DataClassifier.clean_text(result_text)
        if 'type_nlp' in GetTextPreProcess.configuration:
            if GetTextPreProcess.configuration['type_nlp'] is 'tagger':
                result_text = self.tagger_spacy(result_text)
                if GetTextPreProcess.configuration['label'] is None:
                    return result_text
                else:
                    result_text = " ".join([item[GetTextPreProcess.configuration['label']] for item in result_text])
                    return result_text
            if GetTextPreProcess.configuration['type_nlp'] is 'dependency':
                result_text = self.dependency_spacy(result_text)
                # print('result', result)
                if GetTextPreProcess.configuration['label'] is None:
                    return result_text
                else:
                    result_text = " ".join(["" if GetTextPreProcess.configuration['label'] not in item else
                                            item[GetTextPreProcess.configuration['label']] for item in result_text])
                    # print(return_string)
                    return result_text
        if 'type_nlp_blob' in GetTextPreProcess.configuration:
            if GetTextPreProcess.configuration['type_nlp_blob'] is 'tagger':
                blob = TextBlob(result_text)
                words_tagger = blob.tags
                result_text = " ".join([pos_tag for word, pos_tag in words_tagger])
                return result_text
            if GetTextPreProcess.configuration['type_nlp_blob'] is 'dependency':
                blob = TextBlob(result_text)
                words_nouns = blob.noun_phrases
                result_text = " ".join(words_nouns)
                return result_text
        return result_text

    def tagger_spacy(self, text):
        result = []
        try:
            doc = []  # self.nlp(text)
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
                #print(item)
        except Exception as e:
            print("\t** Error tagger: ", str(e))
        return result

    def dependency_spacy(self, text):
        result = []
        try:
            doc = [] # self.nlp(text)
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
            print("\t** Error dependency", str(e))
        return result

# class GetTextPreProcess:
#     configuration = {}
#
#     def __init__(self):
#         self.text_users = []
#         self.high = 0
#         self.current = 0
#         self.nlp = spacy.load('es_core_news_sm')
#         # self.log_file = logging.getLogger('datasetBuild')
#
#     def proper_encoding(self, text):
#         try:
#             text = unicodedata.normalize('NFD', text)
#             text = text.encode('ascii', 'ignore')
#             text = text.decode("utf-8")
#         except Exception as e:
#             self.log_file.error("*** Error proper_encoding: ",  e)
#         return text
#
#     def delete_special_characters(self, text):
#         try:
#             text = re.sub(r'\/|\\|\\.|\,|\;|\:|\n|\?|\)|\(|\!|\¡|\¿|\'|\t', ' ', text)
#             text = re.sub(r"\s+\w\s+", " ", text)
#             text = re.sub("", "", text)
#             text = re.sub("|", "", text)
#             text = re.sub("@", "", text)
#         except Exception as e:
#             self.log_file.error("*** Error delete_special_characters: ",  e)
#         return text.lower()
#
#     def remove_punctuation(self, text):
#         try:
#             punctuation = {'/', '"', '(', ')', '.', ',', '%', ';', '?', '¿', '!', '¡',
#                            ':', '#', '$', '&', '>', '<', '-', '_', '°', '|', '¬', '\\', '*', '+',
#                            '[', ']', '{', '}', '=', "'", '@'}
#             for sign in punctuation:
#                 text = text.replace(sign, '')
#         except Exception as e:
#             self.log_file.error("*** Error remove_punctuation: ",  e)
#         return text
#
#     def clean_text(self, text):
#         try:
#             text = re.sub(r'[\U00010000-\U0010ffff]', ' labelemoji ', text)
#             text = self.proper_encoding(text)
#             text = text.lower()  # converts to lowercase
#             text = re.sub(
#                 r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|' +
#                 '(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
#                 'labelurl', text)  # removes URLs
#             text = re.sub("@([A-Za-z0-9_]{1,40})", "labelmention", text)  # removes mentions
#             text = re.sub("#([A-Za-z0-9_]{1,40})", "labelhashtag", text)  # removes hastags
#             text = self.remove_punctuation(text)  # removes punctuation
#             text = self.delete_special_characters(text)  # removes special char
#         except Exception as e:
#             self.log_file.error("*** Error clean_tweet: ",  e)
#         return text
#
#     def __str__(self):
#         return 'GetTextPreProcess'
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         raise StopIteration
#
#     def process(self, text_process):
#         result_text = text_process
#         # self.log_file.info('result_text', result_text)
#         # self.log_file.info('self.configuration', self.configuration)
#         if self.configuration.get('clean_text', False) is True:
#             result_text = self.clean_text(result_text)
#         if GetTextPreProcess.configuration.get('type_nlp', '') == 'tagger':
#             result_text = self.tagger_spacy(result_text)
#             if GetTextPreProcess.configuration['label'] is None:
#                 return result_text
#             else:
#                 result_text = " ".join([item[GetTextPreProcess.configuration['label']] for item in result_text])
#                 return result_text
#         elif GetTextPreProcess.configuration.get('type_nlp', '') == 'dependency':
#             result_text = self.dependency_spacy(result_text)
#             # self.log_file.info('result', result)
#             if GetTextPreProcess.configuration['label'] is None:
#                 return result_text
#             else:
#                 result_text = " ".join(["" if GetTextPreProcess.configuration['label'] not in item else
#                                         item[GetTextPreProcess.configuration['label']] for item in result_text])
#                 # self.log_file.info(return_string)
#                 return result_text
#
#         if GetTextPreProcess.configuration.get('type_nlp_blob', '') == 'tagger':
#             blob = TextBlob(result_text)
#             words_tagger = blob.tags
#             result_text = " ".join([pos_tag for word, pos_tag in words_tagger])
#             return result_text
#         elif GetTextPreProcess.configuration.get('type_nlp_blob', '') == 'dependency':
#             blob = TextBlob(result_text)
#             words_nouns = blob.noun_phrases
#             result_text = " ".join(words_nouns)
#             return result_text
#         return result_text
#
#     def tagger_spacy(self, text):
#         result = []
#         try:
#             doc = self.nlp(text)  # []  # self.nlp(text)
#             for token in doc:
#                 item = {}
#                 item['text'] = token.text
#                 item['lemma'] = token.lemma_
#                 item['pos'] = token.pos_
#                 item['tag'] = token.tag_
#                 item['dep'] = token.dep_
#                 item['shape'] = token.shape_
#                 item['is_alpha'] = token.is_alpha
#                 item['is_stop'] = token.is_stop
#                 item['is_digit'] = token.is_digit
#                 item['is_punct'] = token.is_punct
#                 result.append(item)
#                 # self.log_file.info(item)
#         except Exception as e:
#             self.log_file.error("\t** Error tagger: ", str(e))
#         return result
#
#     def dependency_spacy(self, text):
#         result = []
#         try:
#             doc = self.nlp(text)
#             for chunk in doc.noun_chunks:
#                 item = {}
#                 item['chunk'] = chunk
#                 item['text'] = chunk.root.text
#                 item['pos_'] = chunk.root.pos_
#                 item['dep_'] = chunk.root.dep_
#                 item['tag_'] = chunk.root.tag_
#                 item['head_text'] = chunk.root.head.text
#                 item['head_pos'] = chunk.root.head.pos_
#                 item['children'] = [{'child': child, 'pos_': child.pos_, 'dep_': child.dep_,
#                                      'tag_': child.tag_, 'head.text': child.head.text,
#                                      'head.pos_': child.head.pos_} for child in chunk.root.children]
#                 result.append(item)
#         except Exception as e:
#             self.log_file.error("\t** Error dependency", str(e))
#         return result

if __name__ == '__main__':
    data_list = DataClassifier()
    # data_list.read_file()
    data_list.read_folder()

