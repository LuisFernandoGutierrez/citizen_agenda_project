
from src.util.data_acces import DataBaseAccessMongo
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import spacy
import re
import networkx as nx
import math
import configparser
import os
import csv
import datetime
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import SnowballStemmer
import json
import uuid
import threading
import queue
from bson import json_util
from root import DIR_CONFIG, DIR_OUTPUT

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class LinguisticsTwitter:
    def __init__(self, mongo_name=None):
        self.config = configparser.ConfigParser()
        self.config.read(DIR_CONFIG + 'config.ini')
        self.mongo_name = mongo_name
        self.coll_profile = ""
        self.coll_follower = ""
        self.coll_tweets = ""
        self.coll_mention = ""
        self.coll_hashtag = ""
        self.config_path = {}
        self.client_mongo = DataBaseAccessMongo()
        self.stop_words_spanish = nltk.corpus.stopwords.words('spanish')
        self.stemmer = SnowballStemmer('spanish')
        self.graph_tf_idf = nx.DiGraph(name="tf_idf")
        self.node_size_tf_idf = {}
        self.node_polarity_tf_idf = {}
        self.node_type_tf_idf = {}
        self.q_topic = queue.Queue()
        self.num_worker_threads = 8
        self.university_text = {}
        self.university_tweets = {}
        self.count_profile = 0
        self.name_query_count = 0
        self.names_data_query = []
        self.nlp = spacy.load('es_core_news_sm')

    def proper_encoding(self, text):
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return text.lower()

    def get_all(self):
        university_text = {}
        tweets = self.client_mongo.select(mongo_db=self.mongo_name,
                                          mongo_db_coll=self.config['MONGO']['coll_tweets'],
                                          projection={'text': 1, 'user.screen_name': 1})
        count_message = 0
        print('len(tweets): ', len(tweets))
        for tweet in tweets:

            current_screen_name = tweet['user']['screen_name']
            count_message += 1

            text_local = ' '.join(re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweet['text'],
                                         re.UNICODE, flags=re.MULTILINE).split())
            text_local = self.proper_encoding(text_local)
            print("\t** ", count_message, text_local)

            if current_screen_name in university_text:
                university_text[current_screen_name] += self.lemmatization(text_local) + ' '
            else:
                university_text[current_screen_name] = self.lemmatization(text_local) + ' '

        self.topic_modelling('all', 'all', university_text.values())
        self.tf_idf_text(university_text)

    def tf_idf_text(self, university_text, name_file='tf_idf'):
        date_file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        print("###### Calculo de TF-IDF ######")
        tfidf = TfidfVectorizer(token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}', stop_words=self.stop_words_spanish,
                                lowercase=True,  ngram_range=(2, 2), max_features=100) # min_df=0.6, max_df=0.9,
        # ngram_range=(1, 2)
        tfs = tfidf.fit_transform(university_text.values())
        #print(university_text)
        print("n_docs: %d, n_features: %d" % tfs.shape)

        docs_num, feature_num = tfs.shape
        feature_names = tfidf.get_feature_names()
        feature_values = {}
        print("###### Calculo de Feature Names ######")
        for x in range(0, feature_num):
            feature_values[feature_names[x]] = tfs[0, x]
            print(" # ", x, " - ", feature_names[x], " - ", tfs[0, x])
            # print(" # ", x, " - ", feature_names[x], " - ", [tfs[n, x] for n in range(0, docs_num)])
        self.save_csv_tf_idf(name_file, date_file, feature_names, list(university_text.keys()), tfs)
        self.save_graph_tf_idf(name_file, date_file, feature_names, list(university_text.keys()), tfs)
        # count = 0
        # for key, value in [(k, feature_values[k]) for k in sorted(feature_values, key=feature_values.get, reverse=True)]:
        #     count += 1
        #     print("# %s - %s: %s" % (count, key, value))

    def save_csv_tf_idf(self, name_model, date_file,  feature_names, key_values, tfs):
        docs_num, feature_num = tfs.shape
        with open(DIR_OUTPUT + "{0}_model_{1}.csv".format(name_model, date_file), 'w') as outcsv:
            writer = csv.writer(outcsv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            title_word = ['#', 'feature']
            title_word += key_values
            writer.writerow(title_word)

            for x in range(0, feature_num):
                row = [x, feature_names[x]]
                row += [tfs[n, x] for n in range(0, docs_num)]
                print(" # ", x, " - ", feature_names[x], " - ", [tfs[n, x] for n in range(0, docs_num)])
                writer.writerow(row)

    def save_graph_tf_idf(self, name_model, date_file,  feature_names, key_values, tfs):
        docs_num, feature_num = tfs.shape
        print(key_values)
        title_word = ['#', 'feature']
        title_word += key_values

        for x in range(0, feature_num):
            self.graph_tf_idf.add_node(feature_names[x], type="word", color="red", level="w")
            self.node_type_tf_idf[feature_names[x]] = 1
            self.node_size_tf_idf[feature_names[x]] = 1

            for n in range(0, docs_num):
                if tfs[n, x] > 0:
                    self.node_size_tf_idf[feature_names[x]] += 1
                    #print('repetida ', feature_names[x], self.node_size_tf_idf[feature_names[x]] )

                #print(key_values[n])
                self.graph_tf_idf.add_node(key_values[n], type="university", color="blue", level="u")
                self.graph_tf_idf.add_edge(feature_names[x], key_values[n], weight=tfs[n, x])
                if key_values[n] not in self.node_size_tf_idf:
                    self.node_type_tf_idf[key_values[n]] = 2
                    self.node_size_tf_idf[key_values[n]] = 1
                else:
                    self.node_size_tf_idf[key_values[n]] += 1

            print("Graph # ", x, " - ", feature_names[x], " - ", [tfs[n, x] for n in range(0, docs_num)])

        print("** finish create_graph_tf_idf")

        nx.write_adjlist(self.graph_tf_idf, DIR_OUTPUT + "{0}_{1}_adjlist_{2}.csv"
                         .format(self.mongo_name, name_model, date_file))
        nx.write_pajek(self.graph_tf_idf, DIR_OUTPUT + "{0}_{1}_pajek_{2}.net"
                       .format(self.mongo_name, name_model, date_file))

        local_node_size = [math.log(float(self.node_size_tf_idf[v])) for v in self.graph_tf_idf]
        # print(self.node_size_tf_idf.keys(), self.node_size_tf_idf.values())
        # print(local_node_size)
        with open(DIR_OUTPUT + "{0}_{1}_pajek_{2}.vec"
                .format(self.mongo_name, name_model, date_file), "w") as text_file:
            print(f"*Vertices {len(local_node_size)}", file=text_file)
            for item in local_node_size:
                print(f"{item}", file=text_file)

        for v in self.graph_tf_idf.nodes:
            self.graph_tf_idf.nodes[v]['size'] = str(float(self.node_type_tf_idf[v]))

        local_node_type = [int(self.node_type_tf_idf[v]) for v in self.graph_tf_idf]

        with open(DIR_OUTPUT + "{0}_{1}_pajek_{2}.clu"
                .format(self.mongo_name, name_model, date_file), "w") as text_file:
            print(f"*Vertices {len(local_node_type)}", file=text_file)
            for item in local_node_type:
                print(f"{item}", file=text_file)
        """
        local_node_polarity = [1 if self.node_polarity_tf_idf[v] > 0 else 2 if self.node_polarity_tf_idf[v] < 0 else 0
                               for v in self.graph_tf_idf]

        with open(ROOT_DIR + "/output/{0}_hashtag_pajek_{1}.clu"
                .format(self.mongo_name, date_file), "w") as text_file:
            print(f"*Vertices {len(local_node_polarity)}", file=text_file)
            for item in local_node_polarity:
                print(f"{item}", file=text_file)
        """
        print("** finish export_graph_tf_idf")

    def lemmatization(self, text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        doc = self.nlp(text)
        texts_out = " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                              token.pos_ in allowed_postags])
        return texts_out

    def lemma_tokens(self, text):
        doc = self.nlp(text)
        response = []
        for token in doc:
            response.append(token.lemma_)
        return ' '.join(response)

    def stemmer_tokens(self, tokens, stemmer):
        stemmed = []
        for item in tokens:
            if item.isalpha():
                stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text.lower())
        stems = self.stem_tokens(tokens, self.stemmer)
        return stems

    def topic_modelling(self, name_file, header, data, num_topics=10):
        date_file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        print("topic_modelling :", len(data))

        vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=self.stop_words_spanish,
                                     lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        data_vectorized = vectorizer.fit_transform(data)

        try:
            # Build a Latent Dirichlet Allocation Model
            lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online')
            lda_Z = lda_model.fit_transform(data_vectorized)
            print("\t**lda:", lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
        except Exception as e:
            lda_model = {}
            lda_Z = {}
            print("\t** ERROR: lda ", str(e))

        try:
            # Build a Non-Negative Matrix Factorization Model
            nmf_model = NMF(n_components=num_topics)
            nmf_Z = nmf_model.fit_transform(data_vectorized)
            print("\t**nmf:", nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
        except Exception as e:
            nmf_model = {}
            nmf_Z = {}
            print("\t** ERROR: nmf ", str(e))

        try:
            # Build a Latent Semantic Indexing Model
            lsi_model = TruncatedSVD(n_components=num_topics)
            lsi_Z = lsi_model.fit_transform(data_vectorized)
            print("\t**lsi:", lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
        except Exception as e:
            lsi_model = {}
            lsi_Z = {}
            print("\t** ERROR: lsi ", str(e))

        print("=" * 20)
        print("\tSave all models")

        self.save_csv_topic_modelling_all(name_file + "_all_topic", date_file, lda_model,
                                          nmf_model, lsi_model, vectorizer)

    def save_csv_topic_modelling(self, name_model, date_file,  model, vectorizer, top_n=10):
        with open(DIR_OUTPUT + "{0}_model_{1}.csv".format(name_model, date_file), 'w') as outcsv:
            writer = csv.writer(outcsv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(['MODEL', 'TOPIC INDEX', 'TOPIC NAME', 'AVERAGE'])
            for idx, topic in enumerate(model.components_):
                print("Topic %d:" % (idx))
                print([(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])
                for i in topic.argsort()[:-top_n - 1:-1]:
                    writer.writerow(
                        [name_model, "topic {0}".format(idx), vectorizer.get_feature_names()[i], topic[i]])

    def save_csv_topic_modelling_all(self, name_model, date_file,  model_lda, model_nmf, model_lsi, vectorizer, top_n=10):
        #print(ROOT_DIR)
        with open(DIR_OUTPUT + "{0}_model_{1}.csv".format(name_model, date_file), 'w') as outcsv:
            writer = csv.writer(outcsv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            print("LDA Model:")
            lda_title = ['MODEL_LDA', 'TOPIC INDEX', 'TOPIC NAME', 'AVERAGE']
            lda_row = []
            try:
                for idx, topic in enumerate(model_lda.components_):
                    print("Topic %d:" % (idx))
                    print([(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])
                    for i in topic.argsort()[:-top_n - 1:-1]:
                        lda_row.append([name_model, "topic {0}".format(idx), vectorizer.get_feature_names()[i], topic[i]])
            except Exception as e:
                print("*** ERROR: LDA Model:", str(e))

            print("=" * 20)

            print("NMF Model:")
            nmf_title = ['MODEL_NMF', 'TOPIC INDEX', 'TOPIC NAME', 'AVERAGE']
            nmf_row = []

            try:
                for idx, topic in enumerate(model_nmf.components_):
                    print("Topic %d:" % (idx))
                    print([(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])
                    for i in topic.argsort()[:-top_n - 1:-1]:
                        nmf_row.append([name_model, "topic {0}".format(idx), vectorizer.get_feature_names()[i], topic[i]])
            except Exception as e:
                print("*** ERROR: NMF Model:", str(e))

            print("=" * 20)

            print("LSI Model:")
            lsi_title = ['MODEL_LSI', 'TOPIC INDEX', 'TOPIC NAME', 'AVERAGE']
            lsi_row = []
            try:
                for idx, topic in enumerate(model_lsi.components_):
                    print("Topic %d:" % (idx))
                    print([(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])
                    for i in topic.argsort()[:-top_n - 1:-1]:
                        lsi_row.append([name_model, "topic {0}".format(idx), vectorizer.get_feature_names()[i], topic[i]])
            except Exception as e:
                print("*** ERROR: LSI Model:", str(e))

            print("=" * 20)

            try:
                writer.writerow(lda_title + nmf_title + lsi_title)
                for x in range(0, len(lda_row)):
                    writer.writerow(lda_row[x] + nmf_row[x] + lsi_row[x])
            except Exception as e:
                print("*** ERROR: LSI Model:", str(e))
        print("=" * 20)


if __name__ == "__main__":
    linguistic = LinguisticsTwitter('medellin_2020-10-05-10-31')
    linguistic.get_all()
