
from src.util.data_acces import DataBaseAccessMongo
import networkx as nx
import configparser
import csv
import datetime
import spacy
import queue
import threading
import time
import unicodedata
from datetime import datetime
from root import DIR_CONFIG, DIR_OUTPUT
from src.util.classification_lexicon import ClassificationLexiconBased
from src.logic.classification import ClassificationModel


class NetworkSubjectivity:
    def __init__(self, mongo_name):
        self.date_file = datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.config = configparser.ConfigParser()
        self.config.read(DIR_CONFIG + 'config.ini')

        self.coll_profile = ""
        self.coll_follower = ""
        self.coll_tweets = ""
        self.coll_mention = ""
        self.coll_hashtag = ""
        self.client_mongo = DataBaseAccessMongo()
        self.graph_mention = nx.DiGraph(name="Mention")
        self.graph_hashtag = nx.DiGraph(name="Hashtag")
        self.graph_follower = nx.DiGraph(name="Follower")

        self.node_size_mention = {}
        self.node_size_hashtag = {}
        self.node_size_follower = {}
        self.node_polarity_mention = {}
        self.node_polarity_hashtag = {}

        self.nlp = spacy.load('es_core_news_sm')
        file_process = 'lexicon_polarity_es_utf8.csv'
        self.cls_pol = ClassificationLexiconBased(file_process)
        self.best_models = ClassificationModel()
        self.best_models.build_models()
        # polarity
        self.q_hashtag = queue.Queue()
        self.q_mention = queue.Queue()
        self.q_polarity = queue.Queue()
        self.num_worker_threads = 8
        self.count_polarity = 0
        self.total_polarity = 0
        self.count_hashtag = 0
        self.count_mention = 0
        self.dict_hashtag = {}
        self.dict_mention = {}
        self.text_messages = []

        self.mongo_name = mongo_name
        self.coll_profile = self.config['MONGO']['COLL_PROFILE']
        self.coll_follower = self.config['MONGO']['COLL_FOLLOWER']
        self.coll_hashtag = self.config['MONGO']['COLL_HASHTAG']
        self.coll_mention = self.config['MONGO']['COLL_MENTION']
        self.coll_tweets = self.config['MONGO']['COLL_TWEETS']

    def worker_queue_polarity(self):
        while True:
            item = self.q_polarity.get()
            if item is None:
                break
            try:
                self.polarity_single_tweet(item)
            except Exception as e:
                print(e)
            self.q_polarity.task_done()

    def worker_queue_hashtag(self):
        while True:
            item = self.q_hashtag.get()
            if item is None:
                break
            self.polarity_single_hashtag(item)
            self.q_hashtag.task_done()

    def worker_queue_mention(self):
        while True:
            item = self.q_mention.get()
            if item is None:
                break
            self.polarity_single_mention(item)
            self.q_mention.task_done()

    def polarity_single_hashtag(self, item):
        self.count_hashtag += 1
        criteria_hashtag = {'id_source': self.dict_hashtag[item]['id_source'],
                            'id_hashtag': self.dict_hashtag[item]['id_hashtag']}

        update_fields = {'positive': self.dict_hashtag[item]['positive'],
                         'negative': self.dict_hashtag[item]['negative'],
                         'neutro': self.dict_hashtag[item]['neutro']}

        self.client_mongo.update(mongo_db=self.mongo_name, mongo_db_coll=self.coll_hashtag,
                                 criteria=criteria_hashtag, update_fields={"$set": update_fields})

        if (self.count_hashtag % 500) == 0:
            time.sleep(5)
            print("*** Update Hashtag: {0} of {1}".format(self.count_hashtag, len(self.dict_hashtag)))

    def polarity_single_mention(self, item):
        self.count_mention += 1
        criteria_mention = {'id_source': self.dict_mention[item]['id_source'],
                            'id_target': self.dict_mention[item]['id_target']}

        update_fields = {'positive': self.dict_mention[item]['positive'],
                         'negative': self.dict_mention[item]['negative'],
                         'neutro': self.dict_mention[item]['neutro']}
        # print(criteria_mention, ' - ', dict_mention[item])
        self.client_mongo.update(mongo_db=self.mongo_name, mongo_db_coll=self.coll_mention,
                                 criteria=criteria_mention, update_fields={"$set": update_fields})
        # print("DB:", self.mongo_name, " Coll:", self.coll_mention, " CRITERIA:", criteria_mention, " UPDATE:",
        #      {"$set": dict_mention[item]})
        if (self.count_mention % 500) == 0:
            time.sleep(5)
            print("*** Update Mention: {0} of {1}".format(self.count_mention, len(self.dict_mention)))

    def polarity_single_tweet(self, tweet):
        self.count_polarity += 1
        polarity = 0
        classification_models = {'gender': {}, 'knowledge': '', 'age_range': {}}
        if tweet['lang'] == 'es':
            text_analysis_text = self.cls_pol.predict(tweet['text'])
            text_gender = self.best_models.predict_classification('gender', tweet['text'])
            text_knowledge = self.best_models.predict_classification('knowledge', tweet['text'])
            text_age_range = self.best_models.predict_classification('age_range', tweet['text'])
            text_depto = self.best_models.predict_classification('depto', tweet['text'])
            text_plan = self.best_models.predict_classification('plan', tweet['text'])
            classification_models['gender'] = text_gender
            classification_models['knowledge'] = text_knowledge
            classification_models['age_range'] = text_age_range
            classification_models['depto'] = text_depto
            classification_models['plan'] = text_plan
            # print(text_analysis_text)
            # Each by Each
            polarity_tweet = {'id_tweet': tweet['id_str'], 'text': self.cls_pol.proper_encoding(tweet['text']),
                              'polarity': 1 if text_analysis_text['best_relative'][1] > 0 else -1 if text_analysis_text['best_relative'][1] < 0 else 0,
                              'average': text_analysis_text['best_relative'][1], 'label': text_analysis_text['best_relative'][0]}
            self.text_messages.append(polarity_tweet)
            print('#', self.count_polarity, ' de ', self.total_polarity, ' - ', polarity_tweet)
            polarity = text_analysis_text['best_relative'][1]
        else:
            polarity_tweet = {'id_tweet': tweet['id_str'], 'text': self.cls_pol.proper_encoding(tweet['text']),
                              'polarity': 0,
                              'average': 0, 'label': 'Neutro'}
            print('#', self.count_polarity, ' de ', self.total_polarity, ' - ', polarity_tweet)
            self.text_messages.append(polarity_tweet)

        criteria_tweet = {'id_str': tweet['id_str']}
        self.client_mongo.update(mongo_db=self.mongo_name, mongo_db_coll=self.coll_tweets,
                                 criteria=criteria_tweet,
                                 update_fields={"$set": {'polarity_process': polarity_tweet,
                                                         'classification_models': classification_models}})

        for hashtag in tweet['hashtags']:
            key = tweet['user']['id_str'] + hashtag['text']
            if key not in self.dict_hashtag:
                self.dict_hashtag[key] = {"positive": 0, "negative": 0, "neutro": 0,
                                          'id_source': tweet['user']['id'], 'id_hashtag': hashtag['text']}
                if polarity > 0:
                    self.dict_hashtag[key]['positive'] += 1
                elif polarity < 0:
                    self.dict_hashtag[key]['negative'] += 1
                else:
                    self.dict_hashtag[key]['neutro'] += 1
            else:
                if polarity > 0:
                    self.dict_hashtag[key]['positive'] += 1
                elif polarity < 0:
                    self.dict_hashtag[key]['negative'] += 1
                else:
                    self.dict_hashtag[key]['neutro'] += 1

        for user_mention in tweet['user_mentions']:
            key = tweet['user']['id_str'] + user_mention['id_str']
            if key not in self.dict_mention:
                self.dict_mention[key] = {"positive": 0, "negative": 0, "neutro": 0,
                                          'id_source': tweet['user']['id'], 'id_target': user_mention['id']}
                if polarity > 0:
                    self.dict_mention[key]['positive'] += 1
                elif polarity < 0:
                    self.dict_mention[key]['negative'] += 1
                else:
                    self.dict_mention[key]['neutro'] += 1
            else:
                if polarity > 0:
                    self.dict_mention[key]['positive'] += 1
                elif polarity < 0:
                    self.dict_mention[key]['negative'] += 1
                else:
                    self.dict_mention[key]['neutro'] += 1

    def update_polarity_queue(self):
        threads = []
        for i in range(self.num_worker_threads):
            t = threading.Thread(target=self.worker_queue_polarity)
            t.start()
            threads.append(t)

        tweets = self.client_mongo.select(mongo_db=self.mongo_name, mongo_db_coll=self.coll_tweets)
                                          #, criteria={'polarity_process': {'$exists': False}})

        self.total_polarity = len(tweets)
        count_temp = 0
        for item in tweets:
            count_temp += 1
            self.q_polarity.put(item)

        self.q_polarity.join()

        # stop workers
        for i in range(self.num_worker_threads):
            self.q_polarity.put(None)
        for t in threads:
            t.join()

    def update_hashtag_queue(self):
        # Queue Hashtag
        print('Queue Hashtag 1 ')
        threads = []
        for i in range(self.num_worker_threads):
            t = threading.Thread(target=self.worker_queue_hashtag)
            t.start()
            threads.append(t)

        print('Queue Hashtag 2 ')
        for item in self.dict_hashtag:
            #print(item)
            self.q_hashtag.put(item)

        # block until all tasks are done
        self.q_hashtag.join()
        print('Queue Hashtag 3 ')
        # stop workers
        for i in range(self.num_worker_threads):
            self.q_hashtag.put(None)
        for t in threads:
            t.join()

    def update_mention_queue(self):
        # Queue Mention
        print('Queue Mention 1 ')
        threads = []
        for i in range(self.num_worker_threads):
            t = threading.Thread(target=self.worker_queue_mention)
            t.start()
            threads.append(t)
        print('Queue Mention 2 ')
        for item in self.dict_mention:
            self.q_mention.put(item)

        # block until all tasks are done
        self.q_mention.join()
        print('Queue Mention 3 ')
        # stop workers
        for i in range(self.num_worker_threads):
            self.q_mention.put(None)
        for t in threads:
            t.join()

    def update_polarity_file_queue(self):
        # Crear Archivo de Resultado de Chats
        # "output/{0}_mention_pajek_{1}.net"
        date_file = datetime.now().strftime("%Y-%m-%d-%H-%M")

        with open(DIR_OUTPUT + "{0}_polarity_pajek_{1}.csv".format(self.mongo_name, date_file), 'w') as outcsv:
            # configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(['INDEX_CHAT', 'TEXT', 'POLARITY', 'AVERAGE', 'LABEL'])
            for item in self.text_messages:
                writer.writerow(
                    [item['id_tweet'], item['text'].encode('utf-8'), item['polarity'], item['average'], item['label']])
        print('End polarity queue')

    def update_polarity(self):
        # update_polarity tweets  175425
        tweets = self.client_mongo.select(mongo_db=self.mongo_name, mongo_db_coll=self.coll_tweets,
                                          criteria={'polarity_process': {'$exists': False}})
        print('** update_polarity tweets ', len(tweets))
        dict_hashtag = {}
        dict_mention = {}

        text_messages = []

        count = 0
        for tweet in tweets:
            count += 1
            polarity = 0
            if tweet['lang'] == 'es':
                text_analysis_text = self.cls_pol.predict(tweet['text'])
                # Each by Each
                polarity_tweet = {'id_tweet': tweet['id_str'], 'text': self.cls_pol.proper_encoding(tweet['text']),
                              'polarity': 1 if text_analysis_text['best_relative'][1] > 0 else -1 if text_analysis_text['best_relative'][1] < 0 else 0,
                              'average': text_analysis_text['best_relative'][1], 'label': text_analysis_text['best_relative'][0]}
                text_messages.append(polarity_tweet)
                print('#', count, ' de ', len(tweets), ' - ', polarity_tweet)
                polarity = text_analysis_text['best_relative'][1]
            else:
                polarity_tweet = {'id_tweet': tweet['id_str'], 'text': self.cls_pol.proper_encoding(tweet['text']),
                                  'polarity': 0,
                                  'average': 0, 'label': 'Neutro'}
                print('#', count, ' de ', len(tweets), ' - ', polarity_tweet)
                text_messages.append(polarity_tweet)

            criteria_tweet = {'id_str': tweet['id_str']}

            self.client_mongo.update(mongo_db=self.mongo_name, mongo_db_coll=self.coll_tweets,
                                     criteria=criteria_tweet,
                                     update_fields={"$set": {'polarity_process': polarity_tweet}})

            self.client_mongo.update(mongo_db=self.mongo_name, mongo_db_coll=self.coll_tweets,
                                     criteria=criteria_tweet,
                                     update_fields={"$set": {'polarity_value': polarity}})

            for hashtag in tweet['hashtags']:
                key = tweet['user']['id_str'] + hashtag['text']
                if key not in dict_hashtag:
                    dict_hashtag[key] = {"positive": 0, "negative": 0, "neutro": 0,
                                         'id_source': tweet['user']['id'], 'id_hashtag': hashtag['text']}
                    if polarity > 0:
                        dict_hashtag[key]['positive'] += 1
                    elif polarity < 0:
                        dict_hashtag[key]['negative'] += 1
                    else:
                        dict_hashtag[key]['neutro'] += 1
                else:
                    if polarity > 0:
                        dict_hashtag[key]['positive'] += 1
                    elif polarity < 0:
                        dict_hashtag[key]['negative'] += 1
                    else:
                        dict_hashtag[key]['neutro'] += 1

            for user_mention in tweet['user_mentions']:
                key = tweet['user']['id_str'] + user_mention['id_str']
                if key not in dict_mention:
                    dict_mention[key] = {"positive": 0, "negative": 0, "neutro": 0,
                                         'id_source': tweet['user']['id'],  'id_target': user_mention['id']}
                    if polarity > 0:
                        dict_mention[key]['positive'] += 1
                    elif polarity < 0:
                        dict_mention[key]['negative'] += 1
                    else:
                        dict_mention[key]['neutro'] += 1
                else:
                    if polarity > 0:
                        dict_mention[key]['positive'] += 1
                    elif polarity < 0:
                        dict_mention[key]['negative'] += 1
                    else:
                        dict_mention[key]['neutro'] += 1

        count_hashtag = 0
        count_mention = 0

        for item in dict_hashtag:
            count_hashtag += 1
            criteria_hashtag = {'id_source': dict_hashtag[item]['id_source'], 'id_hashtag': dict_hashtag[item]['id_hashtag']}
            update_fields = {'positive': dict_hashtag[item]['positive'], 'negative': dict_hashtag[item]['negative'],
                             'neutro': dict_hashtag[item]['neutro']}

            self.client_mongo.update(mongo_db=self.mongo_name, mongo_db_coll=self.coll_hashtag,
                                     criteria=criteria_hashtag, update_fields={"$set": update_fields})
            #print("DB:", self.mongo_name, " Coll:", self.coll_hashtag, " CRITERIA:", criteria_hashtag, " UPDATE:", {"$set" : update_fields})
            print("*** Update Hashtag: {0} of {1}".format(count_hashtag, len(dict_hashtag)))

        for item in dict_mention:
            count_mention += 1
            criteria_mention = {'id_source': dict_mention[item]['id_source'], 'id_target': dict_mention[item]['id_target']}
            update_fields = {'positive': dict_mention[item]['positive'], 'negative': dict_mention[item]['negative'],
                             'neutro': dict_mention[item]['neutro']}
            #print(criteria_mention, ' - ', dict_mention[item])
            self.client_mongo.update(mongo_db=self.mongo_name, mongo_db_coll=self.coll_mention,
                                     criteria=criteria_mention, update_fields={"$set": update_fields})
            #print("DB:", self.mongo_name, " Coll:", self.coll_mention, " CRITERIA:", criteria_mention, " UPDATE:",
            #      {"$set": dict_mention[item]})
            print("*** Update Mention: {0} of {1}".format(count_mention, len(dict_mention)))

        # Crear Archivo de Resultado de Chats
        "output/{0}_mention_pajek_{1}.net"
        date_file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

        with open(DIR_OUTPUT + "{0}_polarity_pajek_{1}.csv".format(self.mongo_name, date_file),'w') as outcsv:
            # configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(['INDEX_CHAT', 'TEXT', 'POLARITY', 'AVERAGE', 'LABEL'])
            for item in text_messages:
                writer.writerow(
                    [item['id_tweet'], item['text'].encode('utf-8'), item['polarity'], item['average'], item['label']])

    def create_graph_mention(self):
        mentions = self.client_mongo.select(mongo_db=self.mongo_name, mongo_db_coll=self.coll_mention
                                            , criteria={'positive': {'$exists': True}})
        print('** mentions ', len(mentions))
        for mention in mentions:
            #print(mention)
            polarity = '0'
            if mention['positive'] > mention['negative']:
                polarity = '1'
            elif mention['negative'] > mention['positive']:
                polarity = '-1'
            self.graph_mention.add_node(mention['screen_name_source'], level=str(mention['type']), polarity='0')
            self.graph_mention.add_node(mention['screen_name_target'], level=str(mention['type']), polarity=polarity)
            self.graph_mention.add_edge(mention['screen_name_source'], mention['screen_name_target'])
            if mention['screen_name_target'] in self.node_size_mention:
                self.node_size_mention[mention['screen_name_target']] += 1
                self.node_polarity_mention[mention['screen_name_target']] += mention['positive'] - mention['negative']
            else:
                self.node_size_mention[mention['screen_name_target']] = 1
                self.node_polarity_mention[mention['screen_name_target']] = mention['positive'] - mention['negative']
            if not mention['screen_name_source'] in self.node_size_mention:
                self.node_size_mention[mention['screen_name_source']] = 1
                self.node_polarity_mention[mention['screen_name_source']] = 0

        print("** finish create_graph_mention")

    def proper_encoding(self, text):
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return text

    def create_graph_hashtag(self):
        self.graph_hashtag = nx.DiGraph(name="Hashtag")
        hashtags = self.client_mongo.select(mongo_db=self.mongo_name, mongo_db_coll=self.coll_hashtag)
        print('** hashtags ', len(hashtags))
        for hashtag in hashtags:
            print(hashtag)
            hashtag['screen_name_source'] = self.proper_encoding(hashtag['screen_name_source'])
            hashtag['id_hashtag'] = self.proper_encoding(hashtag['id_hashtag'])
            self.graph_hashtag.add_node(hashtag['screen_name_source'], level=str(hashtag['type']))
            self.graph_hashtag.add_node(hashtag['id_hashtag'], level=str(hashtag['type']))
            self.graph_hashtag.add_edge(hashtag['screen_name_source'], hashtag['id_hashtag'])
            if hashtag['id_hashtag'] in self.node_size_hashtag:
                self.node_size_hashtag[hashtag['id_hashtag']] += 1
                self.node_polarity_hashtag[hashtag['id_hashtag']] += hashtag['positive'] - hashtag['negative']
            else:
                self.node_size_hashtag[hashtag['id_hashtag']] = 1
                self.node_polarity_hashtag[hashtag['id_hashtag']] = hashtag['positive'] - hashtag['negative']
            if not hashtag['screen_name_source'] in self.node_size_hashtag:
                self.node_size_hashtag[hashtag['screen_name_source']] = 1
                self.node_polarity_hashtag[hashtag['screen_name_source']] = 0

        print("** finish create_graph_hashtag")

    def create_graph_follower(self):
        followers = self.client_mongo.select(mongo_db=self.mongo_name, mongo_db_coll=self.coll_follower)
        print('** follower ', len(followers))
        for follower in followers:
            self.graph_follower.add_node(follower['screen_name_source'], level=str(follower['type']))
            self.graph_follower.add_node(follower['screen_name_target'], level=str(follower['type']))
            self.graph_follower.add_edge(follower['screen_name_source'], follower['screen_name_target'])
            if follower['screen_name_target'] in self.node_size_follower:
                self.node_size_follower[follower['screen_name_target']] += 1
            else:
                self.node_size_follower[follower['screen_name_target']] = 1
            if not follower['screen_name_source'] in self.node_size_follower:
                self.node_size_follower[follower['screen_name_source']] = 1

        print("** finish create_graph_follower")

    def export_graph_mention(self):
        date_file = datetime.now().strftime("%Y-%m-%d-%H-%M")
        nx.write_adjlist(self.graph_mention, DIR_OUTPUT + "{0}_mention_adjlist_{1}.csv"
                         .format(self.mongo_name, date_file))
        nx.write_pajek(self.graph_mention, DIR_OUTPUT + "{0}_mention_pajek_{1}.net"
                       .format(self.mongo_name, date_file))

        local_node_size = [float(self.node_size_mention[v]) for v in self.graph_mention]

        local_node_polarity = [1 if self.node_polarity_mention[v] > 0 else 2 if self.node_polarity_mention[v] < 0 else 0
                               for v in self.graph_mention]

        with open(DIR_OUTPUT + "{0}_mention_pajek_{1}.vec"
                       .format(self.mongo_name, date_file), "w") as text_file:
            print(f"*Vertices {len(local_node_size)}", file=text_file)
            for item in local_node_size:
                print(f"{item}", file=text_file)

        with open(DIR_OUTPUT + "{0}_mention_pajek_{1}.clu"
                       .format(self.mongo_name, date_file), "w") as text_file:
            print(f"*Vertices {len(local_node_polarity)}", file=text_file)
            for item in local_node_polarity:
                print(f"{item}", file=text_file)

        print("** finish export_graph_mention")

    def export_graph_hashtag(self):
        date_file = datetime.now().strftime("%Y-%m-%d-%H-%M")
        nx.write_adjlist(self.graph_hashtag, DIR_OUTPUT + "{0}_hashtag_adjlist_{1}.csv"
                         .format(self.mongo_name, date_file))
        nx.write_pajek(self.graph_hashtag, DIR_OUTPUT + "{0}_hashtag_pajek_{1}.net"
                       .format(self.mongo_name, date_file))

        local_node_size = [float(self.node_size_hashtag[v]) for v in self.graph_hashtag]

        local_node_polarity = [1 if self.node_polarity_hashtag[v] > 0 else 2
                               if self.node_polarity_hashtag[v] < 0 else 0 for v in self.graph_hashtag]

        with open(DIR_OUTPUT + "{0}_hashtag_pajek_{1}.vec"
                       .format(self.mongo_name, date_file), "w") as text_file:
            print(f"*Vertices {len(local_node_size)}", file=text_file)
            for item in local_node_size:
                print(f"{item}", file=text_file)

        with open(DIR_OUTPUT + "{0}_hashtag_pajek_{1}.clu"
                       .format(self.mongo_name, date_file), "w") as text_file:
            print(f"*Vertices {len(local_node_polarity)}", file=text_file)
            for item in local_node_polarity:
                print(f"{item}", file=text_file)

        print("** finish export_graph_hashtag")

    def export_graph_follower(self):
        date_file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        nx.write_adjlist(self.graph_follower, DIR_OUTPUT + "{0}_follower_adjlist_{1}.csv"
                         .format(self.mongo_name, date_file))
        nx.write_pajek(self.graph_follower, DIR_OUTPUT + "{0}_follower_pajek_{1}.net"
                       .format(self.mongo_name, date_file))

        local_node_size = [float(self.node_size_follower[v]) for v in self.graph_follower]

        with open(DIR_OUTPUT + "{0}_follower_pajek_{1}.vec"
                       .format(self.mongo_name, date_file), "w") as text_file:
            print(f"*Vertices {len(local_node_size)}", file=text_file)
            for item in local_node_size:
                print(f"{item}", file=text_file)

        print("** finish export_graph_follower")


if __name__ == "__main__":
    netx = NetworkSubjectivity("medellin_2020-10-05-10-47")
    netx.num_worker_threads = 8  # Num cores multiprocess

    netx.update_polarity_queue()
    netx.update_hashtag_queue()
    netx.update_mention_queue()
    netx.update_polarity_file_queue()

    netx.create_graph_mention()
    netx.create_graph_hashtag()

    netx.export_graph_mention()
    netx.export_graph_hashtag()


