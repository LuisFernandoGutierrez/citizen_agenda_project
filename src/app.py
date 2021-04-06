# from base import Config, Common
# from models import Model
import sys
import os

from src.logic.linguistics import LinguisticsTwitter
from src.logic.network import NetworkSubjectivity
from src.logic.timeline import TwitterSearch

ROOT_PATH = os.path.dirname(
    '/'.join(os.path.abspath(__file__).split('/')[:-1]))
CONFIG_FOLDER = ROOT_PATH + '/config'
sys.path.insert(1, ROOT_PATH)


class Application:
    account_service = None

    def __init__(self, name_database):
        self.name_database = name_database
        self.serviceTwitter = TwitterSearch(name_database)
        self.networkSubjectivity = None
        self.linguistic = None

    def run(self, twitter=True, network=True, linguistic=True):
        # Twitter
        if twitter:
            self.serviceTwitter.main_tweets()
        # Network
        if network:
            self.networkSubjectivity = NetworkSubjectivity(self.name_database)
            self.networkSubjectivity.num_worker_threads = 8  # Num cores multiprocess
            self.networkSubjectivity.update_polarity_queue()
            self.networkSubjectivity.update_hashtag_queue()
            self.networkSubjectivity.update_mention_queue()
            self.networkSubjectivity.update_polarity_file_queue()
            self.networkSubjectivity.create_graph_mention()
            self.networkSubjectivity.create_graph_hashtag()
            self.networkSubjectivity.export_graph_mention()
            self.networkSubjectivity.export_graph_hashtag()
        # Linguistic
        if linguistic:
            self.linguistic = LinguisticsTwitter(self.name_database)
            self.linguistic.get_all()


if __name__ == '__main__':
    # data_base = "medellin_2020-10-08-19-29"
    # app = Application(data_base)
    # app.run(twitter=True, network=True, linguistic=True)
    data_base = "medellin_2020_12"
    app = Application(data_base)
    # app.run(twitter=True)
    app.run(twitter=False, network=True, linguistic=False)
