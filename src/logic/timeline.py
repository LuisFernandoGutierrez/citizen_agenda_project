import configparser
import json
import os
import sys
import time
import csv
from twitter import Api
import twitter
import datetime
import threading
from root import DIR_CONFIG, DIR_INPUT
from src.util.data_acces import DataBaseAccessMongo
from datetime import datetime


class TwitterSearch:
    def __init__(self, mongo_name=None):
        self.date_file = datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.config = configparser.ConfigParser()
        self.config.read(DIR_CONFIG + 'config.ini')
        self.CONSUMER_KEY = self.config['TWITTER']['CONSUMER_KEY']
        self.CONSUMER_SECRET = self.config['TWITTER']['CONSUMER_SECRET']
        self.ACCESS_TOKEN = self.config['TWITTER']['ACCESS_TOKEN']
        self.ACCESS_TOKEN_SECRET = self.config['TWITTER']['ACCESS_TOKEN_SECRET']
        self.USERS = []
        self.LANGUAGES = ['es', 'en']
        self.file_query = ""
        self.file_users = self.config['TIMELINE']['FILE_USERS']
        self.mongo_name = self.config['MONGO']['DB_MONGO'] + '_' + self.date_file if mongo_name is None else mongo_name
        self.coll_profile = self.config['MONGO']['COLL_PROFILE']
        self.coll_follower = self.config['MONGO']['COLL_FOLLOWER']
        self.coll_hashtag = self.config['MONGO']['COLL_HASHTAG']
        self.coll_mention = self.config['MONGO']['COLL_MENTION']
        self.coll_tweets = self.config['MONGO']['COLL_TWEETS']

        self.load_users()
        self.client_mongo = DataBaseAccessMongo()
        self.api = Api(self.CONSUMER_KEY, self.CONSUMER_SECRET, self.ACCESS_TOKEN,
                       self.ACCESS_TOKEN_SECRET, sleep_on_rate_limit=True)

    def load_users(self):
        with open(DIR_INPUT + self.file_users, newline='', encoding='UTF-8') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in data_reader:
                #print(row)
                self.USERS.append(row[0])
            print('** Users Load Count: ', len(self.USERS), ' - Data : ', self.mongo_name)
        # with open(DIR_INPUT + self.file_users, newline='', encoding='UTF-8') as csv_file:
        #     data_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        #     for row in data_reader:
        #         self.USERS.append(row[0].strip())
        #     print('** Users Load Count: ', len(self.USERS), ' - Data : ', self.mongo_name)

    def get_hashtag_tweet(self, tweet):
        list_hastags = []
        for hashtag in tweet.hashtags:
            relation = {'id_source': tweet.user.id, 'name_source': tweet.user.name,
                        'screen_name_source': tweet.user.screen_name, 'id_hashtag': hashtag.text, 'type': 2}
            list_hastags.append(relation)
        return list_hastags

    def get_mention_tweet(self, tweet):
        list_user_mention = []
        for user_mention in tweet.user_mentions:
            relation = {'id_source': tweet.user.id, 'name_source': tweet.user.name,
                        'screen_name_source': tweet.user.screen_name, 'id_target': user_mention.id,
                        'name_target': user_mention.name, 'screen_name_target': user_mention.screen_name, 'type': 3}
            list_user_mention.append(relation)
        return list_user_mention

    def get_all_timeline(self, user_name):
        max_results = 3200
        max_pages = 16
        results = []
        results_hashtag = []
        results_user_mention = []

        rate_limit_user_timeline = self.api.rate_limit.get_limit('/statuses/user_timeline.json')
        print('/statuses/user_timeline.json - ', datetime.fromtimestamp(
            rate_limit_user_timeline.reset).strftime('%Y-%m-%d %H:%M:%S'), ' - ', rate_limit_user_timeline)
        tweets = self.api.GetUserTimeline(screen_name=user_name, count=200)

        if tweets is None:  # 401 (Not Authorized) - Need to bail out on loop entry
            tweets = []

        for tweet in tweets:
            results_hashtag += self.get_hashtag_tweet(tweet)
            results_user_mention += self.get_mention_tweet(tweet)
            results.append(tweet.AsDict())

        print('\t ** INFO: ', sys.stderr, 'Fetched %i tweets' % len(tweets))
        page_num = 1

        if max_results == 200:
            page_num = max_pages  # Prevent loop entry

        while page_num < max_pages and len(tweets) > 0 and len(results) < max_results:
            # Necessary for traversing the timeline in Twitter's v1.1 API:
            # get the next query's max-id parameter to pass in.
            # See https://dev.twitter.com/docs/working-with-timelines.

            for tweet in tweets:
                results_hashtag += self.get_hashtag_tweet(tweet)
                results_user_mention += self.get_mention_tweet(tweet)
                results.append(tweet.AsDict())

            max_id = min([tweet.id for tweet in tweets]) - 1

            rate_limit_user_timeline = self.api.rate_limit.get_limit('/statuses/user_timeline.json')
            print('/statuses/user_timeline.json - ', datetime.fromtimestamp(
                rate_limit_user_timeline.reset).strftime('%Y-%m-%d %H:%M:%S'), ' - ', rate_limit_user_timeline)
            tweets = self.api.GetUserTimeline(screen_name=user_name, count=200, max_id=max_id)

            print('\t ** INFO: ', sys.stderr, 'Fetched %i tweets' % (len(tweets),))
            page_num += 1

        print('\t ** INFO: ', sys.stderr, 'Done fetching tweets')
        return results, results_hashtag, results_user_mention

    def main_users(self):
        for user_name in self.USERS:
            try:
                print('++ start profile...')
                profile = self.api.GetUser(screen_name=user_name)
                print('\t', type(profile), '-', profile.name)
                profile_dict = profile.AsDict()
                profile_dict['role_type'] = 'source'
                profile_dict['role_id'] = ''
                self.client_mongo.save(profile_dict, self.mongo_name, self.coll_profile)

                print('++ start followers...')
                rate_limit_followers = self.api.rate_limit.get_limit('/followers/ids.json')
                print('/followers/ids.json - ', datetime.fromtimestamp(
                    rate_limit_followers.reset).strftime('%Y-%m-%d %H:%M:%S'), ' - ', rate_limit_followers)

                followers = self.api.GetFollowerIDs(screen_name=user_name, total_count=500)
                follower_data = []
                profile_follower_data = []
                print('\t', 'followers: ', user_name, '-', len(followers))

                for follower in followers:
                    profile_follower = self.api.GetUser(user_id=follower)
                    print('\t', type(profile_follower), '-', profile_follower.name)
                    profile_dict_follower = profile_follower.AsDict()
                    profile_dict_follower['role_type'] = 'follower'
                    profile_dict_follower['role_id'] = profile.id
                    # self.client_mongo.save(profile_dict, self.mongo_name, self.coll_profile)
                    profile_follower_data.append(profile_dict_follower)
                    relation = {'id_source': profile.id, 'name_source': profile.name,
                                'screen_name_source': profile.screen_name, 'id_target': follower,
                                'name_target': profile_follower.name,
                                'screen_name_target': profile_follower.screen_name, 'type': 1}
                    follower_data.append(relation)
                if len(follower_data) > 0:
                    print('*** follower_data :', len(follower_data))
                    self.client_mongo.save_to_mongo_coll(follower_data, self.mongo_name, self.coll_follower)
                if len(profile_follower_data) > 0:
                    print('*** profile_follower_data :', len(profile_follower_data))
                    self.client_mongo.save_to_mongo_coll(profile_follower_data, self.mongo_name, self.coll_profile)

            except twitter.error.TwitterError as e:
                print("* twitter.error.TwitterError :", e, ' - ', str(e), ' - ', e.message)
                self.api = Api(self.CONSUMER_KEY, self.CONSUMER_SECRET, self.ACCESS_TOKEN,
                               self.ACCESS_TOKEN_SECRET, sleep_on_rate_limit=True)
                time.sleep(50)

    def main_tweets(self):
        for user_name in self.USERS:
            try:
                print('++ start profile... ', user_name)
                profile = self.api.GetUser(screen_name=user_name)
                print('\t', type(profile), '-', profile.name)
                profile_dict = profile.AsDict()
                profile_dict['role_type'] = 'source'
                profile_dict['role_id'] = ''

                print('++ start tweets...', user_name)
                tweets, hastags, user_mentions = self.get_all_timeline(user_name)
                print('\t', 'tweets: ', user_name, '-', len(tweets))
                print('\t', 'hastags: ', user_name, '-', len(hastags))
                print('\t', 'user_mentions: ', user_name, '-', len(user_mentions))
                if len(tweets) > 0:
                    self.client_mongo.save_to_mongo_coll(tweets, self.mongo_name, self.coll_tweets)
                if len(hastags) > 0:
                    self.client_mongo.save_to_mongo_coll(hastags, self.mongo_name, self.coll_hashtag)
                if len(user_mentions) > 0:
                    self.client_mongo.save_to_mongo_coll(user_mentions, self.mongo_name, self.coll_mention)

            except twitter.error.TwitterError as e:
                print("* twitter.error.TwitterError :", e, ' - ', str(e), ' - ', e.message)
                self.api = Api(self.CONSUMER_KEY, self.CONSUMER_SECRET, self.ACCESS_TOKEN,
                               self.ACCESS_TOKEN_SECRET, sleep_on_rate_limit=True)
                time.sleep(50)

    def main_test(self):
        print(self.api.VerifyCredentials())
        for user_name in self.USERS:
            print(user_name)
            users = self.api.GetFriends()
            print([u.name for u in users])
            statuses = self.api.GetUserTimeline(screen_name=user_name)
            print([s.text for s in statuses])
            profile = self.api.GetUser(screen_name=user_name)
            print(profile)


if __name__ == '__main__':
    serviceTwitter = TwitterSearch()
    threads = []
    t_tweet = threading.Thread(target=serviceTwitter.main_tweets)
    threads.append(t_tweet)
    t_tweet.start()



