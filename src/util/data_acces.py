import configparser
import pymongo
from redis import Redis
from pymongo.errors import BulkWriteError
import json
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class DataBaseAccessMongo:
    def __init__(self):
        self.host = None
        self.port = None
        self.db = None
        self.collection = None

    def drop_database(self, name_database, **mongo_conn_kw):
        # Connects to the MongoDB server running on
        # localhost:27017 by default
        client = pymongo.MongoClient(host=self.host, port=self.port, **mongo_conn_kw)
        # Get a reference to a particular database
        client.drop_database(name_database)

    def save(self, data, mongo_db=None, mongo_db_coll=None, **mongo_conn_kw):
        # Connects to the MongoDB server running on
        # localhost:27017 by default
        mongo_db = self.db if mongo_db is None else mongo_db
        mongo_db_coll = self.collection if mongo_db_coll is None else mongo_db_coll
        client = pymongo.MongoClient(host=self.host, port=self.port, **mongo_conn_kw)
        # Get a reference to a particular database
        db = client[mongo_db]
        # Reference a particular collection in the database
        coll = db[mongo_db_coll]
        # Perform a bulk insert and  return the IDs
        return coll.insert(data)

    def update(self, mongo_db=None, mongo_db_coll=None, criteria=None, update_fields=None, **mongo_conn_kw):
        # Connects to the MongoDB server running on
        # localhost:27017 by default
        mongo_db = self.db if mongo_db is None else mongo_db
        mongo_db_coll = self.collection if mongo_db_coll is None else mongo_db_coll
        client = pymongo.MongoClient(host=self.host, port=self.port, **mongo_conn_kw)
        # Get a reference to a particular database
        db = client[mongo_db]
        # Reference a particular collection in the database
        coll = db[mongo_db_coll]
        # Perform a bulk insert and  return the IDs
        return coll.update_many(criteria, update_fields, upsert=False, bypass_document_validation=True)

    def save_to_mongo_coll(self, data_array, mongo_db=None, mongo_db_coll=None, **mongo_conn_kw):
        try:
            # Connects to the MongoDB server running on
            # localhost:27017 by default
            mongo_db = self.db if mongo_db is None else mongo_db
            mongo_db_coll = self.collection if mongo_db_coll is None else mongo_db_coll
            client = pymongo.MongoClient(host=self.host, port=self.port, **mongo_conn_kw)
            # Get a reference to a particular database
            db = client[mongo_db]
            # Reference a particular collection in the database
            coll = db[mongo_db_coll]
            # Perform a bulk insert and  return the IDs
            return coll.insert_many(data_array, bypass_document_validation=True)

        except BulkWriteError as e:
            print("\t ERROR save_to_mongo_coll: ", e.details['writeErrors'])

    def select(self, mongo_db=None, mongo_db_coll=None, return_cursor=False, criteria=None, projection=None, **mongo_conn_kw):
        # Optionally, use criteria and projection to limit the data that is
        # returned as documented in
        # http://docs.mongodb.org/manual/reference/method/db.collection.find/
        # Consider leveraging MongoDB's aggregations framework for more
        # sophisticated queries.

        mongo_db = self.db if mongo_db is None else mongo_db
        mongo_db_coll = self.collection if mongo_db_coll is None else mongo_db_coll

        client = pymongo.MongoClient(host=self.host, port=self.port, **mongo_conn_kw)
        db = client[mongo_db]
        coll = db[mongo_db_coll]
        if criteria is None:
            criteria = {}

        if projection is None:
            cursor = coll.find(criteria)
        else:
            cursor = coll.find(criteria, projection)

        # Returning a cursor is recommended for large amounts of data
        if return_cursor:
            return cursor
        else:
            return [item for item in cursor]
