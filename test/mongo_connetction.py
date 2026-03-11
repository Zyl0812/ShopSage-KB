from pymongo import MongoClient

client = MongoClient("mongodb://admin:123456@192.168.10.130:27017")

db = client['mydb']

collection = db['students']

for doc in collection.find():
    print(doc)

