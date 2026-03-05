from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://192.168.10.130:19530"
)

collections = client.list_collections()
print(collections)
