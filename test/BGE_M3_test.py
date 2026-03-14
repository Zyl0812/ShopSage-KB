from utils.bge_me_embedding_util import get_bge_m3_embedding_model


embedding_model = get_bge_m3_embedding_model()

embedding_result1 = embedding_model.encode_documents(['item_name'])
embedding_result2 = embedding_model.encode_queries(['item_name'])

print(embedding_result1)
print(embedding_result2)
