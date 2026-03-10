from utils.bge_me_embedding_util import get_bge_m3_embedding_model


embedding_model = get_bge_m3_embedding_model()

embedding_result = embedding_model.encode_documents(['item_name'])

print(embedding_result)