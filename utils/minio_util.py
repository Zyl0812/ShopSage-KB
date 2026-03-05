import logging
import os

from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

load_dotenv()

def get_minio_client():
    """
    获取minio客户端
    """  
    try:
        # 1. 创建客户端
        client = Minio(
            os.getenv("MINIO_ENDPOINT", ''),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=False,
        )

        # 2. 判断桶是否存在
        bucket_name = os.getenv("MINIO_BUCKET_NAME", '')
        bucket_exists = client.bucket_exists(bucket_name)

        if not bucket_exists:
            client.make_bucket(bucket_name)
            logging.info(f"桶{bucket_name}不存在，已自动创建")
        else:
            logging.info(f"桶{bucket_name}已存在")

        return client

    except S3Error as e:
        logging.error(f"获取minio客户端失败：{e}")
        return None


if __name__ == "__main__":
    client = get_minio_client()