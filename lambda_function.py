import urllib.parse
import logging
import os
import io

from PIL import Image
import boto3
import torch
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification, ViTImageProcessor


# db_host = os.environ['DB_HOST']
# db_name = os.environ['DB_NAME']
# db_user = os.environ['DB_USER']
# db_password = os.environ['DB_PASSWORD']

# conn = pymysql.connect(
#             host=db_host,
#             user=db_user,
#             passwd=db_password,
#             db=db_name,
#             connect_timeout=5
#         )


os.environ["TRANSFORMERS_CACHE"] = "/cache_dir/"


s3 = boto3.client("s3")

model = AutoModelForImageClassification.from_pretrained(
    "./model", local_files_only=True, cache_dir="/cache_dir/"
)
processor = ViTImageProcessor.from_pretrained(
    "./model", local_files_only=True, cache_dir="/cache_dir/"
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image)


def handler(event, context):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("nsfw_image_detection lambda_handler started")
    # 이벤트에서 S3 버킷과 오브젝트 키 정보 추출
    bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
    key = urllib.parse.unquote_plus(
        event["Records"][0]["s3"]["object"]["key"], encoding="utf-8"
    )
    logger.info("S3 Object Key: %s", key)

    try:
        # S3 오브젝트 메타데이터 가져오기
        image_obj = s3.get_object(Bucket=bucket_name, Key=key)
        # 이미지 데이터를 바이트 형식으로 읽습니다.
        image_data = image_obj['Body'].read()
        img = Image.open(io.BytesIO(image_data))
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
        logger.info("Logits: %s", logits)
        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        logger.info("Image classification result: %s", label)

        # with conn.cursor() as cur:
        #     # 메타데이터를 RDS에 저장
        #     logger.info("Saving metadata to RDS")
        #     cur.execute("SELECT post_id FROM posts WHERE post_id = %s", (key,))
        #     result = cur.fetchall()
        #     for row in result:
        #         post_id = row[0]
        #     cur.execute("INSERT INTO tags (post_id, content) VALUES (%s, %s, %s)", (post_id, label))
        #     conn.commit()

        return {"statusCode": 200, "body": "Metadata stored successfully in MySQL"}
    except Exception as e:
        logger.error("Error: %s", e)
        raise e
