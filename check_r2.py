import os
import boto3
from dotenv import load_dotenv

load_dotenv()

endpoint = os.environ.get("R2_ENDPOINT")
access_key = os.environ.get("R2_ACCESS_KEY")
secret_key = os.environ.get("R2_SECRET_KEY")
bucket = os.environ.get("R2_BUCKET")

if not all([endpoint, access_key, secret_key, bucket]):
    raise SystemExit("Missing R2_ENDPOINT / R2_ACCESS_KEY / R2_SECRET_KEY / R2_BUCKET")

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name="auto",
)

# Test: write, read, delete a small object
key = "healthcheck/test.txt"
s3.put_object(Bucket=bucket, Key=key, Body=b"ok")
print("Upload OK")

body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
print("Download OK:", body)

s3.delete_object(Bucket=bucket, Key=key)
print("Delete OK")
print("R2 is working.")