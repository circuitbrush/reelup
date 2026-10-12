"""Cloudflare R2 helper (S3-compatible).

Environment variables (.env locally, Render -> Environment when deployed):
    R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY      -> token that can create buckets + write (output side)
  optional, recommended:
    R2_BASE_ACCESS_KEY, R2_BASE_SECRET_KEY         -> READ-ONLY token for the base-videos bucket
(The older names R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY still work as a fallback.)

Safety: there is deliberately NO delete function in this module, and every write call refuses to touch
a bucket that was registered with protect() (the base-videos bucket).
"""
import os
import functools
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

_PROTECTED = set()


def _env(*names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


def _endpoint():
    raw = _env("R2_ENDPOINT")
    if raw:
        u = urlparse(raw)  # drops any accidental /bucket path
        return f"{u.scheme}://{u.netloc}" if u.scheme and u.netloc else raw
    acc = _env("R2_ACCOUNT_ID")
    return f"https://{acc}.r2.cloudflarestorage.com" if acc else ""


def _main_creds():
    return _env("R2_ACCESS_KEY", "R2_ACCESS_KEY_ID"), _env("R2_SECRET_KEY", "R2_SECRET_ACCESS_KEY")


def _base_creds():
    return _env("R2_BASE_ACCESS_KEY", "R2_BASE_ACCESS_KEY_ID"), _env("R2_BASE_SECRET_KEY", "R2_BASE_SECRET_ACCESS_KEY")


def protect(bucket):
    """Mark a bucket read-only for this process (the base videos bucket)."""
    if bucket:
        _PROTECTED.add(bucket)


def enabled():
    ak, sk = _main_creds()
    return bool(_endpoint() and ak and sk)


@functools.lru_cache(maxsize=2)
def _make(use_base_creds):
    ak, sk = _base_creds() if use_base_creds else _main_creds()
    return boto3.client(
        "s3",
        endpoint_url=_endpoint(),
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}, max_pool_connections=32),
    )


def _c(bucket):
    """Protected (base) bucket -> read-only credentials if provided, otherwise the main ones."""
    ak, sk = _base_creds()
    return _make(bucket in _PROTECTED and bool(ak and sk))


def _guard(bucket):
    if bucket in _PROTECTED:
        raise PermissionError(f"Bucket '{bucket}' is protected (read-only). Write blocked.")


# ---------------- reads (allowed on any bucket)
def list_objects(bucket, prefix=""):
    out, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = _c(bucket).list_objects_v2(**kw)
        out += [{"key": o["Key"], "size": o["Size"]} for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def download_file(bucket, key, local_path):
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    _c(bucket).download_file(bucket, key, local_path)
    return local_path


def get_bytes(bucket, key):
    try:
        return _c(bucket).get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def presigned_url(bucket, key, expires=86400):
    return _c(bucket).generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)


# ---------------- writes (blocked on protected buckets)
def bucket_exists(bucket):
    try:
        _c(bucket).head_bucket(Bucket=bucket)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchBucket", "NotFound"):
            return False
        raise


def ensure_bucket(bucket):
    """Create the bucket if it does not exist. Returns 'exists' or 'created'."""
    _guard(bucket)
    if bucket_exists(bucket):
        return "exists"
    try:
        _c(bucket).create_bucket(Bucket=bucket)
        return "created"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            return "exists"
        if code in ("AccessDenied", "403"):
            raise RuntimeError(
                "R2 token is not allowed to create buckets. Use an 'Admin Read & Write' API token for "
                "R2_ACCESS_KEY / R2_SECRET_KEY, or create the bucket manually in the dashboard.") from e
        raise


def upload_file(bucket, local_path, key, content_type=None):
    _guard(bucket)
    extra = {"ContentType": content_type} if content_type else None
    _c(bucket).upload_file(local_path, bucket, key, ExtraArgs=extra)
    return key


def put_bytes(bucket, key, data, content_type="application/octet-stream"):
    _guard(bucket)
    _c(bucket).put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return key