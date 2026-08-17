import asyncio
import os
import shutil
from app.core.config import settings

try:
    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError

    _BOTO3_AVAILABLE = True
except ImportError:  # object storage optional — local filesystem is the fallback
    boto3 = None  # type: ignore
    Config = None  # type: ignore

    class ClientError(Exception):  # type: ignore
        pass

    _BOTO3_AVAILABLE = False

class ObjectStorageService:
    def __init__(self):
        self.endpoint = settings.MINIO_ENDPOINT
        self.access_key = settings.MINIO_ACCESS_KEY
        self.secret_key = settings.MINIO_SECRET_KEY
        self.bucket = settings.MINIO_BUCKET
        self.secure = settings.MINIO_SECURE
        self.local_path = settings.LOCAL_STORAGE_PATH
        
        self.use_local = True
        self.s3 = None
        if _BOTO3_AVAILABLE:
            try:
                self.s3 = boto3.client(
                    's3',
                    endpoint_url=f"http{'s' if self.secure else ''}://{self.endpoint}",
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=Config(signature_version='s3v4', connect_timeout=2, retries={'max_attempts': 1})
                )
                self.s3.list_buckets()
                self.use_local = False
            except Exception:
                self.s3 = None
                self.use_local = True

        if self.use_local:
            os.makedirs(self.local_path, exist_ok=True)

    def _safe_local_path(self, key: str) -> str:
        """Resolve a storage key inside the local root; reject traversal."""
        root = os.path.abspath(self.local_path)
        path = os.path.abspath(os.path.join(root, key))
        if not path.startswith(root + os.sep) and path != root:
            raise ValueError("Invalid storage key")
        return path

    async def upload_image(self, file_bytes: bytes, key: str, content_type: str = "image/jpeg") -> str:
        if self.use_local:
            path = self._safe_local_path(key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            await asyncio.to_thread(self._write_file, path, file_bytes)
            return key
        
        await asyncio.to_thread(
            self.s3.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=file_bytes,
            ContentType=content_type
        )
        return key

    def _write_file(self, path: str, data: bytes):
        with open(path, 'wb') as f:
            f.write(data)
            
    async def download_image(self, key: str) -> bytes:
        if self.use_local:
            path = self._safe_local_path(key)
            return await asyncio.to_thread(self._read_file, path)
            
        response = await asyncio.to_thread(
            self.s3.get_object,
            Bucket=self.bucket,
            Key=key
        )
        return await asyncio.to_thread(response['Body'].read)
        
    def _read_file(self, path: str) -> bytes:
        with open(path, 'rb') as f:
            return f.read()

    async def delete_object(self, key: str):
        if self.use_local:
            path = self._safe_local_path(key)
            if os.path.exists(path):
                await asyncio.to_thread(os.remove, path)
            return
            
        await asyncio.to_thread(
            self.s3.delete_object,
            Bucket=self.bucket,
            Key=key
        )
        
    async def generate_presigned_url(self, key: str, expiry: int = 3600) -> str:
        if self.use_local:
            return f"/local-storage/{key}"
            
        return await asyncio.to_thread(
            self.s3.generate_presigned_url,
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=expiry
        )

    async def copy_object(self, src_key: str, dst_key: str):
        if self.use_local:
            src_path = self._safe_local_path(src_key)
            dst_path = self._safe_local_path(dst_key)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            await asyncio.to_thread(shutil.copy2, src_path, dst_path)
            return
            
        copy_source = {'Bucket': self.bucket, 'Key': src_key}
        await asyncio.to_thread(
            self.s3.copy_object,
            CopySource=copy_source,
            Bucket=self.bucket,
            Key=dst_key
        )
        
    async def object_exists(self, key: str) -> bool:
        if self.use_local:
            path = self._safe_local_path(key)
            return os.path.exists(path)
            
        try:
            await asyncio.to_thread(
                self.s3.head_object,
                Bucket=self.bucket,
                Key=key
            )
            return True
        except ClientError:
            return False

storage_service = ObjectStorageService()
