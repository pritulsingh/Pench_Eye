import asyncio
import os
import boto3
from botocore.client import Config
from app.core.config import settings

class ObjectStorageService:
    def __init__(self):
        self.endpoint = settings.MINIO_ENDPOINT
        self.access_key = settings.MINIO_ACCESS_KEY
        self.secret_key = settings.MINIO_SECRET_KEY
        self.bucket = settings.MINIO_BUCKET
        self.secure = settings.MINIO_SECURE
        self.local_path = settings.LOCAL_STORAGE_PATH
        
        self.use_local = False
        try:
            self.s3 = boto3.client(
                's3',
                endpoint_url=f"http{'s' if self.secure else ''}://{self.endpoint}",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(signature_version='s3v4')
            )
            self.s3.list_buckets()
        except Exception:
            self.use_local = True
            os.makedirs(self.local_path, exist_ok=True)

    async def upload_file(self, file_bytes: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        if self.use_local:
            path = os.path.join(self.local_path, key)
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
            
    async def download_file(self, key: str) -> bytes:
        if self.use_local:
            path = os.path.join(self.local_path, key)
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

    async def delete_file(self, key: str):
        if self.use_local:
            path = os.path.join(self.local_path, key)
            if os.path.exists(path):
                await asyncio.to_thread(os.remove, path)
            return
            
        await asyncio.to_thread(
            self.s3.delete_object,
            Bucket=self.bucket,
            Key=key
        )
        
    async def generate_presigned_url(self, key: str, expiry_seconds: int = 3600) -> str:
        if self.use_local:
            return f"/local-storage/{key}"
            
        return await asyncio.to_thread(
            self.s3.generate_presigned_url,
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=expiry_seconds
        )

    async def copy_file(self, src_key: str, dst_key: str):
        if self.use_local:
            src_path = os.path.join(self.local_path, src_key)
            dst_path = os.path.join(self.local_path, dst_key)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            await asyncio.to_thread(self._copy_local, src_path, dst_path)
            return
            
        copy_source = {'Bucket': self.bucket, 'Key': src_key}
        await asyncio.to_thread(
            self.s3.copy_object,
            CopySource=copy_source,
            Bucket=self.bucket,
            Key=dst_key
        )
        
    def _copy_local(self, src: str, dst: str):
        import shutil
        shutil.copy2(src, dst)
        
    async def get_file_size(self, key: str) -> int:
        if self.use_local:
            path = os.path.join(self.local_path, key)
            if os.path.exists(path):
                return os.path.getsize(path)
            return 0
            
        response = await asyncio.to_thread(
            self.s3.head_object,
            Bucket=self.bucket,
            Key=key
        )
        return response['ContentLength']
