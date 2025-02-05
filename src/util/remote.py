import re
from pathlib import Path
from typing import Union, Optional
import boto3
import botocore.exceptions as boto_exceptions
from cached_path.schemes import SchemeClient, add_scheme_client
from typing import Optional, Tuple
import io

def add_cached_path_clients():
    add_scheme_client(WekaClient)

class WekaClient(SchemeClient):
    recoverable_errors = SchemeClient.recoverable_errors + (
        boto_exceptions.HTTPClientError,
        boto_exceptions.ConnectionError,
    )

    scheme = "weka"

    def __init__(self, resource: str) -> None:
        SchemeClient.__init__(self, resource)
        self.bucket_name, self.path = WekaClient._split_cloud_path(resource, "weka")
        self.s3 = boto3.client("s3")
        self.object_info = None

    @staticmethod
    def _split_cloud_path(url: str, provider: str) -> Tuple[str, str]:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if not parsed.netloc or not parsed.path:
            raise ValueError(f"bad {provider} path {url}")
        bucket_name = parsed.netloc
        provider_path = parsed.path
        if provider_path.startswith("/"):
            provider_path = provider_path[1:]
        return bucket_name, provider_path

    def _ensure_object_info(self):
        if self.object_info is None:
            try:
                self.object_info = self.s3.head_object(Bucket=self.bucket_name, Key=self.path)
            except boto_exceptions.ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    raise FileNotFoundError(f"weka://{self.bucket_name}/{self.path}") from e
                raise e

    def get_etag(self) -> Optional[str]:
        self._ensure_object_info()
        assert self.object_info is not None
        return self.object_info.get("ETag")

    def get_size(self) -> Optional[int]:
        self._ensure_object_info()
        assert self.object_info is not None
        return self.object_info.get("ContentLength")

    def get_resource(self, temp_file: io.BufferedWriter) -> None:
        self.s3.download_fileobj(Fileobj=temp_file, Bucket=self.bucket_name, Key=self.path)

    def get_bytes_range(self, index: int, length: int) -> bytes:
        response = self.s3.get_object(
            Bucket=self.bucket_name, Key=self.path, Range=f"bytes={index}-{index+length-1}"
        )
        return response["Body"].read()

def is_remote_url(path: Union[str, Path]) -> bool:
    """Return True if the path appears to be a remote URL (proto://...)."""
    return re.match(r"^[a-z0-9]+://", str(path)) is not None


def file_size(path: Union[str, Path]) -> int:
    p = Path(path)
    return p.stat().st_size if p.is_file() else 0


def upload(src: Union[str, Path], dst: str, overwrite: bool = False) -> None:
    from shutil import copy2
    src_path = Path(src)
    dst_path = Path(dst)
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Destination {dst} exists and overwrite is False.")
    copy2(src_path, dst_path)


def get_bytes_range(src: Union[str, Path], start: int, length: int) -> bytes:
    with open(src, "rb") as f:
        f.seek(start)
        return f.read(length)


def find_latest_checkpoint(dir: Union[str, Path]) -> Optional[Union[str, Path]]:
    from os import listdir
    from os.path import join, isdir
    best_step = -1
    best_path = None
    for name in listdir(dir):
        if name.startswith("model"):
            try:
                step = int(''.join(filter(str.isdigit, name)))
                if step > best_step:
                    best_step = step
                    best_path = join(dir, name)
            except ValueError:
                continue
    return best_path
