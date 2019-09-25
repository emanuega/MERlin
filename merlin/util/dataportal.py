import os
import boto3
import botocore
from google.cloud import storage
from urllib import parse
from abc import abstractmethod, ABC
from typing import List


class DataPortal(ABC):

    """
    A superclass for reading files within a specified directory from a
    data storage service.
    """

    def __init__(self, basePath: str):
        super().__init__()

        self._basePath = basePath

    @abstractmethod
    def is_available(self) -> bool:
        """ Determine if the basePath reperesented by this DataPortal is
        currently accessible.

        Returns: True if the basePath is available, otherwise False.
        """
        pass

    @abstractmethod
    def open_file(self, fileName: str) -> 'FilePortal':
        """ Open a reference to a file within the basePath represented
        by this DataPortal.

        Args:
            fileName: the name of the file to open, relative to basePath
        Returns: A FilePortal referencing the specified file.
        """
        pass

    @staticmethod
    def _filter_file_list(inputList: List[str], extensionList: List[str]
                          ) -> List[str]:
        if not extensionList:
            return inputList
        return [f for f in inputList if any(
            [f.endswith(x) for x in extensionList])]

    @abstractmethod
    def list_files(self, extensionList: List[str]=None) -> List[str]:
        """ List all the files within the base path represented by this
        DataReader.

        Args:
            extensionList: a list of extensions of files to filter for. Only
                files ending in one of the extensions will be returned.
        Returns: a list of the file paths
        """
        pass


class LocalDataPortal(DataPortal):

    """
    A class for accessing data that is stored in a local file system.
    """

    def __init__(self, basePath: str):
        super().__init__(basePath)

    def is_available(self):
        return os.path.exists(self._basePath)

    def open_file(self, fileName):
        return LocalFilePortal(os.path.join(self._basePath, fileName))

    def list_files(self, extensionList=None):
        allFiles = [os.path.join(self._basePath, currentFile)
                    for currentFile in os.listdir(self._basePath)]
        return self._filter_file_list(allFiles, extensionList)


class S3DataPortal(DataPortal):

    """
    A class for accessing data that is stored in a S3 filesystem
    """

    def __init__(self, basePath: str, **kwargs):
        super().__init__(basePath)

        t = parse.urlparse(basePath)
        self._bucketName = t.netloc
        self._prefix = t.path.strip('/')
        self._s3 = boto3.resource('s3', **kwargs)

    def is_available(self):
        objects = list(self._s3.Bucket(self._bucketName).objects.limit(10)
                       .filter(Prefix=self._prefix))
        return len(objects) > 0

    def open_file(self, fileName):
        return S3FilePortal('/'.join([self._basePath, fileName]), s3=self._s3)

    def list_files(self, extensionList=None):
        allFiles = ['s3://%s/%s' % (self._bucketName, f.key)
                    for f in self._s3.Bucket(self._bucketName)
                    .objects.filter(Prefix=self._prefix)]
        return self._filter_file_list(allFiles, extensionList)


class GCloudDataPortal(DataPortal):

    """
    A class for accessing data that is stored in Google Cloud Storage.
    """

    def __init__(self, basePath: str, **kwargs):
        super().__init__(basePath)

        t = parse.urlparse(basePath)
        self._bucketName = t.netloc
        self._prefix = t.path.strip('/')
        self._client = storage.Client(**kwargs)

    def is_available(self):
        blobList = list(self._client.list_blobs(
            self._bucketName, prefix=self._prefix, max_results=10))
        return len(blobList) > 0

    def open_file(self, fileName):
        return GCloudFilePortal('/'.join([self._basePath, fileName]),
                                self._client)

    def list_files(self, extensionList=None):
        allFiles = ['gc://%s/%s' % (self._bucketName, f.name)
                    for f in self._client.list_blobs(
                        self._bucketName, prefix=self._prefix)]
        return self._filter_file_list(allFiles, extensionList)


class FilePortal(ABC):

    """
    A superclass for reading a specified file from a data storage service.
    """

    def __init__(self, fileName: str):
        super().__init__()
        self._fileName = fileName

    def get_file_name(self) -> str:
        """ Get the name of the file accessed by this file portal.

        Returns: The full file name
        """
        return self._fileName

    def get_sibling_with_extension(self, newExtension: str) -> 'FilePortal':
        """ Open the file with the same base name as this file but with the
        specified extension.

        Args:
            newExtension: the new extension
        Returns: A reference to the file with the extension exchanged.
        """
        # TODO
        pass

    @abstractmethod
    def read_as_text(self) -> str:
        """ Read the contents of this file as a string.

        Returns: the file contents as a string
        """
        pass

    @abstractmethod
    def read_file_bytes(self, startByte: int, endByte: int) -> bytes:
        """ Read bytes within the specified range from this file.

        Args:
            startByte: the index of the first byte to read (inclusive)
            endByte: the index at the end of the range of bytes to read
                (exclusive)
        Returns: The bytes between startByte and endByte within this file.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """ Close this file portal."""
        pass


class LocalFilePortal(FilePortal):

    """
    A file portal for accessing a file in a local file system.
    """

    def __init__(self, fileName: str):
        super().__init__(fileName)
        self._fileHandle = open(fileName, 'rb')

    def read_as_text(self):
        self._fileHandle.seek(0)
        return self._fileHandle.read().decode('utf-8')

    def read_file_bytes(self, startByte, endByte):
        self._fileHandle.seek(startByte)
        return self._fileHandle.read(endByte-startByte)

    def close(self) -> None:
        self._fileHandle.close()


class S3FilePortal(FilePortal):

    """
    A file portal for accessing a file from s3.
    """

    def __init__(self, fileName: str, s3):
        super().__init__(fileName)
        t = parse.urlparse(fileName)
        self._bucketName = t.netloc
        self._prefix = t.path.strip('/')
        self._s3 = s3

        self._fileHandle = self._s3.Object(self._bucketName, self._prefix)

    def read_as_text(self):
        return self._fileHandle.get()['Body'].read().decode('utf-8')

    def read_file_bytes(self, startByte, endByte):
        return self._fileHandle.get(
            Range='bytes=%i-%i' % (startByte, endByte-1))['Body'].read()

    def close(self) -> None:
        pass


class GCloudFilePortal(FilePortal):

    """
    A file portal for accessing a file from Google Cloud.
    """

    def __init__(self, fileName: str, client):
        super().__init__(fileName)
        self._client = client
        t = parse.urlparse(fileName)
        self._bucketName = t.netloc
        self._prefix = t.path.strip('/')
        self._bucket = self._client.get_bucket(self._bucketName)

        self._fileHandle = self._bucket.get_blob(self._prefix)

    def read_as_text(self):
        return self._fileHandle.download_as_string().decode('utf-8')

    def read_file_bytes(self, startByte, endByte):
        return self._fileHandle.download_as_string(
            start=startByte, end=endByte-1)

    def close(self) -> None:
        pass
