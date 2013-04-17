import boto
from boto.s3.key import Key
from boto.s3.bucket import Bucket
import util
import re

def upload(bucket_name, local_filename, s3_path):
    connection = boto.connect_s3()
    bucket = Bucket(connection, bucket_name)
    key = Key(bucket)
    key.key = s3_path
    key.set_contents_from_filename(local_filename)
    key.make_public()

def ls(bucket_name, prefix='', pattern=None):
    connection = boto.connect_s3()
    bucket = Bucket(connection, bucket_name)
    keys = [k.key for k in bucket.list(prefix=prefix)]
    if pattern:
        regex = re.compile(pattern)
        keys = [k for k in keys if regex.search(k)]
    return keys

def url(bucket_name, path):
    return 'http://%s.s3.amazonaws.com/%s' % (bucket_name, path)
