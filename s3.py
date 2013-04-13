from boto.s3.key import Key
from boto.s3.bucket import Bucket
import util

def upload(bucket_name, local_filename, s3_path):
    connection = boto.connect_s3()
    bucket = Bucket(connection, bucket_name)
    key = Key(bucket)
    key.key = s3_path
    key.set_contents_from_filename(local_filename)
    key.make_public()
