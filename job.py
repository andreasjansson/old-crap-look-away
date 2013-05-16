import pika
import json
import ConfigParser
import os
import sys
import datetime
import socket
import cPickle
import psycopg2
import psycopg2.extras

_config = None
def config(section):
    global _config
    if _config is None:
        _config = parse_config()
    return dict(_config.items(section))

def parse_config():
    config = ConfigParser.RawConfigParser()
    config.read(os.path.expanduser('~/.job'))
    return config

class Job(object):

    def __init__(self, name):
        self.name = name

        db_type = config('job')['db']
        if db_type == 'postgresql':
            self.db = PostgreSQL(self.name)
        else:
            raise Exception('Unknown db: %s' % db_type)

        queue_type = config('job')['queue']
        if queue_type == 'rabbitmq':
            self.queue = RabbitMQ(self.name)
        else:
            raise Exception('Unknown queue: %s' % queue_type)

        self.hostname = socket.gethostname()

    def put_data(self, data):
        self.queue.put(self.name, json.dumps(data))

    def run_worker(self, do_work):
        while True:
            item = self.queue.get()

            if item is None:
                break

            do_work(self, item.data)
            item.ack()

    def log(self, message):
        print message

    def log_exception(self, exception):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        self.log('EXCEPTION: %s in %s (line %d)' % (
            exc_type, fname, exc_tb.tb_lineno))

    def store(self, key, value):
        self.db.store(key, value)

    def store_instance(self, key, value):
        self.db.store(key, value, '%d-%d' % (INDEX(), COUNT()))

    def clear(self):
        try:
            self.queue.clear()
        except Exception, e:
            sys.stderr.write('Failed to clear queue: %s\n' % str(e))
        try:
            self.db.clear()
        except Exception, e:
            sys.stderr.write('Failed to clear database: %s\n' % str(e))

    def get_data(self, key=None, row_limit=None):
        return self.db.fetch(key, row_limit=row_limit)

    def reduce_instances(self, key, count, reducer):
        instance_keys = ['%d-%d' % (i, count) for i in range(count)]
        data = self.db.fetch(key, instance_keys)
        return reduce(reducer, data)

    def get_queue_length(self):
        return self.queue.length()


def INDEX():
    host_count = int(os.environ.get('HOST_COUNT', 1))
    host_index = int(os.environ.get('HOST_INDEX', 0))
    return int(os.environ.get('INSTANCE_INDEX', 0)) + host_index * int(os.environ.get('INSTANCE_COUNT', 1))

def COUNT():
    return int(os.environ.get('HOST_COUNT', 1)) * int(os.environ.get('INSTANCE_COUNT', 1))

def cross_partition(data):
    index = INDEX()
    count = COUNT()
    list1 = []
    list2 = []
    for i, d in enumerate(data):
        if (i - index) % count == 0:
            list2.append(d)
        else:
            list1.append(d)
    return list1, list2

class RabbitMQ:

    def __init__(self, name):
        self.name = name
        self.channel = None

    def get(self):
        method, header_frame, body = self._channel().basic_get(self.name)

        if method is None: # empty queue
            return None

        return RabbitItem(method, header_frame, body)

    def put(self, data):
        self._channel().basic_publish(
            exchange='',
            routing_key=self.name,
            body=data
        )

    def clear(self):
        self._channel().queue_delete(queue=self.name)

    def length(self):
        return self._channel().queue_declare(
            self.name, passive=True).method.message_count

    def _channel(self):
        if self.channel is not None:
            return self.channel
        self.conn = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=config('rabbitmq')['host'],
                port=int(config('rabbitmq')['port']),
                credentials=pika.PlainCredentials(
                    config('rabbitmq')['user'],
                    config('rabbitmq')['pass'],
                )
            ))
        self.channel = self.conn.channel()
        self.channel.queue_declare(queue=self.name)

        return self.channel
        
class RabbitItem:

    def __init__(method, header_frame, body, channel):
        self.data = json.loads(body)
        self.method = method
        self.header_frame = header_frame
        self.channel = channel

    def ack(self):
        self.channel.basic_ack(self.method.delivery_tag)

class PostgreSQL:

    def __init__(self, name):
        self.name = name
        self.conn = None
        self.cursor = None

    def store(self, key, value, instance_key=None):
        if instance_key is None:
            instance_key = ''

        # pg upsert
        self._cursor().execute(
            'update ' + self.name +
            ' set value = %s where key = %s and instance_key = %s',
            (cPickle.dumps(value), key, instance_key))

        self._cursor().execute(
            'insert into ' + self.name +
            ' (key, value, instance_key) select %s, %s, %s' +
            ' where not exists (select 1 from ' + self.name +
            ' where key = %s and instance_key = %s)',
            (key, cPickle.dumps(value), instance_key, key, instance_key))
        self._conn().commit()
        
    def clear(self):
        self._cursor().execute('drop table %s' % self.name)

    def fetch(self, key=None, instance_keys=None, row_limit=None):
        sql = 'select key, value, instance_key from %s' % self.name
        where = []
        params = []
        if key is not None:
            where.append('key = %s')
            params.append(key)
        if instance_keys:
            where.append('instance_key in %s')
            params.append(tuple(instance_keys))
        else:
            where.append("instance_key = ''")

        if where:
            sql += ' where ' + ' and '.join(where)

        self._cursor().execute(sql, params)

        res = self._cursor().fetchall()

        if not res:
            return []

        data = {}
        for item in res:
            k = item['key']
            v = cPickle.loads(item['value'])
            if key in data:
                data[k].append(v)
            else:
                data[k] = [v]

        if key is None:
            return data
        return data[key]

    def _conn(self):
        if self.conn:
            return self.conn

        self.conn = psycopg2.connect(host=config('postgresql')['host'],
                                user=config('postgresql')['user'],
                                password=config('postgresql')['pass'],
                                database=config('postgresql')['database'])

        self.cursor = self.conn.cursor()

        self.cursor.execute('select 1 from information_schema.tables where table_name = \'%s\'' % self.name)

        row = self.cursor.fetchone()
        if row != (1,):
            self.cursor.execute('create table %s ( id serial primary key, key varchar(50) not null, value text not null, instance_key varchar(200) )' % self.name)
            try:
                self.conn.commit()
            except psycopg2.ProgrammingError:
                self.conn.rollback()

        return self.conn

    def _cursor(self):
        if self.cursor:
            return self.cursor
        self.cursor = self._conn().cursor(
            cursor_factory=psycopg2.extras.DictCursor)
        return self.cursor
