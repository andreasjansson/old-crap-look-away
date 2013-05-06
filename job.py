import pycassa
import pika
import json
import ConfigParser
import os
import sys
import datetime
import socket
import cPickle

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
        self.rabbitmq_conn = None
        self.rabbitmq_channel = None
        self.cassandra_pool = None
        self.hostname = socket.gethostname()

    def put_data(self, data):
        self.rabbitmq().basic_publish(
            exchange='',
            routing_key=self.name,
            body=json.dumps(data)
        )

    def run_worker(self, do_work):
        while True:
            method, header_frame, body = self.rabbitmq().basic_get(self.name)

            if method is None: # empty queue
                break

            try:
                data = json.loads(body)
            except ValueError:
                self.log('Failed to parse as json: %s' % body)
                return

            #try:
            do_work(self, data)
            self.rabbitmq().basic_ack(method.delivery_tag)
            #except BlahException, e:
            #    self.log_exception(e)

    def log(self, message):
        print message
        self.cassandra('log').insert(
            self.hostname, {datetime.datetime.utcnow(): message})

    def log_exception(self, exception):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        self.log('EXCEPTION: %s in %s (line %d)' % (
            exc_type, fname, exc_tb.tb_lineno))

    def store(self, key, value):
        self.cassandra('data').insert(key, {'data': cPickle.dumps(value)})

    def store_instance(self, key, value):
        self.cassandra('instance_data').insert(
            key, {'%d-%d' % (INDEX(), COUNT()): cPickle.dumps(value)})

    def clear(self):
        try:
            self.rabbitmq().queue_delete(queue=self.name)
        except Exception, e:
            sys.stderr.write('Failed to clear queue: %s\n' % str(e))
        try:
            self.cassandra_sys().drop_keyspace(self.name)
        except Exception, e:
            sys.stderr.write('Failed to clear database: %s\n' % str(e))

    def get_log(self, max_count=2000):
        return list(self.cassandra('log')
                    .get_range(column_count=max_count))

    def get_data(self, row_count=None):
        data = {}
        for key, columns in self.cassandra('data').get_range(
            row_count=row_count):
            data[key] = cPickle.loads(columns['data'])
        return data

    def get_data_count(self):
        count = 0
        for _ in self.cassandra('data').get_range(
            column_count=0, filter_empty=False):
            count +=1
        return count

    def reduce_instances(self, key, count, reducer):
        columns = ['%d-%d' % (i, count) for i in range(count)]
        data = []
        for value in self.cassandra('instance_data').get(
            key=key, columns=columns).values():
            data.append(cPickle.loads(value))
        return reduce(reducer, data)

    def get_queue_length(self):
        return self.rabbitmq().queue_declare(
            self.name, passive=True).method.message_count

    def rabbitmq(self):
        if self.rabbitmq_channel is not None:
            return self.rabbitmq_channel
        self.rabbitmq_conn = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=config('rabbitmq')['host'],
                port=int(config('rabbitmq')['port']),
                credentials=pika.PlainCredentials(
                    config('rabbitmq')['user'],
                    config('rabbitmq')['pass'],
                )
            ))
        self.rabbitmq_channel = self.rabbitmq_conn.channel()
        self.rabbitmq_channel.queue_declare(queue=self.name)

        return self.rabbitmq_channel

    def cassandra_sys(self, column_family=None):
        server = '%s:%s' % (config('cassandra')['host'],
                            int(config('cassandra')['port']))

        credentials = {'username': config('cassandra')['user'],
                       'password': config('cassandra')['pass']}

        return pycassa.system_manager.SystemManager(server, credentials)
        
    def cassandra(self, column_family=None):
        if self.cassandra_pool is None:
            server = '%s:%s' % (config('cassandra')['host'],
                                int(config('cassandra')['port']))

            credentials = {'username': config('cassandra')['user'],
                           'password': config('cassandra')['pass']}

            sys = pycassa.system_manager.SystemManager(server, credentials)
            if self.name not in sys.list_keyspaces():
                sys.create_keyspace(self.name,
                                    strategy_options={'replication_factor': '1'})
                sys.create_column_family(self.name, 'data',
                                         key_validation_class=pycassa.ASCII_TYPE,
                                         comparator_type=pycassa.ASCII_TYPE,
                                         default_validation_class=pycassa.BYTES_TYPE)
                sys.create_column_family(self.name, 'instance_data',
                                         key_validation_class=pycassa.ASCII_TYPE,
                                         comparator_type=pycassa.ASCII_TYPE,
                                         default_validation_class=pycassa.BYTES_TYPE)
                sys.create_column_family(self.name, 'log',
                                         comparator_type=pycassa.TIME_UUID_TYPE,
                                         key_validation_class=pycassa.ASCII_TYPE,
                                         default_validation_class=pycassa.ASCII_TYPE)

            self.cassandra_pool = pycassa.pool.ConnectionPool(
                self.name, [server], credentials, timeout=60
            )

        if column_family:
            return pycassa.ColumnFamily(self.cassandra_pool, column_family)
        return self.cassandra_pool


def INDEX():
    host_count = int(os.environ['HOST_COUNT'])
    host_index = int(os.environ['HOST_INDEX'])
    return int(os.environ['INSTANCE_INDEX']) + host_index * int(os.environ['INSTANCE_COUNT'])

def COUNT():
    return int(os.environ['HOST_COUNT']) * int(os.environ['INSTANCE_COUNT'])

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
