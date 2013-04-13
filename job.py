import pycassa
import pika
import json

_config = None
def config(section):
    global _config
    if _config is None:
        _config = parse_config()
    return dict(_config.items(section))

def parse_config():
    config = ConfigParser.RawConfigParser()
    config.read(['/etc/job.cfg', os.path.expanduser('~/.job.cfg')])
    return config

class Job(object):

    def __init__(self, name):
        self.name = name

    def put_data(self, data):
        self.rabbitmq().basic_publish(
            exchange='',
            routing_key=self.name,
            body=json.dumps(data)
        )

    def run_worker(self, do_work):
        def callback(channel, method, properties, body):
            try:
                data = json.loads(body)
            except ValueError:
                self.log('Failed to parse as json: %s' % body)
                return

            try:
                do_work(self, body)
            except Exception, e:
                self.log_exception(e)

        self.rabbitmq().basic_consume(callback, queue=self.name, no_ack=True)

    def log(self, message):
        column_family = pycassa.ColumnFamily(self.cassandra(), 'log')
        column_family.insert('log', {time.time(): message})

    def log_exception(self, exception):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        self.log('EXCEPTION: %s in %s (line %d)' % (
            exc_type, fname, exc_tb.tb_lineno))

    def store(self, key, value):
        column_family = pycassa.ColumnFamily(self.cassandra(), 'data')
        column_family.insert(key, json.dumps(value))

    def rabbitmq(self):
        if self.rabbitmq_channel is not None:
            return self.rabbitmq_channel
        self.rabbitmq_conn = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=config('rabbitmq')['host'],
                port=config('rabbitmq')['port'],
            ))
        self.rabbitmq_channel = self.rabbitmq_conn.channel()
        self.rabbitmq_channel.queue_declare(queue=self.name)

        return self.rabbitmq_channel

    def cassandra(self):
        if self.cassandra_pool is not None:
            return self.cassandra_pool

        server = '%s:%s' % (config('cassandra')['host'],
                            config('cassandra')['port'])
            

        sys = pycassa.system_manager.System_Manager(server)
        if self.name not in sys.list_keyspaces():
            sys.create_keyspace(self.name,
                                strategy_options={'replication_factor': '1'})
            sys.create_column_family(self.name, 'data')
            sys.create_column_family(self.name, 'log')

        self.cassandra_pool = pycassa.pool.ConnectionPool(self.name, server))

        return self.cassandra_pool
