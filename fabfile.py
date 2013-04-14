import boto.ec2
import boto
from fabric.api import *
import datetime
import pandas
import numpy as np
from ec2types import ec2types
import signal
import time
import sys

@runs_once
def price():
    now = datetime.datetime.now()
    one_day_ago = now - datetime.timedelta(days=1)
    price_history = _ec2().get_spot_price_history(
        start_time=one_day_ago.isoformat(),
        end_time=now.isoformat(),
        product_description='Linux/UNIX',
        availability_zone='us-east-1b',
    )

    data = {}
    latest_price = {}
    latest_time = {}
    for item in price_history:
        t = item.instance_type

        if t not in data:
            data[t] = []
            latest_time[t] = item.timestamp
            latest_price[t] = item.price
        else:
            if most_recent_time[t] < item.timestamp:
                most_recent_time[t] = item.timestamp
                most_recent_price[t] = item.price

        data[t].append(item.price)

    for t, prices in data.iteritems():
        item = {}
        item['recent'] = np.round(most_recent_price[t], 3)
        item['median'] = np.round(np.median(prices), 3)
        item['stddev'] = np.round(np.std(prices), 3)
        item['max'] = np.round(np.max(prices), 3)
        data[t] = item

    pricing = pandas.DataFrame(data).transpose()
    types = pandas.DataFrame(ec2types).transpose()
    types = types[['compute_units', 'memory', 'linux_cost']]

    data = pandas.concat([pricing, types], axis=1, join='inner')
    data = data.sort(['linux_cost'])
    data = data[['recent', 'median', 'stddev', 'max', 'compute_units', 'memory', 'linux_cost']]
    
    print str(data)

@runs_once
def new(instance_type, price, n=1):
    print 'Creating spot requests'
    requests = _ec2().request_spot_instances(
        price=price,
        image_id='ami-b8d147d1', # ubuntu 12.10, instance store
        count=n,
        security_groups=['default'],
        instance_type=instance_type,
        placement='us-east-1b',
        key_name=os.path.basename(env.key_filename).replace('.pem', '')
    )

    request_ids = [r.id for r in requests]

    def sigint_handler(signum, frame, terminate=True):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        print 'Caught SIGINT'
        requests = _ec2().get_all_spot_instance_requests(request_ids)
        print 'Cancelling spot requests'
        for r in requests:
            r.cancel()

        instance_ids = [r.instance_id for r in requests if r.instance_id is not None]

        if terminate and len(instance_ids) > 0:
            print 'Terminating instances'
            _ec2().terminate_instances(instance_ids)

        sys.exit(1)

    signal.signal(signal.SIGINT, sigint_handler)

    while True:
        requests = _ec2().get_all_spot_instance_requests(request_ids)
        if not all([r.state == 'open' for r in requests]):
            break

        print 'Waiting for spot requests to be fulfilled [%s]' % (
            ', '.join([r.status.code for r in requests]))
        time.sleep(5)

    print 'Spot request statuses: [%s]' % (
        ', '.join([r.status.code for r in requests]))

    active_requests = filter(lambda r: r.state == 'active', requests)
    if len(active_requests) == 0:
        print 'No requests succeeded, giving up'

    instance_ids = [r.instance_id for r in active_requests]
    for instance_id in instance_ids:
        _ec2().create_tags(instance_id, {'Name': 'worker-idle', 'type': 'worker'})

@parallel
def start(role, code_dir='code', puppet_dir='puppet'):
    sudo('apt-get update')
    sudo('apt-get -y install puppet')

    # change Name tag to role for affected instances

@runs_once
def info():
    instances = _all_instances()
    for i in instances:
        print '%10s %10s' % (i.id, i.tags['Name'])

_ec2_connection = None
def _ec2():
    global _ec2_connection
    if _ec2_connection is None:
        _ec2_connection = boto.ec2.connection.EC2Connection()
    return _ec2_connection

def _all_instances(name=None):
    filters = {'tag:type': 'worker'}
    if name:
        filters['tag:Name'] = 'worker-' + name
    filters['instance-state-name'] = 'running'
    reservations = _ec2().get_all_instances(filters=filters)
    instances = [i for r in reservations for i in r.instances]
    return instances


env.key_filename = 'job.pem'
env.hosts = [i.public_dns_name for i in _all_instances('idle')]
env.user = 'ubuntu'
