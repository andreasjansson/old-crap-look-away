import boto.ec2
import boto
from fabric.api import *
import fabric
import datetime
import signal
import time
import sys
import os
import fabric.contrib.project as project
import cPickle

@runs_once
def price():
    from ec2types import ec2types
    import pandas
    import numpy as np

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
            if latest_time[t] < item.timestamp:
                latest_time[t] = item.timestamp
                latest_price[t] = item.price

        data[t].append(item.price)

    for t, prices in data.iteritems():
        item = {}
        item['recent'] = np.round(latest_price[t], 3)
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

        if terminate and instance_ids:
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
    if not active_requests:
        print 'No requests succeeded, giving up'

    instance_ids = [r.instance_id for r in active_requests]
    for instance_id in instance_ids:
        _ec2().create_tags(instance_id, {'Name': 'worker-idle',
                                         'type': 'worker'})

def ssh():
    local('ssh -i %s %s@%s' % (env.key_filename, env.user, env.host))

@parallel
def build(puppet_dir='puppet', init_filename='init.pp'):
    sudo('apt-get update')
    sudo('apt-get -y install puppet')

    if not puppet_dir.endswith('/'):
        puppet_dir += '/'
    remote_puppet_dir = '/etc/puppet'
    sudo('chown -R %s %s' % (env.user, remote_puppet_dir))
    project.rsync_project(local_dir=puppet_dir, remote_dir=remote_puppet_dir,
                          ssh_opts='-o StrictHostKeyChecking=no')
    sudo('puppet apply %s/%s' % (remote_puppet_dir, init_filename))

@parallel
def run(code_dir, script, new_role='active', workers_per_instance=None):
    # set workers_per_instance to the number of compute_units
    # for the instance type if None

    # partition data based on index, number of instances,
    # and workers_per_instance

    # change Name tag to new_role for affected instances

    pass

def info():
    i = _host_instance()
    print '%10s %10s' % (i.id, i.tags['Name'])

__ec2 = None
def _ec2():
    global __ec2
    if __ec2 is None:
        __ec2 = boto.ec2.connection.EC2Connection()
    return __ec2

__instances = None
__dns_instances = None
def _all_instances(name=None):
    global __instances
    global __dns_instances
    if __instances is None:
        cache = 'instances.cache.pkl'
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                __instances = cPickle.load(f)
        else:
            filters = {'instance-state-name': 'running',
                       'tag:type': 'worker'}
            reservations = _ec2().get_all_instances(filters=filters)
            __instances = [i for r in reservations for i in r.instances]
            with open(cache, 'wb') as f:
                cPickle.dump(__instances, f)
        __dns_instances = {i.public_dns_name: i for i in __instances}
    if name:
        return [i for i in instances if i.tags['Name'] == 'worker-' + name]
    return __instances

def _get_roledefs():
    instances = _all_instances()
    defs = {}
    for i in instances:
        role = i.tags['Name'].split('-', 1)[0]
        dns = i.public_dns_name
        if role in defs:
            defs[role].append(dns)
        else:
            defs[role] = [dns]
    return defs

def _host_instance():
    return _instance_by_dns(env.host)

def _instance_by_dns(dns):
    return __dns_instances.get(dns, None)

env.disable_known_hosts = True
env.key_filename = 'job.pem'
if not env.hosts:
    env.hosts = [i.public_dns_name for i in _all_instances()]
env.roledefs = _get_roledefs()
env.user = 'ubuntu'

