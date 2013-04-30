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
from ec2types import ec2types

@runs_once
def price():
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
    data = data[['recent', 'median', 'stddev', 'max', 'compute_units',
                 'memory', 'linux_cost']]
    
    print str(data)

@runs_once
def new(instance_type, price, n=1):

    uncache()

    ubuntu1304_instance_store = 'ami-762d491f'
    ubuntu1304_hvm = 'ami-08345061'

    print 'Creating spot requests'
    requests = _ec2().request_spot_instances(
        price=price,
        image_id=ubuntu1304_hvm if instance_type in ['cc2.8xlarge', 'cr1.8xlarge']
            else ubuntu1304_instance_store,
        count=n,
        security_groups=['default'],
        instance_type=instance_type,
        placement='us-east-1b',
        key_name=os.path.basename(env.key_filename).replace('.pem', ''),
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

def scp(remote_path, local_path='.'):
    local('scp -C -i %s %s@%s:"%s" %s' % (env.key_filename, env.user, env.host, remote_path, local_path))

@parallel
def build(puppet_dir='puppet', init_filename='init.pp'):
    sudo('apt-get update')
    sudo('apt-get -y install puppet')
    sudo('chmod 777 /opt')

    if not puppet_dir.endswith('/'):
        puppet_dir += '/'
    remote_puppet_dir = '/etc/puppet'
    sudo('chown -R %s %s' % (env.user, remote_puppet_dir))
    project.rsync_project(local_dir=puppet_dir, remote_dir=remote_puppet_dir,
                          ssh_opts='-o StrictHostKeyChecking=no')
    sudo('puppet apply %s/%s' % (remote_puppet_dir, init_filename))

@parallel
def run(job_name, script, args='', code_dir='.', n=None,
        exclude=['.git', 'puppet']):
    instance = _host_instance()
    workers_per_instance = n

    if n is None:
        workers_per_instance = ec2types[instance.instance_type]['compute_units']
    else:
        workers_per_instance = int(workers_per_instance)

    if not code_dir.endswith('/'):
        code_dir += '/'

    remote_code_dir = '/opt/' + job_name
    exclude_opts = ' '.join(['--exclude ' + pattern for pattern in exclude])
    project.rsync_project(local_dir=code_dir, remote_dir=remote_code_dir,
                          extra_opts=exclude_opts,
                          ssh_opts='-o StrictHostKeyChecking=no')

    sudo('chmod +x %s/%s' % (remote_code_dir, script))
    sudo('echo > /var/log/%s.stdout.log' % job_name)
    sudo('echo > /var/log/%s.stderr.log' % job_name)

    sudo('''echo '
instance $N
script
    HOST_INDEX=%d HOST_COUNT=%d INSTANCE_INDEX=$N INSTANCE_COUNT=%d %s/%s %s %s >> /var/log/%s.stdout.log 2>> /var/log/%s.stderr.log
end script' > /etc/init/%s.conf''' %
         (_host_index(), len(env.hosts), workers_per_instance, remote_code_dir,
          script, job_name, args, job_name, job_name, job_name))

    for i in xrange(workers_per_instance):
        sudo('start %s N=%d' % (job_name, i))

    _set_instance_name(instance, job_name)

    log(True)

@parallel
def log(from_beginning=False):
    job_name = _host_role()
    sudo('tail %s -n0 -f /var/log/%s.stdout.log /var/log/%s.stderr.log' %
         ('-c +1' if from_beginning else '', job_name, job_name))

@parallel
def stop(n=None):
    workers_per_instance = n

    job_name = _host_role()
    instance = _host_instance()
    if workers_per_instance is None:
        workers_per_instance = ec2types[instance.instance_type]['compute_units']
    else:
        workers_per_instance = int(workers_per_instance)

    for i in xrange(workers_per_instance):
        sudo('stop %s N=%d' % (job_name, i), warn_only=True, quiet=True)

    _set_instance_name(instance, 'idle')

@parallel
def terminate():
    uncache()

    instance_id = _host_instance().id
    print 'Terminating instance %s' % instance_id
    
    _ec2().terminate_instances([instance_id])

@runs_once
def info():
    import dateutil.parser
    import dateutil.tz

    #instances = _all_instances()
    reservations = _ec2().get_all_instances()
    instances = [i for r in reservations for i in r.instances]

    for i in instances:
        launch_time = dateutil.parser.parse(i.launch_time)
        launch_time = launch_time.astimezone(dateutil.tz.tzlocal())
        print '%10s %15s %15s %15s %10s %20s' % (
            i.id, i.ip_address, i.tags['Name'], i.instance_type, i.state,
            launch_time.strftime('%Y-%m-%d %H:%M:%S'))

def uncache():
    if os.path.exists('instances.cache.pkl'):
        os.unlink('instances.cache.pkl')

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
            filters = {'tag:type': 'worker',
                       'instance-state-name': 'running'}
            reservations = _ec2().get_all_instances(filters=filters)
            __instances = [i for r in reservations for i in r.instances]
            with open(cache, 'wb') as f:
                cPickle.dump(__instances, f)
        __dns_instances = {i.ip_address: i for i in __instances}
    if name:
        return [i for i in instances if i.tags['Name'] == 'worker-' + name]
    return __instances

def _get_roledefs():
    instances = _all_instances()
    defs = {}
    for i in instances:
        role = i.tags['Name'].split('-', 1)[1]
        dns = i.ip_address
        if role in defs:
            defs[role].append(dns)
        else:
            defs[role] = [dns]
    return defs

def _host_instance():
    return _instance_by_dns(env.host)

def _host_index():
    return env.hosts.index(env.host)

def _host_role():
    return _host_instance().tags['Name'].split('-', 1)[1]

def _instance_by_dns(dns):
    return __dns_instances.get(dns, None)

def _set_instance_name(instance, name):
    instance.tags['Name'] = 'worker-' + name
    _ec2().create_tags(instance.id, {'Name': 'worker-' + name})
    uncache()

env.roledefs = _get_roledefs()
env.disable_known_hosts = True
env.key_filename = 'job.pem'

if not env.hosts:
    if env.roles:
        env.hosts = [i for r in env.roles for i in env.roledefs[r]]
    if not env.hosts:
        env.hosts = [i.ip_address for i in _all_instances()]
env.user = 'ubuntu'

