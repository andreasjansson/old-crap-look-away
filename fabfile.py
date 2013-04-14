import boto.ec2
import boto
from fabric.api import *
import datetime
import pandas
import numpy as np
from ec2types import ec2types

def price(instance_type=None):
    now = datetime.datetime.now()
    one_day_ago = now - datetime.timedelta(days=1)
    price_history = _ec2().get_spot_price_history(
        start_time=one_day_ago.isoformat(),
        end_time=now.isoformat(),
        instance_type=instance_type,
        product_description='Linux/UNIX',
        availability_zone='us-east-1b',
    )

    data = {}
    for item in price_history:
        if item.instance_type not in data:
            data[item.instance_type] = []
        data[item.instance_type].append(item.price)

    for t, prices in data.iteritems():
        item = {}
        item['mean'] = np.round(np.mean(prices), 3)
        item['median'] = np.round(np.median(prices), 3)
        item['stddev'] = np.round(np.std(prices), 3)
        item['max'] = np.round(np.max(prices), 3)
        data[t] = item

#    import cPickle
#    with open('price_history.cache.pkl', 'wb') as f:
#        cPickle.dump(data, f)

#    import cPickle
#    with open('price_history.cache.pkl', 'r') as f:
#        data = cPickle.load(f)

    pricing = pandas.DataFrame(data).transpose()
    types = pandas.DataFrame(ec2types).transpose()
    types = types[['compute_units', 'memory', 'linux_cost']]

    data = pandas.concat([pricing, types], axis=1)
    data = data.sort(['linux_cost'])
    data = data[['mean', 'median', 'stddev', 'max', 'compute_units', 'memory', 'linux_cost']]
    
    print str(data)

_ec2_connection = None
def _ec2():
    global _ec2_connection
    if _ec2_connection is None:
        _ec2_connection = boto.ec2.connection.EC2Connection()
    return _ec2_connection
