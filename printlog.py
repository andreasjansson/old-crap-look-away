#!/usr/bin/python

import job
import sys
import pycassa
import datetime

def main(name):
    j = job.Job(name)
    for host, columns in j.get_log():
        print '**** %s ****' % host
        for time_uuid, message in columns.iteritems():
            time = datetime.datetime.fromtimestamp(
                pycassa.util.convert_uuid_to_time(time_uuid))
            print '%s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), message)
        print ''
    

if __name__ == '__main__':
    main(sys.argv[1])
