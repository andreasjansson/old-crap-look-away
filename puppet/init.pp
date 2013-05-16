Exec {
  path => '/usr/bin:/bin:/usr/sbin:/sbin',
}

include python

package { ['python-numpy', 'python-scipy']:
  ensure => present,
}

package { 'libpq-dev':
  ensure => present,
}

include pymad

pip::install {'boto': }
pip::install {'pika': }
pip::install {'pycassa': }
pip::install {'scikit-learn': require => Package['python-numpy'], }
pip::install {'pandas': require => Package['python-numpy'], }
pip::install {'psycopg2': require => Package['libpq-dev'], }

include job
