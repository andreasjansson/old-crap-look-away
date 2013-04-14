Exec {
  path => '/usr/bin:/bin:/usr/sbin:/sbin',
}

include python

package { ['python-numpy', 'python-scipy']:
  ensure => present,
}

include pymad

pip::install {'boto': }
pip::install {'pika': }
pip::install {'pycassa': }
pip::install {'scikit-learn': require => Package['python-numpy'], }

include job
