class python {

  package { ['python2.7', 'python2.7-dev', 'python-setuptools', ]:
    ensure => present,
  }

}
