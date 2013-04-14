class pip {

  package { 'python-pip':
    ensure => present,
    require => Class['Python'],
  }

}
