class job {

  file { '/home/ubuntu/.boto':
    ensure => present,
    source => 'puppet:///modules/job/.boto',
  }

  file { '/home/ubuntu/.job':
    ensure => present,
    source => 'puppet:///modules/job/.job',
  }

}
