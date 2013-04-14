class job {

  file { '/root/.boto':
    ensure => present,
    source => 'puppet:///modules/job/.boto',
  }

  file { '/root/.job':
    ensure => present,
    source => 'puppet:///modules/job/.job',
  }

}
