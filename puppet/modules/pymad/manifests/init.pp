class pymad
{
  exec { 'wget -O /opt/pymad-0.6.tar.gz http://spacepants.org/src/pymad/download/pymad-0.6.tar.gz':
    creates => '/opt/pymad-0.6.tar.gz',
    alias => 'wget',
  }

  exec { 'tar xzf pymad-0.6.tar.gz':
    cwd => '/opt',
    creates => '/opt/pymad-0.6',
    alias => 'untar',
    require => Exec['wget'],
  }

  exec { 'python config_unix.py':
    cwd => '/opt/pymad-0.6',
    alias => 'config',
    creates => '/opt/pymad-0.6/Setup',
    require => [Exec['untar'], Class['Python']],
  }

  package { 'libmad0-dev':
    ensure => present,
  }

  exec { 'python setup.py install':
    cwd => '/opt/pymad-0.6',
    require => [Exec['config'], Package['libmad0-dev']],
  }
}
