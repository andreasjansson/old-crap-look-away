define pip::install($extra = '') {
  include pip

  exec { "pip install $extra $name":
    require => Class['pip'],
    path => ['/bin', '/usr/bin', '/usr/local/bin'],
  }

}
