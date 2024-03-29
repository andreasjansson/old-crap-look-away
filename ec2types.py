# TODO: scrape this or get it straight from
# https://github.com/powdahound/ec2instances.info
ec2types = {
    'm1.small': {
        'name': 'M1 Small',
        'memory': 1.7,
        'compute_units': 1,
        'storage': 160,
        'ioperf': 'Moderate',
        'architecture': '32/64-bit',
        'maxips': 8,
        'linux_cost': 0.060,
        'windows_cost': 0.091,
    },
    'm1.medium': {
        'name': 'M1 Medium',
        'memory': 3.75,
        'compute_units': 2,
        'storage': 410,
        'ioperf': 'Moderate',
        'architecture': '32/64-bit',
        'maxips': 12,
        'linux_cost': 0.12,
        'windows_cost': 0.182,
    },
    'm1.large': {
        'name': 'M1 Large',
        'memory': 7.5,
        'compute_units': 4,
        'storage': 850,
        'ioperf': 'High / 500 Mbps',
        'architecture': '64-bit',
        'maxips': 30,
        'linux_cost': 0.24,
        'windows_cost': 0.364,
    },
    'm1.xlarge': {
        'name': 'M1 Extra Large',
        'memory': 15,
        'compute_units': 8,
        'storage': 1690,
        'ioperf': 'High / 1000 Mbps',
        'architecture': '64-bit',
        'maxips': 60,
        'linux_cost': 0.48,
        'windows_cost': 0.728,
    },
    't1.micro': {
        'name': 'Micro',
        'memory': 0.6,
        'compute_units': 2,
        'storage': 0,
        'ioperf': 'Low',
        'architecture': '32/64-bit',
        'maxips': 1,
        'linux_cost': 0.02,
        'windows_cost': 0.02,
    },
    'm2.xlarge': {
        'name': 'High-Memory Extra Large',
        'memory': 17.10,
        'compute_units': 6.5,
        'storage': 420,
        'ioperf': 'Moderate',
        'architecture': '64-bit',
        'maxips': 60,
        'linux_cost': 0.41,
        'windows_cost': 0.51,
    },
    'm2.2xlarge': {
        'name': 'High-Memory Double Extra Large',
        'memory': 34.2,
        'compute_units': 13,
        'storage': 850,
        'ioperf': 'High',
        'architecture': '64-bit',
        'maxips': 120,
        'linux_cost': 0.82,
        'windows_cost': 1.02,
    },
    'm2.4xlarge': {
        'name': 'High-Memory Quadruple Extra Large',
        'memory': 68.4,
        'compute_units': 26,
        'storage': 1690,
        'ioperf': 'High / 1000 Mbps',
        'architecture': '64-bit',
        'maxips': 240,
        'linux_cost': 1.64,
        'windows_cost': 2.04,
    },
    'm3.xlarge': {
        'name': 'M3 Extra Large',
        'memory': 15,
        'compute_units': 13,
        'storage': 0,
        'ioperf': 'Moderate / 500 Mbps',
        'architecture': '64-bit',
        'maxips': 60,
        'linux_cost': 0.50,
        'windows_cost': 0.78,
    },
    'm3.2xlarge': {
        'name': 'M3 Double Extra Large',
        'memory': 30,
        'compute_units': 26,
        'storage': 0,
        'ioperf': 'High / 1000 Mbps',
        'architecture': '64-bit',
        'maxips': 120,
        'linux_cost': 1.00,
        'windows_cost': 1.56,
    },
    'c1.medium': {
        'name': 'High-CPU Medium',
        'memory': 1.7,
        'compute_units': 5,
        'storage': 350,
        'ioperf': 'Moderate',
        'architecture': '32/64-bit',
        'maxips': 12,
        'linux_cost': 0.145,
        'windows_cost': 0.225,
    },
    'c1.xlarge': {
        'name': 'High-CPU Extra Large',
        'memory': 7,
        'compute_units': 20,
        'storage': 1690,
        'ioperf': 'High / 1000 Mbps',
        'architecture': '64-bit',
        'maxips': 60,
        'linux_cost': 0.58,
        'windows_cost': 0.90,
    },
    'cc1.4xlarge': {
        'name': 'Cluster Compute Quadruple Extra Large',
        'memory': 23,
        'compute_units': 33.5,
        'storage': 1690,
        'ioperf': '',
        'architecture': 'Xeon X5570',
        'maxips': 1,
        'linux_cost': 1.30,
        'windows_cost': 1.61,
    },
    'cc2.8xlarge': {
        'name': 'Cluster Compute Eight Extra Large',
        'memory': 60.5,
        'compute_units': 88,
        'storage': 3370,
        'ioperf': '',
        'architecture': 'Xeon E5-2670',
        'maxips': 240,
        'linux_cost': 2.40,
        'windows_cost': 2.97,
    },
    'cg1.4xlarge': {
        'name': 'Cluster GPU Quadruple Extra Large',
        'memory': 22,
        'compute_units': 33.5,
        'storage': 1690,
        'ioperf': '',
        'architecture': 'Xeon X5570',
        'maxips': 1,
        'linux_cost': 2.10,
        'windows_cost': 2.60,
    },
    'hi1.4xlarge': {
        'name': 'High I/O Quadruple Extra Large',
        'memory': 60.5,
        'compute_units': 35,
        'storage': 2048,
        'ioperf': '',
        'architecture': '64-bit',
        'maxips': 1,
        'linux_cost': 3.10,
        'windows_cost': 3.58,
    },
    'hs1.8xlarge': {
        'name': 'High Storage Eight Extra Large',
        'memory': 117.00,
        'compute_units': 35,
        'storage': 49152,
        'ioperf': '',
        'architecture': '64-bit',
        'maxips': 1,
        'linux_cost': 4.600,
        'windows_cost': 4.931,
    },
    'cr1.8xlarge': {
        'name': 'High Memory Cluster Eight Extra Large',
        'memory': 244.00,
        'compute_units': 88,
        'storage': 240,
        'ioperf': '',
        'architecture': '64-bit',
        'maxips': 1,
        'linux_cost': 3.500,
        'windows_cost': 3.831,
    },
}
