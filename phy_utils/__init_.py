import pkg_resources

name = 'phy_utils'

try:
    __version__ = pkg_resources.get_distribution('phy_utils').version
except pkg_resources.DistributionNotFound:
    __version__ = ''
