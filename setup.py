from setuptools import setup
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("BeAMF").version
    # print(__version__)
except DistributionNotFound:
     # package is not installed
    pass

if __name__ == "__main__":
    setup()
