from setuptools import setup

version = '0.9'

with open('README.md', 'r') as readme:
    long_description = readme.read()

with open('requirements.txt', 'r') as req:
    install_requires = req.read().splitlines()

setup(
    name='geomatics',
    packages=['geomatics'],
    version=version,
    description='Geospatial tools for creating timeseries of from n-dimensional scientific data file formats',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    project_urls=dict(Documentation='https://geomatics.readthedocs.io',
                      Source='https://github.com/rileyhales/geomatics'),
    license='BSD 3-Clause',
    python_requires='>=3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
    ],
    install_requires=install_requires
)

# todo increment version number to 0.10
# todo test the download gldas function
# todo test the pygrib engine
# todo note that the cfgrib standard dimension names are latitudes and longitudes
# todo write a test for all functions with gldas data
# todo collect speed test information for a LARGE sample of data
# todo docs about pygrib
# todo test docs in the docs environment
# todo add the dateutil dependency
# todo drop the geojson to shapefile with pyshp function? geopandas can do that...
