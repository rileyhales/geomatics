from setuptools import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

dependencies = ['rasterio', 'rasterstats', 'xarray', 'netcdf4', 'requests', 'shapefile']

setup(
    name='halesgis',
    packages=['halesgis'],
    version='0.0.1',
    description='Package for accessing data and APIs developed for the GEOGloWS initiative',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    project_urls=dict(Documentation='https://hales-gis.readthedocs.io',
                      Source='https://github.com/rileyhales/hales-gis'),
    license='BSD 3-Clause',
    license_family='BSD',
    package_data={'': ['*.ipynb', '*.html']},
    include_package_data=True,
    python_requires='>=3',
    install_requires=dependencies
)
