from setuptools import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name='geomatics',
    packages=['geomatics'],
    version='0.1',
    description='GIS tools developed by Riley Hales for the BYU Hydroinformatics Lab',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    project_urls=dict(Documentation='https://geomatics.readthedocs.io',
                      Source='https://github.com/rileyhales/geomatics'),
    license='BSD 3-Clause',
    license_family='BSD',
    python_requires='>=3',
    install_requires=['rasterio', 'rasterstats', 'xarray', 'netcdf4', 'python-dateutil', 'numpy', 'pandas']
)
