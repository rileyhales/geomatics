********
GFS Data
********

Opening GFS Files in Python
---------------------------
GFS files are complex and irregularly formatted grib files. When opening a GFS grib file in xarray, as is done by
geomatics, you will generally need to provide some additional kwargs which get used to filter out duplicate names and
data that would otherwise happen. Provide these with the xr_kwargs parameter and they will get passed to xarray through
geomatics. If you do not provide one of these keys and you are trying to open GFS or other complex data, you will
probably get an error message that looks something like this:

.. code-block:: shell

	cfgrib.dataset.DatasetBuildError: multiple values for unique key, try re-open the file with one of:
        filter_by_keys={'cfVarName': 'vis', 'typeOfLevel': 'surface'}

It wants you to provide a dictionary of additional filters used to open a subset of the complicated file. Specifically,
it wants you to use ``filter_by_keys`` as one of the dictionary keys.

.. code-block:: python

	# Usually this is enough
	{'filter_by_keys': {'cfVarName': <name of your variable>}}
	# Sometimes you also need to specify the variable's level, another of the coordinate variables/dimensions
	{'filter_by_keys': {'cfVarName': <name of your variable>, 'typeOfLevel': <variables level>}}

You can specify this in geomatics by using the optional ``xr_kwargs`` argument

.. code-block:: python
    :emphasize-lines: 5

	point_series(files,
                 var,
                 coords,
                 ... (other parameters),
                 xr_kwargs={'filter_by_keys': {'cfVarName': <name of your variable>, 'typeOfLevel': <variable's level>}}
                 )

Coordinates in GFS files
------------------------
The x/longitude/east-west coordinate is numbered from 0 to 360 degrees instead of -180 to 180 degrees. if your input
coordinates are negative, you'll need to add 360 to them in order to get the correct numbers.
