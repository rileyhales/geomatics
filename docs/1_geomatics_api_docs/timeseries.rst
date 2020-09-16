**********
timeseries
**********

These functions create timeseries from multidimensional data array files (netCDF, grib, HDF5, GeoTiff) for a single variable.
These functions are most heavily tested with up to 4 dimensional data (x, y, z, time) but have been tested on dimensionally larger
datasets.

Explanation of Multidimensional Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Some data are 1 dimensional. The 1 dimension is usually an identifier number, a length, or a size. The single variable varies in
value with respect to that 1 dimension. A common example of 1 dimensional data are the National Water Model netCDF files. They
contain several 1 dimensional datasets such as:

- Stream discharge: flow in the array depends on the stream's identifying number (stored in the 1-dimension)
- Streamflow velocity: the velocity data array varies with respect to a single other condition such as the stream's identifier or time.

Most spatial data is 2: an X dimension, usually lon or longitude, and a Y dimension, usually lat or latitude, where. The data in the
single variable changes with respect to the 2 dimensions. Some examples of data commonly stored in this format include:

- DEM data
- Most geospatial raster/gridded data

Some data are 3 dimensional: X, Y, and Z dimensions which usually correspond to longitude, latitude, and either altitude, elevation
or depth. Common examples of data stored in this format include:

- Meteorological observation/simulation data where measurements are made at several elevations
- Soil moisture maps where data varies in 2 dimensions across the land surface and also with depth.

In each of the previous examples, data varied with respect to spatial dimensions only. Some files include multiple time steps or data
in the same file. Thus the data would be spatiotemporal data where each varies with respect to time or the time of observation is relevant.
Each of the previous 1, 2, and 3 dimensional data could be 2, 3, and 4 dimensional data where the additional dimension is time. While all data
does have a time associated with it's creation or forecast time, not all file structures list it explicitly.

Handling Time Information
~~~~~~~~~~~~~~~~~~~~~~~~~
Most data use 2 spatial dimensions. If this is the case and your data contains time-sensitive information, the time information is probably
in the file's name. In this case, use the `strp` keyword argument to provide a pattern for extracting the date.

If your data contains a temporal dimension, you usually just need to provide the name of the time dimension in the file (e.g. time, t,
simulation_time) and geomatics will detect the correct times for you.

Timeseries Functions
~~~~~~~~~~~~~~~~~~~~

.. automodule:: geomatics.timeseries
	:members: time_series
