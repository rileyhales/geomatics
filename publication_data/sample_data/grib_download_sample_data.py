import os
import datetime
import geomatics


if __name__ == '__main__':
    save_sample_data_dir = os.path.join(os.path.dirname(__file__), 'grib_data')
    steps = 20
    timestamp = datetime.datetime(year=2020, month=6, day=15, hour=18, minute=0, second=0)
    sample_vars = ['ABSV', 'ACPCP', 'ALBDO', 'APCP']
    convertlatlon = True
    geomatics.data.download_noaa_gfs(save_sample_data_dir, steps, timestamp, sample_vars, convertlatlon)
