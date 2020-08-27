#!/usr/bin/env bash
# Assuming this file is always run on a system that contains curl
# Assumes there is a ~/.netrc file containing you username & password to earthdata.nasa.gov
# example .netrc file contents: machine urs.earthdata.nasa.gov login your_user_name password your_password

mkdir hdf_data
cd hdf_data
cat ../hdf_sample_data_urls.txt | tr -d '\r' | xargs -n 1 -P 4 curl -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies