import requests
import datetime
import os
import json

__all__ = ['download_noaa_gfs', 'get_livingatlas_geojson']


def download_noaa_gfs(save_path, steps):
    """
    Downloads Grib files containing the latest NOAA GFS forecast. The files are saved to a specified directory and are
        named for the timestamp of the forecast and the time that the forecast is predicting for. The timestamps are in
        YYYYMMDDHH time format. E.G a file named gfs_2020010100_2020010512.grb means that the file contains data from
        the forecast created Jan 1 2020 at 00:00:00 for the time Jan 5 2020 at 12PM.

    Args:
        save_path: an absolute file path to the directory where you want to save the gfs files
        steps: the number of 6 hour forecast steps to download. E.g. 4 steps = 1 day

    Returns:
        None
    """
    # determine which forecast we should be looking for
    now = datetime.datetime.utcnow() - datetime.timedelta(hours=4)
    if now.hour >= 18:
        fc_hour = '18'
    elif now.hour >= 12:
        fc_hour = '12'
    elif now.hour >= 6:
        fc_hour = '06'
    else:
        fc_hour = '00'
    fc_date = now.strftime('%Y%m%d')
    timestamp = datetime.datetime.strptime(fc_date + fc_hour, '%Y%m%d%H')

    fc_time_steps = []
    for step in range(steps):
        step = str(6 * (step + 1))
        while len(step) < 3:
            step = '0' + step
        fc_time_steps.append(step)

    for step in fc_time_steps:
        # build the url to download the file from
        url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t' + fc_hour + 'z.pgrb2.0p25.f' + \
              step + '&all_lev=on&all_var=on&dir=%2Fgfs.' + fc_date + '%2F' + fc_hour

        # set the file name: gfs_DATEofFORECAST_TIMESTEPofFORECAST.grb
        file_timestep = timestamp + datetime.timedelta(hours=int(step))
        filename = 'gfs_{0}_{1}.grb'.format(timestamp.strftime('%Y%m%d%H'), file_timestep.strftime("%Y%m%d%H"))
        filepath = os.path.join(save_path, filename)

        # download the file
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=10240):
                    if chunk:
                        f.write(chunk)
    return


def get_livingatlas_geojson(location):
    """
    Requests a geojson from the ESRI living atlas services for World Regions or Generalized Country Boundaries

    Args:
        location: the name of the Country or World Region, properly spelled and capitalized

    Returns:
        a json python object, dict like
    """
    countries = [
        'Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antarctica',
        'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
        'Bahrain', 'Baker Island', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda',
        'Bhutan', 'Bolivia', 'Bonaire', 'Bosnia and Herzegovina', 'Botswana', 'Bouvet Island', 'Brazil',
        'British Indian Ocean Territory', 'British Virgin Islands', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso',
        'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic',
        'Chad', 'Chile', 'China', 'Christmas Island', 'Cocos Islands', 'Colombia', 'Comoros', 'Congo', 'Congo DRC',
        'Cook Islands', 'Costa Rica', "Côte d'Ivoire", 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic',
        'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
        'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Falkland Islands', 'Faroe Islands', 'Fiji',
        'Finland', 'France', 'French Guiana', 'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia',
        'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Glorioso Island', 'Greece', 'Greenland', 'Grenada',
        'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',
        'Heard Island and McDonald Islands', 'Honduras', 'Howland Island', 'Hungary', 'Iceland', 'India',
        'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Jan Mayen', 'Japan',
        'Jarvis Island', 'Jersey', 'Johnston Atoll', 'Jordan', 'Juan De Nova Island', 'Kazakhstan', 'Kenya',
        'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya',
        'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',
        'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 'Mayotte', 'Mexico', 'Micronesia',
        'Midway Islands', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique',
        'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger',
        'Nigeria', 'Niue', 'Norfolk Island', 'North Korea', 'Northern Mariana Islands', 'Norway', 'Oman',
        'Pakistan', 'Palau', 'Palestinian Territory', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru',
        'Philippines', 'Pitcairn', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Réunion', 'Romania',
        'Russian Federation', 'Rwanda', 'Saba', 'Saint Barthelemy', 'Saint Eustatius', 'Saint Helena',
        'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin', 'Saint Pierre and Miquelon',
        'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia',
        'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten', 'Slovakia', 'Slovenia',
        'Solomon Islands', 'Somalia', 'South Africa', 'South Georgia', 'South Korea', 'South Sudan', 'Spain',
        'Sri Lanka', 'Sudan', 'Suriname', 'Svalbard', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Tajikistan',
        'Tanzania', 'Thailand', 'The Former Yugoslav Republic of Macedonia', 'Timor-Leste', 'Togo', 'Tokelau',
        'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu',
        'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay',
        'US Virgin Islands', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela', 'Vietnam', 'Wake Island',
        'Wallis and Futuna', 'Yemen', 'Zambia', 'Zimbabwe']
    regions = ('Antarctica', 'Asiatic Russia', 'Australia/New Zealand', 'Caribbean', 'Central America', 'Central Asia',
               'Eastern Africa', 'Eastern Asia', 'Eastern Europe', 'European Russia', 'Melanesia', 'Micronesia',
               'Middle Africa', 'Northern Africa', 'Northern America', 'Northern Europe', 'Polynesia', 'South America',
               'Southeastern Asia', 'Southern Africa', 'Southern Asia', 'Southern Europe', 'Western Africa',
               'Western Asia', 'Western Europe')

    # get the geojson data from esri
    base = 'https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/'
    if location in regions:
        url = base + 'World_Regions/FeatureServer/0/query?f=pgeojson&outSR=4326&where=REGION+%3D+%27' + location + '%27'
    elif location in countries:
        url = base + 'World__Countries_Generalized_analysis_trim/FeatureServer/0/query?f=pgeojson&outSR=4326&where=NAME+%3D+%27' + location + '%27'
    else:
        raise Exception('Country or World Region not recognized')

    req = requests.get(url=url)
    return json.loads(req.text)
