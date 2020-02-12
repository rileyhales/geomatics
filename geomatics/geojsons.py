import json
import shapefile
import requests


def geojson_to_shp(geojson, savepath):
    """
    Turns a valid dict, json, or geojson containing polygon data in a geographic coordinate system into a shapefile

    Args:
        geojson: a valid geojson as a dictionary or json python object
        savepath: the full file path to save the shapefile to, including the file_name.shp

    Returns:

    """
    # turn the geojson into a dictionary if it isn't
    if not isinstance(geojson, dict):
        try:
            geojson = json.loads(geojson)
        except json.JSONDecodeError:
            raise Exception('Unable to extract a dictionary or json like object from the argument geojson')

    # create the shapefile
    fileobject = shapefile.Writer(target=savepath, shpType=shapefile.POLYGON, autoBalance=True)

    # label all the columns in the .dbf
    geomtype = geojson['features'][0]['geometry']['type']
    if geojson['features'][0]['properties']:
        for attribute in geojson['features'][0]['properties']:
            fileobject.field(str(attribute), 'C', '30')
    else:
        fileobject.field('Name', 'C', '50')

    # add the geometry and attribute data
    for feature in geojson['features']:
        # record the geometry
        if geomtype == 'Polygon':
            fileobject.poly(polys=feature['geometry']['coordinates'])
        elif geomtype == 'MultiPolygon':
            for i in feature['geometry']['coordinates']:
                fileobject.poly(polys=i)

        # record the attributes in the .dbf
        if feature['properties']:
            fileobject.record(**feature['properties'])
        else:
            fileobject.record('unknown')

    # close writing to the shapefile
    fileobject.close()

    # create a prj file
    with open(savepath + '.prj', 'w') as prj:
        prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
                  'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')

    return


def request_livingatlas_geojson(location):
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
