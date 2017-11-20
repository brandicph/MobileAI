import requests
import json
import logging
from math import sin, cos, sqrt, atan2, radians


def distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    rlat1 = radians(lat1)
    rlon1 = radians(lon1)
    rlat2 = radians(lat2)
    rlon2 = radians(lon2)

    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1

    a = sin(dlat / 2)**2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


LATITUDE = 55.7819451 # Latitude geocoordinate
LONGITUDE = 12.5173204 # Longitude geocoordinate
RADIUS = 500 # Radius in meters
SERVICE = 2 # Service id (Mobile=2, Other=1) - full list: https://mastedatabasen.dk/Master/antenner/tjenester.xml
TECHNOLOGY = 29 # Technology id (GSM=39, GSM-R=40, UTMS=7, LTE=29) - full list: https://mastedatabasen.dk/Master/antenner/teknologier.xml
LIMIT = 100 # Maximum entities

response = requests.get('https://mastedatabasen.dk/Master/antenner/{},{},{}.json?tjenesteart={}&teknologi={}&maxantal={}'.format(LATITUDE, LONGITUDE, RADIUS, SERVICE, TECHNOLOGY, LIMIT))
response_json = response.text
response_dict = response.json()

for value in response_dict:
    lat = float(value['wgs84koordinat']['bredde'])
    lon = float(value['wgs84koordinat']['laengde'])
    dist = distance(LATITUDE, LONGITUDE, lat, lon) * 1000
    print('type: %-8s freq: %-8s name: %-25s dist: %0.f m' % (value['teknologi']['navn'], value['frekvensbaand'], value['unik_station_navn'], dist))
