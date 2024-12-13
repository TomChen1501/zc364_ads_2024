from .config import *
import requests
import pymysql
import csv
import time
import pandas as pd
import math
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import osmium
import osmium.filter
from shapely.geometry import Point
from pyproj import CRS

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

def hello_world():
  print("Hello from the data science library!")

def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored 
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)


def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def populate_table_with_load_infile(conn, file_name, table_name):
    try:
      cur = conn.cursor()
      
      query = f"""
        LOAD DATA LOCAL INFILE '{file_name}'
        INTO TABLE `{table_name}`
        FIELDS TERMINATED BY ',' 
        OPTIONALLY ENCLOSED by '\"' 
        LINES STARTING BY '' 
        TERMINATED BY '\n';
      """
      cur.execute(query)
      conn.commit()

      
      print(f"Data successfully loaded from {file_name} into {table_name}.")
    
    except Exception as e:
      print(f"An error occured: {e}")   


def housing_upload_join_data(conn, year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  conn.commit()
  print('Data stored for year: ' + str(year))


def building_price_matching(buildings_with_address_df, pp_data_df):
  buildings_with_address_df['addr:housenumber'] = buildings_with_address_df['addr:housenumber'].str.strip().str.lower()
  buildings_with_address_df['addr:street'] = buildings_with_address_df['addr:street'].str.strip().str.lower()
  buildings_with_address_df['addr:postcode'] = buildings_with_address_df['addr:postcode'].str.strip().str.lower()

  pp_data_df['primary_addressable_object_name'] = pp_data_df['primary_addressable_object_name'].str.strip().str.lower()
  pp_data_df['secondary_addressable_object_name'] = pp_data_df['secondary_addressable_object_name'].str.strip().str.lower()
  pp_data_df['street'] = pp_data_df['street'].str.strip().str.lower()
  pp_data_df['postcode'] = pp_data_df['postcode'].str.strip().str.lower()

  # add a temporary index colum to keep traack of original row indices
  pp_data_df = pp_data_df.reset_index().rename(columns={'index': 'original_index'})

  # Perform an exact match on house number, street, and postcode
  initial_match = pp_data_df.merge(
      buildings_with_address_df,
      left_on=['primary_addressable_object_name', 'street', 'postcode'],
      right_on=['addr:housenumber', 'addr:street', 'addr:postcode'],
      how='inner',
      suffixes=('_pp', '_osm')
  )

  unmatched_pp = pp_data_df[~pp_data_df.index.isin(initial_match['original_index'])]

  secondary_match = unmatched_pp.merge(
      buildings_with_address_df,
      left_on=['secondary_addressable_object_name', 'street', 'postcode'],
      right_on=['addr:housenumber', 'addr:street', 'addr:postcode'],
      how='inner',
      suffixes=('_pp', '_osm')
  )

  final_matched_df = pd.concat([initial_match, secondary_match])
  return final_matched_df


def query_pp_postcode_data_within_box(conn, place_name, latitude, longitude, box_size_km):
  bounds = get_box_bounds(latitude, longitude, box_size_km)
  query = f"""SELECT pp.price, pp.primary_addressable_object_name, pp.secondary_addressable_object_name, pp.street, pp.postcode, po.latitude, pp.date_of_transfer, po.longitude 
          FROM pp_data AS pp
          INNER JOIN postcode_data AS po ON pp.postcode = po.postcode
          WHERE po.latitude BETWEEN {bounds['south']} AND {bounds['north']}
            AND po.longitude BETWEEN {bounds['west']} AND {bounds['east']};"""

  pp_data_df = query_to_dataframe(conn, query)
  return pp_data_df


def query_to_dataframe(conn, query):
  cursor = conn.cursor()
  cursor.execute(query)

  column_names = [desc[0] for desc in cursor.description]
  rows = cursor.fetchall()

  df = pd.DataFrame(rows, columns = column_names)
  return df


def get_box_bounds(lat, lon, box_size_km=1):
    """
    Calculate the approximate width and height in degrees for a box centered
    around a latitude and longitude with a given box size in kilometers.

    Parameters:
    - lat (float): Latitude of the center point in degrees.
    - lon (float): Longitude of the center point in degrees.
    - box_size_km (float): Width and height of the box in kilometers (default is 1 km).

    Returns:
    - dict: Dictionary with the bounding box's latitude and longitude offsets.
    """
    # Earth's radius in kilometers
    R = 6371.0

    half_box_km = box_size_km / 2.0

    lat_offset = (half_box_km / R) * (180 / math.pi)  # radians to degrees

    lon_offset = (half_box_km / (R * math.cos(math.radians(lat)))) * (180 / math.pi)

    # Calculate bounds
    north = lat + lat_offset
    south = lat - lat_offset
    east = lon + lon_offset
    west = lon - lon_offset

    return {
        'north': north,
        'south': south,
        'east': east,
        'west': west,
        'lat_offset': lat_offset,
        'lon_offset': lon_offset
    }


def get_buildings_within_box(place_name, tags, latitude, longitude, box_size_km=1):
  bounds = get_box_bounds(latitude, longitude, box_size_km)
  north = bounds['north']
  south = bounds['south']
  east = bounds['east']
  west = bounds['west']

  pois = ox.geometries_from_bbox(north, south, east, west, tags)
  buildings = pois[pois['building'].notnull()]

  buildings = buildings[buildings.geometry.type == 'Polygon']
  buildings['area_sqm'] = buildings.geometry.area

  address_columns = ["addr:housenumber", "addr:street", "addr:postcode"]
  buildings_with_address = buildings.dropna(subset=address_columns)

  # Create a DataFrame listing buildings with a full address and area
  buildings_with_address_df = buildings_with_address[["addr:housenumber", "addr:street", "addr:postcode", "area_sqm"]]

  # Plot the buildings
  fig, ax = plt.subplots(figsize=(10, 10))

  # Plot all buildings in the area
  buildings.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=0.5, alpha=0.6, label="Buildings without full address")

  # Plot buildings with full addresses in a different color
  buildings_with_address.plot(ax=ax, facecolor="blue", edgecolor="black", linewidth=0.5, alpha=0.8, label="Buildings with full address")

  # Set plot limits and labels
  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("Longitude")
  ax.set_ylabel("Latitude")
  ax.set_title(f"Buildings in {place_name} with and without Full Address Information")
  plt.legend()
  plt.show()

  return buildings_with_address_df


def extract_osm_data_from_tags(tags, filename):
  data = []
  fp = osmium.FileProcessor(filename).with_filter(osmium.filter.EntityFilter(osmium.osm.NODE)).with_filter(osmium.filter.TagFilter(*tags))

  # Iterate through the osmium file processor object
  for obj in fp:
      if not hasattr(obj, 'location'):
          continue
      # Extract the data before it gets removed
      obj_data = {
          "id": getattr(obj, "id", None),
          "latitude": obj.location.lat,
          "longitude": obj.location.lon,
          "tags": dict(obj.tags),
          "geometry": Point(obj.location.lon, obj.location.lat) if hasattr(obj, "location") else None
      }
      data.append(obj_data)

  # Convert the list of data to a GeoDataFrame
  crs4326 = CRS.from_string("EPSG:4326")
  osm_gdf = gpd.GeoDataFrame(data, crs=crs4326)
  return osm_gdf

def preprocess_osm_data(osm_data_origin, features, max_feature_length = 1000000): 
  osm_data = osm_data_origin.copy()
  for k, v in features:
      feature_length = osm_data[osm_data['tags'].apply(lambda x: x.get(k)) == v].shape[0]
      if feature_length > max_feature_length:
          osm_data = osm_data.drop(osm_data[osm_data['tags'].apply(lambda x: x.get(k)) == v].sample(frac=1-max_feature_length/feature_length, random_state=42).index)

  crs27700 = CRS.from_string("EPSG:27700")
  osm_data = osm_data.to_crs(crs27700)
  return osm_data

def read_and_preprocess_ns_sec_data(file_path):
  """Read the NS-SEC data from a file and return a DataFrame"""
  ns_sec_data = pd.read_csv(file_path)
  
  # Calculate the total observations for each Output Areas Code
  ns_sec_data['Total Observations'] = ns_sec_data.groupby('Output Areas Code')['Observation'].transform('sum')

  # Normalize the Observation column
  ns_sec_data['Normalized Observation'] = (ns_sec_data['Observation'] / ns_sec_data['Total Observations'])

  filtered_ns_sec_data = ns_sec_data[ns_sec_data['National Statistics Socio-economic Classification (NS-SeC) (10 categories) Code'] == 9].rename(columns={"Output Areas Code": "OA21CD"})

  # Save the result to a new file
  output_file = "filtered_normalized_output_areas_code.csv"
  filtered_ns_sec_data.to_csv(output_file, index=False)

  print(f"Normalized data saved to {output_file}")

  return filtered_ns_sec_data

def merge_and_filter_columns(output_areas_gdf, ns_sec_data, features):
    sampled_merged_gdf = output_areas_gdf.merge(ns_sec_data, on='OA21CD')

    # Drop irrelevant columns
    column_names = ['OA21CD'] + [f'{feature}_weight' for _, feature in features] + ['Normalized Observation']

    filtered_sampled_merged_gdf = sampled_merged_gdf[column_names]
    return filtered_sampled_merged_gdf

def access_population_density_data(filtered_merged_gdf, output_areas_gdf_bng):
    filtered_merged_gdf = filtered_merged_gdf.copy()
    temp_gdf = filtered_merged_gdf.merge(output_areas_gdf_bng[['OA21CD', 'area']], on='OA21CD')

    filtered_merged_gdf['population density'] = temp_gdf['Total Observations'] / temp_gdf['area']

    # Remove the outlier
    filtered_merged_gdf = filtered_merged_gdf[filtered_merged_gdf['OA21CD'] != 'E00187556'].copy()

    # Normalize the population density
    filtered_merged_gdf['population density'] = filtered_merged_gdf['population density']/filtered_merged_gdf['population density'].max()

    return filtered_merged_gdf

def access_dependent_children_data(file_path):
    dependent_children_data = pd.read_csv(file_path, skiprows = 6)
    dependent_children_data = dependent_children_data[['Local authority code (2022)', 'Local authority name (2022)', 'Age of dependent child', ' Total number of dependent children: 2021 ']]
    dependent_children_data[' Total number of dependent children: 2021 '] = dependent_children_data[' Total number of dependent children: 2021 '].replace(',', '', regex=True).replace('c', '5', regex=True).astype(int)

    dependent_children_data['Total dependent children'] = dependent_children_data.groupby('Local authority code (2022)')[' Total number of dependent children: 2021 '].transform('sum')

    dependent_children_data['Normalized dependent children'] = (dependent_children_data[' Total number of dependent children: 2021 '] / dependent_children_data['Total dependent children'])

    return dependent_children_data

def produce_filtered_merged_dependent_children_gdf(lad_featured_gdf, dependent_children_data):
    filtered_dependent_children_data = dependent_children_data[dependent_children_data['Age of dependent child'] == 0]

    merged_dependent_children_gdf = lad_featured_gdf.merge(filtered_dependent_children_data, left_on='LAD22CD', right_on='Local authority code (2022)', how='inner')

    filtered_merged_dependent_children_gdf = merged_dependent_children_gdf[['LAD22CD', 'library_weight', 'student_accommodation_weight', 'playground_weight', 'sports_centre_weight', 'pitch_weight', 'garden_weight', 'sports_hall_weight', 'fitness_centre_weight', ' Total number of dependent children: 2021 ', 'Normalized dependent children']]

    return filtered_merged_dependent_children_gdf



