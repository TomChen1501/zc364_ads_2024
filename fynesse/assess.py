from .config import *

from . import access
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def get_five_year_interval(year):
  return f"{(year // 5) * 5}-{((year // 5) * 5) + 4}"

# This should be in the assess stage
def price_area_correlation(final_matched_df):
  data = final_matched_df[['price', 'area_sqm', 'date_of_transfer']]

  # Extract the year from `date_of_transfer` for color coding
  data['year_of_transfer'] = pd.to_datetime(data['date_of_transfer']).dt.year

  # Calculate the correlation between price and area
  correlation = data[['price', 'area_sqm']].corr().iloc[0, 1]
  print(f"Correlation between price and area: {correlation}")

  # Scatter plot with color based on the year of transfer
  plt.figure(figsize=(10, 6))
  sns.scatterplot(x='area_sqm', y='price', hue='year_of_transfer', palette='viridis', data=data)
  plt.title("Scatter Plot of Price vs. Area with Year of Transfer")
  plt.xlabel("Area (sqm)")
  plt.ylabel("Price")
  plt.legend(title='Year of Transfer', bbox_to_anchor=(1.05, 1), loc='upper left') 
  plt.show()

  # Apply the function to create a new column with five-year intervals
  data['five_year_interval'] = data['year_of_transfer'].apply(get_five_year_interval)

  # Group the data by five-year intervals and calculate the correlation between price and area for each interval
  correlations = data.groupby('five_year_interval').apply(lambda x: x['price'].corr(x['area_sqm']))

  # Display the correlation results
  print(correlations)
  print("average correlation: " + str(correlations.sum()/len(correlations)))
  return correlation

def calculate_weights(output_areas_gdf, osm_gdf, feature_name, max_distance = 20000): # only works for bng coordinates
    output_areas_gdf[f'{feature_name}_weight'] = 0.0

    for index, area in output_areas_gdf.iterrows():
        polygon = area['geometry'] 
        osm_gdf['distance'] = polygon.distance(osm_gdf['geometry'])
        osm_gdf['weight'] = 1 - (osm_gdf['distance'] / max_distance) ** 2
        osm_gdf['weight'] = np.maximum(osm_gdf['weight'], 0)
        output_areas_gdf.at[index, f'{feature_name}_weight'] = osm_gdf['weight'].sum()


def produce_correlation_features_student_percentage(features):
    correlation_features = [f'{feature}_weight' for _, feature in features] + ['school_weight', 'population density', 'Normalized Observation']
    return correlation_features

def plot_correration_matrix(correlation_features, result_gdf):
    correlation_matrix = result_gdf[correlation_features].corr()

    # Plot a heatmap
    plt.figure(figsize=(21, 14))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title('Feature Correlation Heatmap with Target Variable (y)')
    plt.tight_layout()
    plt.show()

def calculate_correlation_with_target_variable(correlation_features, result_gdf, target_variable):
    correlation_with_y = result_gdf[correlation_features].corr()[target_variable].drop(target_variable)

    plt.figure(figsize=(10, 6))
    correlation_with_y.sort_values().plot(kind='barh', color='skyblue')
    plt.title('Feature Correlation with Target Variable (y)')
    plt.xlabel('Correlation')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def calculate_osm_tags_weight_features(features, osm_tags_gdf, output_areas_gdf, max_distance = 20000):

    for key, value in tqdm(features):
        feature_gdf = osm_tags_gdf[osm_tags_gdf['tags'].apply(lambda x: x.get(key)) == value]

        calculate_weights(output_areas_gdf, feature_gdf.copy(), value, max_distance)

def categorize_housing_type(tenure_type, property_type, pp_data_gdf_bng):
    result = {}
    for t in tenure_type:
        for p in property_type:
            result[(t, p)] = pp_data_gdf_bng[(pp_data_gdf_bng['tenure_type'] == t) & (pp_data_gdf_bng['property_type'] == p)]
            print(f"Number of {t} and {p}: {result[(t, p)].shape[0]}")
    return result

def sample_categorize_housing_type(categorize_housing_type_dict, sample_size=2000):
    result = {}
    for key, value in categorize_housing_type_dict.items():
        if value.shape[0] > sample_size:
            result[key] = value.sample(frac=sample_size/value.shape[0], random_state=42)
        else:
            result[key] = value
        print(f"Number of {key}: {result[key].shape[0]}")
    return result
