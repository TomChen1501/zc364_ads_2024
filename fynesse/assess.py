from .config import *

from . import access
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
