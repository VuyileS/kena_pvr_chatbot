import plotly.graph_objects as go
import pandas as pd
import os
import snowflake.connector
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

# Establish your Snowflake connection
conn = snowflake.connector.connect(
    user=os.getenv('snowflake_username'),
    password=os.getenv('snowflake_password'),
    account=os.getenv('snowflake_account'),
    warehouse=os.getenv('snowflake_warehouse'),
    database=os.getenv('snowflake_database'),
    schema=os.getenv('snowflake_schema')
)

# Query data from Snowflake
data = pd.read_sql_query("""
    WITH
    DATA AS (
        SELECT
            PATIENT_ID,
            IP_ADDRESS,
            CITY,
            REGION,
            COUNTRY,
            COUNTRY_CODE,
            TIMEZONE,
            MOBILE,
            LON AS LONGITUDE,
            LAT AS LATITUDE,
            STATUS
        FROM RAW.STATIC_DATA.KENA_USERS_IP_LOCATION
    )
    SELECT
        U.CREATED_AT::DATE      AS CREATED_AT,
        D.CITY AS city, 
        D.REGION AS region, 
        D.COUNTRY,
        D.LONGITUDE AS longitude,
        D.LATITUDE AS latitude
    FROM ANALYTICS.PROD.STG_KENA_CLINIC__USERS U
    JOIN DATA D
    ON U.USER_ID = D.PATIENT_ID
""", con=conn)

# Filter for South Africa users
data = data[data['COUNTRY'] == 'South Africa']

# Convert to datetime and sort
data['timestamp'] = pd.to_datetime(data['CREATED_AT'])
data = data.sort_values('timestamp')

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['LONGITUDE'], data['LATITUDE']))

# Load world boundaries, South Africa district boundaries, and province boundaries
world = gpd.read_file("data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")  # Update with your path
south_africa_districts = gpd.read_file("data/Districts/Districts.shp")  # Update with path to district boundaries
south_africa_provinces = gpd.read_file("data/Province_Boundary/Province_Boundary.shp")  # Update with path to provincial boundaries

# Ensure all layers are in the same CRS
if south_africa_districts.crs != world.crs:
    south_africa_districts = south_africa_districts.to_crs(world.crs)
if south_africa_provinces.crs != world.crs:
    south_africa_provinces = south_africa_provinces.to_crs(world.crs)

# Prepare the base map
fig, ax = plt.subplots(figsize=(10, 6))
world.plot(ax=ax, color='lightgrey')

# Plot the districts with lighter borders
south_africa_districts.plot(ax=ax, color='none', edgecolor='lightgrey', linewidth=0.5)

# Plot the provinces with a slightly darker line for clearer distinction
south_africa_provinces.plot(ax=ax, color='none', edgecolor='black', linewidth=1)

# Function to update each frame in the animation
def update(num):
    ax.clear()  # Clear the previous frame
    world.plot(ax=ax, color='lightgrey')  # Redraw the base map
    south_africa_districts.plot(ax=ax, color='turquoise', edgecolor='black', linewidth=0.5)  # Redraw districts with lighter borders
    south_africa_provinces.plot(ax=ax, color='none', edgecolor='black', linewidth=1)  # Redraw provinces with darker borders
    
    # Set zoom to focus on South Africa
    ax.set_xlim([16, 33])  # Longitude range for South Africa
    ax.set_ylim([-35, -22])  # Latitude range for South Africa
    
    # Plot all previous points up to the current frame
    current_gdf = gdf.iloc[:num+1]
    current_gdf.plot(ax=ax, color='red', markersize=100, edgecolor='k', alpha=0.7)
    
    # Annotate the most recent city only
    latest_row = current_gdf.iloc[-1]
    ax.text(latest_row.geometry.x, latest_row.geometry.y, latest_row['CITY'], fontsize=12, ha='center', color='crimson')

    # Title showing the current timestamp
    ax.set_title(f'Geolocation Data Over Time - {latest_row["timestamp"].strftime("%Y-%m-%d %H:%M")}', fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

# Create the animation
ani = FuncAnimation(fig, update, frames=len(gdf), repeat=False, interval=1000)

# To save as a GIF, uncomment the following line (requires imagemagick)
# ani.save('geolocation_animation.gif', writer='imagemagick', fps=1)

plt.show()
