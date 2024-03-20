import geostatspy
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
from shapely.geometry import Point
import numpy as np

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tobler.area_weighted import area_interpolate
import seaborn as sns 

from scipy.interpolate import NearestNDInterpolator
import plotly.express as px

import itertools
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from rasterio.transform import Affine
import rasterio
import rasterio.mask
import shapely

from pykrige.ok import OrdinaryKriging
from scipy.interpolate import NearestNDInterpolator
from tobler.area_weighted import area_interpolate

from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate  import RBFInterpolator

# Function to adjust the date and time
def adjust_datetime(date_str):

    # Split the date and time
    date_part, time_part = date_str.split(' ')
    
    # Check if the time is '24:00' and adjust
    if time_part == '24:00':
        time_part = '00:00'
        date_str = date_part + ' ' + time_part
        return date_str

    else:
        # Return the original datetime string
        return date_str

def pixel2poly(x, y, z, resolution):
    """
    x: x coords of cell
    y: y coords of cell
    z: matrix of values for each (x,y)
    resolution: spatial resolution of each cell
    """
    polygons = []
    values = []
    half_res = resolution / 2
    for i, j  in itertools.product(range(len(x)), range(len(y))):
        minx, maxx = x[i] - half_res, x[i] + half_res
        miny, maxy = y[j] - half_res, y[j] + half_res
        polygons.append(Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]))
        if isinstance(z, (int, float)):
            values.append(z)
        else:
            values.append(z[j, i])
    return polygons, values

def export_kde_raster(Z, XX, YY, min_x, max_x, min_y, max_y, proj, filename):
    '''Export and save a kernel density raster.'''

    # Get resolution
    xres = (max_x - min_x) / len(XX)
    yres = (max_y - min_y) / len(YY)

    # Set transform
    transform = Affine.translation(min_x - xres / 2, min_y - yres / 2) * Affine.scale(xres, yres)

    # Export array as raster
    with rasterio.open(
            filename,
            mode = "w",
            driver = "GTiff",
            height = Z.shape[0],
            width = Z.shape[1],
            count = 1,
            dtype = Z.dtype,
            crs = proj,
            transform = transform,
    ) as new_dataset:
            new_dataset.write(Z, 1)

# Define the inverse distance function
def inverse_distance_weighting(x, y, df, power=2):
    distances = np.sqrt((df['longitude'] - x) ** 2 + (df['latitude'] - y) ** 2)
    weights = 1.0 / distances ** power
    return np.average(df['PM2.5'], weights=weights)


### Load Data
bici_data = gpd.read_file('ciclorruta.gpkg')
municipalities = gpd.read_file('localidades.geojson')
bici = gpd.read_file('redbiciusuario.geojson')
stat_name = gpd.read_file('estacion_calida_aire.gpkg')
municipalities_loc = pd.read_excel('../municipalities_loc.xlsx')

# Specify the directory you want to list files from
directory = "../data_bogota2"
filename_lst = []
# List all files and directories in the specified directory
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        filename_lst.append(filename)

# Get the names of stations from the data files
station_name_lst = []
for file in filename_lst:
    a = pd.read_excel(directory + '/' + file)
    station_name = a['station_name'][0]
    station_name_lst.append(station_name)

# Dictionary with all the information
data_dic = {}
for name, file in zip(station_name_lst, filename_lst):
    # check if the name is in the name list with the location 
    if name in list(municipalities_loc['Name']):
        # read the data into a dataframe
        temp_data = pd.read_excel(directory + '/' + file)
        # reformat datetime
        temp_data['DateTime'] =  temp_data['DateTime'].astype(str)
        temp_data['DateTime'] = temp_data['DateTime'].apply(adjust_datetime)
        temp_data['DateTime'] = pd.to_datetime(temp_data['DateTime'], infer_datetime_format=True)
        # add columns of latitute and longitude and geometry
        temp_data['longitude'] = municipalities_loc['Longitud'][municipalities_loc['Name'] == name].iloc[0]
        temp_data['latitude'] = municipalities_loc['Latitud'][municipalities_loc['Name'] == name].iloc[0]
        temp_data['geometry'] = temp_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        temp_data.replace('----', np.nan, inplace=True)
        # Convert to GeoDataframe
        temp_geo_data = gpd.GeoDataFrame(temp_data, geometry='geometry')
        # add geodata to the dictionary 
        data_dic[name] = temp_geo_data


# merge municipalities so all of them have at least one value
mun_list = ['TEUSAQUILLO', 'BARRIOS UNIDOS']
mun_list2 = ['PUENTE ARANDA', 'ANTONIO NARIÑO', 'LOS MARTIRES', 'BOSA']
mun_list3 = ['TUNJUELITO', 'RAFAEL URIBE URIBE']
mun_list4 = ['CANDELARIA', 'SANTA FE', 'CHAPINERO']
mun_list5 = ['TEUSAQUILLO', 'BARRIOS UNIDOS', 'CHAPINERO', 'LOS MARTIRES', 'PUENTE ARANDA', 'ANTONIO NARIÑO' , 'TUNJUELITO', 'RAFAEL URIBE URIBE', 'CANDELARIA', 'SANTA FE', 'SUMAPAZ']

mun_set = municipalities[municipalities.LocNombre.isin(mun_list)]
mun_set2 = municipalities[municipalities.LocNombre.isin(mun_list2)]
mun_set3 = municipalities[municipalities.LocNombre.isin(mun_list3)]
mun_set4 = municipalities[municipalities.LocNombre.isin(mun_list4)]
mun_set5 = municipalities[~municipalities.LocNombre.isin(mun_list5)]

dissolved_filtered_gdf = mun_set.dissolve()
dissolved_filtered_gdf2 = mun_set2.dissolve()
dissolved_filtered_gdf3 = mun_set3.dissolve()
dissolved_filtered_gdf4 = mun_set4.dissolve()

municipalities2 = pd.concat([mun_set5, dissolved_filtered_gdf, dissolved_filtered_gdf2,\
    dissolved_filtered_gdf3, dissolved_filtered_gdf4])

from shapely.geometry import Polygon
# Define the coordinates for the square based on bounds
# Get the bounds
minx, maxx = municipalities2.bounds.minx.min(), municipalities2.bounds.maxx.max()
miny, maxy = 4.50, municipalities2.bounds.maxy.max()

coordinates = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]

# Create a Shapely Polygon from the coordinates
polygon = Polygon(coordinates)

# Create a GeoDataFrame
polygon_gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon])
polygon_gdf.set_crs(epsg=4686, inplace=True)

municipalities3  = gpd.overlay(municipalities2, polygon_gdf, how='intersection')

station_names = list(data_dic.keys())
# Specify the month you want 
selected_month = 2
def select_by_month(selected_month, station_names, data_dic):
    data_by_month = pd.DataFrame()
    for station in station_names:
        if station != 'Carvajal_Sevillana':
            # get the geo dataframe from station
            temp_d = data_dic.get(station)
            # filter the data by month
            temp_d = temp_d[temp_d['DateTime'].dt.month == selected_month]
            temp_d['DateTime'] = temp_d['DateTime'].dt.strftime("%Y-%m-%d")
            # select this specific columns 
            temp_d = temp_d[['DateTime', 'PM10', 'CO',
                'PM2.5','station_name', 'longitude',
                'latitude']]  #  'NO', 'NO2', 'NOX',
            #drop nan values 
            temp_d = temp_d.dropna()
            # group by mean value in the month
            group = temp_d.groupby('DateTime', as_index=False)[['PM10', 'CO', 'PM2.5']].mean()  #  'NO', 'NO2', 'NOX', into station_names[16] - Bolivia
            # merge data mean on the month with geospatial information
            temp_merged = group.merge(temp_d[['DateTime', 'station_name', 'longitude','latitude']].iloc[[0]], on = 'DateTime', how = 'left' )
            data_by_month = pd.concat([data_by_month, temp_merged])
            data_by_month.fillna(method='ffill', inplace=True)
  
            # print(count)
            # count +=1
            # print(station)

    # # transforme dataframe into geodataframe 
    # data_by_month = gpd.GeoDataFrame(data_by_month, geometry='geometry', )
    # data_by_month.set_crs(epsg=4686, inplace=True)

    return data_by_month

def select_monthly(selected_month, station_names, data_dic):
    data_by_month = pd.DataFrame()
    for station in station_names:
        if station != 'Carvajal_Sevillana':
            # get the geo dataframe from station
            temp_d = data_dic.get(station)
            # filter the data by month
            temp_d = temp_d[temp_d['DateTime'].dt.month == selected_month]
            temp_d['DateTime'] = temp_d['DateTime'].dt.strftime("%Y-%m")
            # select this specific columns 
            temp_d = temp_d[['DateTime', 'PM10', 'CO',
                'PM2.5','station_name', 'longitude',
                'latitude', 'geometry']]  #  'NO', 'NO2', 'NOX',
            #drop nan values 
            temp_d = temp_d.dropna()
            # group by mean value in the month
            group = temp_d.groupby('DateTime', as_index=False)[['PM10', 'CO', 'PM2.5']].mean()  #  'NO', 'NO2', 'NOX', into station_names[16] - Bolivia
            # merge data mean on the month with geospatial information
            temp_merged = group.merge(temp_d[['DateTime', 'station_name', 'longitude','latitude', 'geometry']].iloc[[0]], on = 'DateTime', how = 'inner' )
            data_by_month = pd.concat([data_by_month, temp_merged])
            # print(count)
            # count +=1
            # print(station)

    # transforme dataframe into geodataframe 
    data_by_month = gpd.GeoDataFrame(data_by_month, geometry='geometry', )
    data_by_month.set_crs(epsg=4686, inplace=True)

    return data_by_month

# Specify the hour you want 
selected_hour = 1
def select_by_hour(selected_hour, station_names, data_dic ):
    data_by_hour = pd.DataFrame()
    for station in station_names:
        if station != 'Carvajal_Sevillana':
            # get the geo dataframe from station
            temp_d = data_dic.get(station)
            # filter the data by hour
            temp_d = temp_d[temp_d['DateTime'].dt.hour == selected_hour]
            temp_d['DateTime'] = temp_d['DateTime'].dt.strftime( "%H:%M:%S")
            # select this specific columns 
            temp_d = temp_d[['DateTime', 'PM10', 'CO',
                'PM2.5','station_name', 'longitude',
                'latitude', 'geometry']]  #  'NO', 'NO2', 'NOX',
            #drop nan values 
            temp_d = temp_d.dropna()
            # group by mean value in the hour
            group = temp_d.groupby('DateTime', as_index=False)[['PM10', 'CO', 'PM2.5']].mean()  #  'NO', 'NO2', 'NOX', into station_names[16] - Bolivia
            # merge data mean on the hour with geospatial information
            temp_merged = group.merge(temp_d[['DateTime', 'station_name', 'longitude','latitude', 'geometry']].iloc[[0]], on = 'DateTime', how = 'inner' )
            data_by_hour = pd.concat([data_by_hour, temp_merged])
            # print(count)
            # count +=1
            # print(station)

    # transforme dataframe into geodataframe 
    data_by_hour = gpd.GeoDataFrame(data_by_hour, geometry='geometry', )
    data_by_hour.set_crs(epsg=4686, inplace=True)

    return data_by_hour

def plot_actual_data_daily(muni, selected_day = 1):
    # plot the values of PM2.5 in Bogota
    fig, ax = plt.subplots(1, figsize = (5, 5))
    label = 'PM2.5 Levels [$\mu g/m3$]'

    muni = muni[muni['DateTime'].dt.day == selected_day]

    # Plot values of PM2.5 for each municipality
    muni_plot = muni.plot(column='PM2.5', ax=ax, legend=False, cmap='Blues')
    # Plot the location of the stations
    #station_plot = data_by_day.plot(ax=ax, column='station_name', legend=True)
    # Create a colorbar for PM2.5
    norm = Normalize(vmin=muni['PM2.5'].min(), vmax=muni['PM2.5'].max())
    #norm = Normalize(vmin=12, vmax=35)
    sm = ScalarMappable(norm=norm, cmap='Blues')
    cbar = fig.colorbar(sm, ax=ax,fraction=0.046, pad=0.04)
    cbar.set_label(label)
    # Overlay bicycle network (roads)
    bici.plot(ax=ax, color='#5655AA', alpha=0.6, linestyle='-')
    day_title = str(muni['DateTime'].iloc[0])[:-8]
    plt.title(f"PM2.5 in Bogota during {day_title}", y = 1.02 )
    plt.suptitle("Choropleth map - Actual Data", x = 0.58)
    plt.ylabel('Latitude')
    plt.xlabel('Logitude')

    return plt.savefig('gif_make_daily_actual/' + str(muni['DateTime'].iloc[0])[:-8] + '_actual' + '.png')

def plot_actual_data(muni):
    # plot the values of PM2.5 in Bogota
    fig, ax = plt.subplots(1, figsize = (5, 5))
    label = 'PM2.5 Levels [$\mu g/m3$]'

    # Plot values of PM2.5 for each municipality
    muni_plot = muni.plot(column='PM2.5', ax=ax, legend=False, cmap='Blues')
    # Plot the location of the stations
    #station_plot = data_by_month.plot(ax=ax, column='station_name', legend=True)
    # Create a colorbar for PM2.5
    norm = Normalize(vmin=muni['PM2.5'].min(), vmax=muni['PM2.5'].max())
    sm = ScalarMappable(norm=norm, cmap='Blues')
    cbar = fig.colorbar(sm, ax=ax,fraction=0.046, pad=0.04,)
    cbar.set_label(label)
    # Overlay bicycle network (roads)
    bici.plot(ax=ax, color='#5655AA', alpha=0.6, linestyle='-')
    plt.title(f"PM2.5 in Bogota during {muni['DateTime'].iloc[0]}", y = 1.02 )
    plt.suptitle("Choropleth map - Actual Data", x = 0.58)
    plt.ylabel('Latitude')
    plt.xlabel('Logitude')

    return plt.savefig('gif_make_month_actual/' + str(muni['DateTime'].iloc[0]) + '_actual' + '.png')

def plot_actual_data_hourly(muni):
    # plot the values of PM2.5 in Bogota
    fig, ax = plt.subplots(1, figsize = (5, 5))
    label = 'PM2.5 Levels [$\mu g/m3$]'

    # Plot values of PM2.5 for each municipality
    muni_plot = muni.plot(column='PM2.5', ax=ax, legend=False, cmap='Blues')
    # Plot the location of the stations
    #station_plot = data_by_hour.plot(ax=ax, column='station_name', legend=True)
    # Create a colorbar for PM2.5
    norm = Normalize(vmin=muni['PM2.5'].min(), vmax=muni['PM2.5'].max())
    sm = ScalarMappable(norm=norm, cmap='Blues')
    cbar = fig.colorbar(sm, ax=ax,fraction=0.046, pad=0.04,)
    cbar.set_label(label)
    # Overlay bicycle network (roads)
    bici.plot(ax=ax, color='#5655AA', alpha=0.6, linestyle='-')
    plt.title(f"PM2.5 in Bogota during {muni['DateTime'].iloc[0]}", y = 1.02 )
    plt.suptitle("Choropleth map - Actual Data", x = 0.58)
    plt.ylabel('Latitude')
    plt.xlabel('Logitude')

    return plt.savefig('gif_make_hour_actual/' + str(muni['DateTime'].iloc[0]) + '_nearest' + '.png')

#function to create animation of geoplots
def create_animation(folder_name = 'gif_make_daily_actual', animation_name = 'actual_day_PM2.5.gif'):
    # Create animation from images
    # Specify the directory you want to list files from
    import imageio
    gif_images = []
    # Get the list of files and directories
    files_and_dirs = os.listdir(folder_name)

    # Sort the list alphabetically
    files_and_dirs.sort()
    # List all files and directories in the specified directory
    for im in files_and_dirs:
        if im.lower().endswith('.png'):
            gif_images.append(imageio.imread(f'{folder_name}/{im}'))
    imageio.mimsave(animation_name, gif_images, fps=1)

# function to estimate nearest interpolator by day
def estimate_by_nearest_interpolator(muni, daily_id = False, monthly_id = False, hourly_id = False):
    # Create a 100 by 100 grid
    # Horizontal and vertical cell counts should be the same
    resolution = 1000  # cell size in meters
    gridx = np.linspace(muni.bounds.minx.min(), muni.bounds.maxx.max(), resolution)
    gridy = np.linspace(muni.bounds.miny.min(), muni.bounds.maxy.max(), resolution)

    # Evaluate the method on grid
    nearest_model = NearestNDInterpolator(x = list(zip(muni["longitude"], muni["latitude"])),
                                y = muni["PM2.5"])
    PM25_nearest = nearest_model(*np.meshgrid(gridx, gridy))

    # Export raster
    export_kde_raster(Z = PM25_nearest, XX = gridx, YY = gridy,
                    min_x = muni.bounds.minx.min(),
                    max_x = muni.bounds.maxx.max(),
                    min_y = muni.bounds.miny.min(),
                    max_y = muni.bounds.maxy.max(),
                    proj = 4686, filename = "near_bogota_PM25.tif")

    # Open raster
    raster_nearPM25 = rasterio.open("near_bogota_PM25.tif")

    # Create polygon with extent of raster
    poly_shapely = shapely.box(*raster_nearPM25.bounds)

    # Create a dictionary with needed attributes and required geometry column
    attributes_df = {'Attribute': ['name1'], 'geometry': poly_shapely}

    # Convert shapely object to a GeoDataFrame
    raster_near_extent = gpd.GeoDataFrame(attributes_df, geometry = 'geometry', crs = 4686)

    # Mask raster to counties shape
    out_image_near, out_transform_near = rasterio.mask.mask(raster_nearPM25, muni.geometry.values, crop = True)

    from rasterio.plot import show
    # Plot data
    fig, ax = plt.subplots(1, figsize = (5, 5))
    img_show = show(out_image_near, ax = ax, transform = out_transform_near, cmap = "Blues", vmin = PM25_nearest.min(), vmax = PM25_nearest.max()) #hot_r, gist_heat_r, gist_ncar_r
    muni.plot(ax = ax, color = 'none', edgecolor = 'dimgray')
    plt.gca().invert_yaxis()
    # Add color bar
    cbar = fig.colorbar(img_show.get_images()[0], ax=ax, fraction=0.046, pad=0.04,)
    cbar.set_label('PM2.5 Levels [$\mu g/m3$]')

    if daily_id == True:
        name_fig = 'gif_make_day_near/' + str(muni['DateTime'].iloc[0])[:-8] + '_nearest' + '.png'
        title_fig = str(muni['DateTime'].iloc[0])[:-8]
    elif monthly_id == True:
        name_fig = 'gif_make_month/' + str(muni['DateTime'].iloc[0]) + '_nearest' + '.png'
        title_fig = str(muni['DateTime'].iloc[0]) 
    elif hourly_id == True:
        name_fig = 'gif_make_hour/' + str(muni['DateTime'].iloc[0]) + '_nearest' + '.png'
        title_fig = str(muni['DateTime'].iloc[0]) 

    plt.title(f"PM2.5 in Bogota during {title_fig}", y = 1.02 )
    plt.suptitle("Nearest Neighbour Estimation", x = 0.58)
    plt.ylabel('Latitude')
    plt.xlabel('Logitude')
    # Overlay bicycle network (roads)
    bici.plot(ax=ax, color='#5655AA', alpha=0.6, linestyle='-')

        
    return plt.savefig(name_fig)


def estimate_by_inv_distance(muni, daily_id = False, monthly_id = False, hourly_id = False):
    min_longitude = -74.25
    max_longitude = -73.95
    min_latitude = 4.5
    max_latitude = 4.85

    # Define the regular grid
    x_grid = np.arange(min_longitude, max_longitude, 0.001)
    y_grid = np.arange(min_latitude, max_latitude, 0.001)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Estimate the values on the grid using the inverse distance method
    Z = np.zeros_like(X)
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            Z[j, i] = inverse_distance_weighting(X[j, i], Y[j, i], muni)

    # Export raster
    export_kde_raster(Z = Z, XX = x_grid, YY = y_grid,
                    min_x = muni.bounds.minx.min(),
                    max_x = muni.bounds.maxx.max(),
                    min_y = muni.bounds.miny.min(),
                    max_y = muni.bounds.maxy.max(),
                    proj = 4686, filename = "inverse_bogota_PM25.tif")

    # Open raster
    raster_nearPM25 = rasterio.open("inverse_bogota_PM25.tif")

    # Create polygon with extent of raster
    poly_shapely = shapely.box(*raster_nearPM25.bounds)

    # Create a dictionary with needed attributes and required geometry column
    attributes_df = {'Attribute': ['name1'], 'geometry': poly_shapely}

    # Convert shapely object to a GeoDataFrame
    raster_near_extent = gpd.GeoDataFrame(attributes_df, geometry = 'geometry', crs = 4686)

    # Mask raster to counties shape
    out_image_near, out_transform_near = rasterio.mask.mask(raster_nearPM25, muni.geometry.values, crop = True)

    from rasterio.plot import show
    # Plot data
    fig, ax = plt.subplots(1, figsize = (5, 5))
    img_show = show(out_image_near, ax = ax, transform = out_transform_near, cmap = "Blues", vmin = Z.min(), vmax = Z.max()) #hot_r, gist_heat_r, gist_ncar_r
    muni.plot(ax = ax, color = 'none', edgecolor = 'dimgray')
    plt.gca().invert_yaxis()

    # Overlay bicycle network (roads)
    bici.plot(ax=ax, color='#5655AA', alpha=0.6, linestyle='-')

    # Add color bar
    cbar = fig.colorbar(img_show.get_images()[0], ax=ax, fraction=0.046, pad=0.04,)
    cbar.set_label('PM2.5 Levels [$\mu g/m3$]')
    plt.suptitle("Inverse distance Estimation", x = 0.59)

    if daily_id == True:
        name_fig = 'gif_make_inv_day/' + str(muni['DateTime'].iloc[0])[:-8] + '_inverse' + '.png'
        title_fig = str(muni['DateTime'].iloc[0])[:-8]
    elif monthly_id == True:
        name_fig = 'gif_make_inv_month/' + str(muni['DateTime'].iloc[0]) + '_inverse' + '.png'
        title_fig = str(muni['DateTime'].iloc[0]) 
    elif hourly_id == True:
        name_fig = 'gif_make_inv_hour/' + str(muni['DateTime'].iloc[0]) + '_inverse' + '.png'
        title_fig = str(muni['DateTime'].iloc[0]) 

    plt.title(f"PM2.5 in Bogota during {title_fig}", y = 1.02 )
    plt.ylabel('Latitude')
    plt.xlabel('Logitude')

    return plt.savefig(name_fig)