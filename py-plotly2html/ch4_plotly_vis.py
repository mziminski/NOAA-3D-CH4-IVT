
# # Basic Imports
from plotly2html.polys import to_coords # custom function to convert geojson to coords

import numpy as np
import numpy.ma as ma
import xarray as xr
import pandas as pd
import geopandas as gpd

import plotly.graph_objects as go
import plotly.io as pio


pio.templates.default = "plotly_dark" # set default template

# Inspirations from: https://chart-studio.plotly.com/~Dreamshot/9152/import-plotly-plotly-version-/#/
def get_xy(geo_data):
    xy = dict(x=[], y=[])

    for feature in geo_data:
        coords = to_coords(feature)
        for coord in coords:
            if coord == None:
                xy['x'].append(np.nan)
                xy['y'].append(np.nan)
                continue

            xy['x'].append(coord[0])
            xy['y'].append(coord[1])
        # break # only do one feature, for testing

    return xy
        
    
def create_volume_plot(data, basemap, filename=None):

    data_vals = data.values

    lyr, lat, lon = data_vals.shape

    # print(f'lyr: {lyr}, lat: {lat}, lon: {lon}') # lyr: 25, lat: 90, lon: 120

    Z, Y, X = np.mgrid[0:25:lyr*1j, -90:90:lat*1j, -180:180:lon*1j]

    # plot base figure
    fig = go.Figure()

    # plot volume
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=data_vals.flatten(),
        colorscale='turbo',   # choose a colorscale
        # colorbar_len=0.8,
        opacity=0.2, # needs to be small to see through all surfaces
        surface_count=25, # needs to be a large number for good volume rendering
    ))

    # plot green pane for testing
    # fig.add_trace({
    #     'type': 'mesh3d',        
    #     'x': [180,180,-180,-180],
    #     'y': [90,-90,-90,90],
    #     'z': [-1,-1,-1,-1], 
    #     'delaunayaxis':'z',
    #     'color': 'green',
    #     'opacity': 0.5,
    # })

    # plot basemap
    fig.add_trace(go.Scatter3d(
        x=basemap['x'], 
        y=basemap['y'], 
        z=[0]*len(basemap['x']), 
        mode='lines', 
        line=dict(color='white', width=3),
    ))

    layout = go.Layout(scene=dict(
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.0)
        ),
    ))


    fig.update_layout(layout)
    # fig.show() # keep commented out if on cluster

    if filename:
        fig.write_html(filename,  full_html=False, include_plotlyjs='cdn')



 # Author: crgnam
# Source: https://community.plotly.com/t/applying-full-color-image-texture-to-create-an-interactive-earth-globe/60166
def map_surf_2_sphere(texture, size): 
    N_lat = int(texture.shape[0])
    N_lon = int(texture.shape[1])
    
    theta = np.linspace(0,2*np.pi,N_lat) - np.pi
    phi = np.linspace(0,np.pi,N_lon)
    
    # Set up coordinates for points on the sphere
    x0 = size * np.outer(np.cos(theta),np.sin(phi))
    y0 = size * np.outer(np.sin(theta),np.sin(phi))
    z0 = size * np.outer(np.ones(N_lat),np.cos(phi))
    
    # Set up trace
    return x0, y0, z0


# Author: rbrundritt
# Source: https://rbrundritt.wordpress.com/2008/10/14/conversion-between-spherical-and-cartesian-coordinates-systems/
def cart2sph(_lat, _lon, size):
    '''
    Convert x,y,z cartesian coordinates to r,theta,phi spherical coordinates
    - phi is latitude (y)
    - theta is longitude (x)
    - rho is radius
    '''

    lonr, latr = np.deg2rad(_lon), np.deg2rad(_lat) # convert to radians

    x0 = size * np.cos(latr) * np.cos(lonr) 
    y0 = size * np.cos(latr) * np.sin(lonr)
    z0 = size * np.sin(latr)

    return x0, y0, z0



def create_sphere_plot(data, level, basemap, filename=None):
    '''
    earthRadius: radius of the earth in km
    '''

    texture = data.isel(levels=level).values.T[:,::-1]

    x,y,z = map_surf_2_sphere(texture, 1)  # 1 is the radius of the sphere

    lon, lat= ma.array(basemap['x']), ma.array(basemap['y'])


    bx, by, bz = cart2sph(lat, lon, 1)

    # create base plot
    fig = go.Figure()

    # modified from @empet from https://chart-studio.plotly.com/~empet/14795/wind-speed-forecast/#/plot
    layout = go.Layout(scene=dict(
        xaxis=dict(ticks='', title='', showgrid=False, showline=False, 
                zeroline=False, showbackground=False, showticklabels=False),
        yaxis=dict(ticks='', title='', showgrid=False, showline=False, 
                zeroline=False, showbackground=False, showticklabels=False),
        zaxis=dict(ticks='', title='', showgrid=False, showline=False, 
                zeroline=False, showbackground=False, showticklabels=False),
        aspectratio=dict(x=1, y=1, z=1),
        # camera=dict(projection=dict(type='orthographic')),
    ))

    # plot surface
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=texture,
        colorscale='turbo',
    ))

    # plot basemap
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz, 
        mode='lines', 
        line=dict(color='white', width=3),
    ))


    fig.update_layout(layout)
    # fig.show() # keep commented out if on cluster

    if filename:
        fig.write_html(filename, full_html=False, include_plotlyjs='cdn')


if __name__ == '__main__':

    # import coastline data
   
    # Shoreline data is from https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html
    # downloaded the gshhg-shp-2.3.7.zip file

    # resolution of coastline data, c, l, i, h, f
    # c = crude, l = low, i = intermediate, h = high, f = full
    res = 'i' 
    no_antartica = gpd.read_file(f'data/GSHHS_shp/{res}/GSHHS_{res}_L1.shp')
    antartica =  gpd.read_file(f'data/GSHHS_shp/{res}/GSHHS_{res}_L6.shp')

    world = pd.concat((no_antartica,antartica))
    world = world[world['area'] > 1e4]

    land = world.geometry
    base = get_xy(land)

    # import CH4 mole fraction data
    fp = './data/mole_frac/mix_20160501.nc4' 

    with xr.open_dataset(fp, group='glb300x200', decode_times=False) as ds:

        data = ds['mix'].isel(tracers=0).isel(times=0)

        # create plots
        create_volume_plot(data, base, "html/volume-plot.html")
        create_sphere_plot(data, 0, base, "html/sphere-plot.html") # 0 = level 0 