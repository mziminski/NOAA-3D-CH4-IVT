import os
import sys
import time
import json
import numpy as np
import numpy.ma as ma
import pandas as pd
import geopandas as gpd
import xarray as xr

from polys import to_coords

from dash.exceptions import PreventUpdate
from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

from datetime import datetime as dt

### Global Vars
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], update_title=None)

# Inspirations from: https://chart-studio.plotly.com/~Dreamshot/9152/import-plotly-plotly-version-/#/
def get_xy(_geo_data):
    xy = dict(x=[], y=[])

    for feature in _geo_data:
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


# Author: crgnam
# Source: https://community.plotly.com/t/applying-full-color-image-texture-to-create-an-interactive-earth-globe/60166
def texture2sphere(_texture, _size): 
    N_lat = int(_texture.shape[0])
    N_lon = int(_texture.shape[1])
    
    theta = np.linspace(0,2*np.pi,N_lat) - np.pi
    phi = np.linspace(0,np.pi,N_lon)
    
    # Set up coordinates for points on the sphere
    x0 = _size * np.outer(np.cos(theta),np.sin(phi))
    y0 = _size * np.outer(np.sin(theta),np.sin(phi))
    z0 = _size * np.outer(np.ones(N_lat),np.cos(phi))
    
    # Set up trace
    return x0,y0,z0

# Author: rbrundritt
# Source: https://rbrundritt.wordpress.com/2008/10/14/conversion-between-spherical-and-cartesian-coordinates-systems/
def cart2sph(_lon, _lat, size):
    '''
    Convert x,y,z cartesian coordinates to r,theta,phi spherical coordinates
    - phi is latitude (y)
    - theta is longitude (x)
    - rho is radius
    '''
   
    lonr, latr = np.deg2rad(_lon), np.deg2rad(_lat)

    x0 = size * np.cos(latr) * np.cos(lonr)
    y0 = size * np.cos(latr) * np.sin(lonr)
    z0 = size * np.sin(latr)

    return x0, y0, z0


# Author: rbrundritt
# Source: https://rbrundritt.wordpress.com/2008/10/14/conversion-between-spherical-and-cartesian-coordinates-systems/
def sphr2cart(_x, _y, _z):
    r = np.sqrt(_x**2 + _y**2 + _z**2)
    lat = np.rad2deg(np.arcsin(_z/r))
    lon = np.rad2deg(np.arctan2(_y, _x))
    return lon, lat



app.layout = html.Div([
    dcc.Store(id='data-store', data={}, storage_type='memory'),
    dcc.Store(id='basemap-store', data={}, storage_type='memory'),
    # dcc.Store(id='vol-fig-store', data={}, storage_type='session'),
    # dcc.Store(id='sphr-fig-store', data={}, storage_type='session'),
    html.H1(children='NOAA Carbon Tracker: 3D CH4 Mole Fraction (Pressure Corrected) Visualization', style={'textAlign':'center'}),
    html.Hr(),
    html.Div(className='container-fluid', children=[
        html.Div(className='row', children=[
            html.Div(className='col-2', children=[
                html.Div(children='Current Date:', style={'textAlign':'left'}),
                dcc.DatePickerSingle(
                    id='my-date-picker-single',
                    clearable=True,
                    reopen_calendar_on_clear=True,
                    min_date_allowed=dt(2021, 7, 1).date(),
                    max_date_allowed=dt(2021, 7, 31).date(),
                    initial_visible_month=dt(2021, 7, 1).date(),
                    date=dt(2021, 7, 1).date(),
                ),
            ]),
            html.Div(className='col-10', children=[
                html.Div(children='Current CH4 Presure Slice:', style={'textAlign':'center'}),
                dcc.Slider(0, 100, 10, # min, max, step
                        value=0,
                        id='my-slider',
                        marks={i: '{} hPa'.format(1000 - (i * 10)) for i in range(0, 101, 10)}
                ),
            ]),
        ]),
    ]),
    html.Hr(),
    html.Div(className='container-fluid bg-white', children=[
        html.Div(className='row', children=[
            html.Div(className='col-6', children=[
                dcc.Loading(
                    id='loading-1',
                    children=[dcc.Graph(id='vol-plt', figure={})],
                    type="circle",
                    style={'color':'#000000'}
                )
                
            ]),
            html.Div(className='col-6', id='gray', children=[
                dcc.Loading(
                    id='loading-2',
                    children=[dcc.Graph(id='sphr-plt', figure={})],
                    type='circle',
                )
                
            ]),
        ]),
    ]),
])


@app.callback(
    Output('data-store', 'data'),
    Input('my-date-picker-single', 'date'),
    State('data-store', 'data')
)
def load_data(_date, _data):
    # get data root directory path
    DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data') # if data files inside of dash-app root, remove 1 os.path.dirname call
    # get prior/posterior file paths
    data_fp = os.path.join(DATA_ROOT, '202107_101.nc')
    # read the data into datatree data structure
    ds = xr.open_dataset(data_fp, decode_times=False)
    # process the datatrees, and save results
    day = dt.strptime(_date, '%Y-%m-%d').day - 1
    _data = ds['mix'].isel(day=day)

    return json.dumps(_data.to_dict())


# output the stored clicks in the table cell.
@app.callback(
    Output('basemap-store', 'data'),
    # Since we use the data prop in an output,
    # we cannot get the initial data on load with the data prop.
    # To counter this, you can use the modified_timestamp
    # as Input and the data as State.
    # This limitation is due to the initial None callbacks
    # https://github.com/plotly/dash-renderer/pull/81
    Input('basemap-store', 'modified_timestamp'),
    State('basemap-store', 'data')
)
def load_basemap(_ts, _data):
    if _ts is None:
        raise PreventUpdate
    
    res = 'i'
    # get data root directory path
    DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data') # if data files inside of dash-app root, remove 1 os.path.dirname call
    # get no_antartica/antartica file paths
    no_antartica_fp = os.path.join(DATA_ROOT, f'GSHHS_shp/{res}/GSHHS_{res}_L1.shp')
    antartica_fp = os.path.join(DATA_ROOT, f'GSHHS_shp/{res}/GSHHS_{res}_L6.shp')
    no_antartica = gpd.read_file(no_antartica_fp)
    antartica =  gpd.read_file(antartica_fp)
    # concat the two, and remove small areas
    world = pd.concat((no_antartica,antartica))
    world = world[world['area'] > 1e3]
    # convert to xy coords
    _data = get_xy(world.geometry)

    return json.dumps(_data)
    

# Add controls to build the interaction
@callback(
    Output('vol-plt', 'figure'),
    Input('data-store', 'data'),
    Input('basemap-store', 'data'),
    Input('my-slider', 'value')
)
def gen_vol_plt(_data, _basemap, _level):

    # plot base figure
    fig = go.Figure()

    # plot volume
    _json = json.loads(_data)
    data = xr.DataArray.from_dict(_json).values

    lyr, lat, lon = data.shape

    # print(f'lyr: {lyr}, lat: {lat}, lon: {lon}') # lyr: 25, lat: 90, lon: 120
    Z, Y, X = np.mgrid[0:lyr-1:lyr-1*1j, -90:90:lat*1j, -180:180:lon*1j]

    fig.add_trace(go.Volume(
        name='',
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=data.flatten(),
        coloraxis='coloraxis',
        opacity=0.2, # needs to be small to see through all surfaces
        surface_count=12, # needs to be a large number for good volume rendering
        hovertemplate =
        '<b>lon</b>: %{x:,.3f}°<br>'+
        '<b>lat</b>: %{y:,.3f}°<br>'+
        '<b>Pressure</b>: %{text} hPa<br>'+
        '<b>Mix</b>: %{value:,.3f} ppb<br>',
        text=[f'{v:,.3f}' for v in (np.flip(Z) * 10).flatten('F')],
    ))

    # plot basemap
    _json = json.loads(_basemap)
    fig.add_trace(go.Scatter3d(
        name='',
        x=_json['x'], 
        y=_json['y'], 
        z=[0]*len(_json['x']), 
        mode='lines', 
        line=dict(color='white', width=3),
        hovertemplate='',
        text='',
    ))

    # plot green pane for testing
    fig.add_trace(go.Mesh3d(
        name='',
        x=[190,190,-190,-190],
        y=[100,-100,-100,100],
        z=[_level] * 4,
        delaunayaxis='z',
        color='white',
        opacity=0.5,
        hovertemplate=
        '<b>lon/lat slice-plane<b> @<br>'+
        '<b>Pressure</b>: %{text} hPa<br>',
        text=[f'{v:,.3f}' for v in (np.flip(Z) * 10).flatten('F')],
    ))

    # plot layout
    layout = go.Layout(
        title='Volumetric CH4 Mixing Ratio',
        scene=dict(
            xaxis=dict(ticks='', title='', showgrid=False, showline=False, 
                       zeroline=False, showbackground=False, showticklabels=False),
            yaxis=dict(ticks='', title='', showgrid=False, showline=False, 
                       zeroline=False, showbackground=False, showticklabels=False),
            zaxis=dict(ticks='', title='', showgrid=False, showline=False, 
                       zeroline=False, showbackground=False, showticklabels=False),
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=-2.0, z=1.25)
            ),
        ),
        # width=800,
        height=800,
        coloraxis=dict(
            colorscale='turbo',
            colorbar_title=dict(text='PPB', side= 'right'),
            cmin=0,
            # cmax=2600,
        )
    )

    fig.update_layout(layout)

    return fig


# Add controls to build the interaction
@callback(
    Output('sphr-plt', 'figure'),
    Input('data-store', 'data'),
    Input('basemap-store', 'data'),
    Input('my-slider', 'value')
)
def gen_sphr_plt(_data, _basemap, _pressure):
    '''
    earthRadius: radius of the earth in km
    '''

    # create base plot
    fig = go.Figure()

    # plot surface
    _json = json.loads(_data)
    texture = xr.DataArray.from_dict(_json).isel(pressure=_pressure).values.T[:,::-1]

    x,y,z = texture2sphere(texture, 1)  # 1 is the radius of the sphere

    fig.add_trace(go.Surface(
        name='',
        x=x, y=y, z=z,
        customdata=np.array([sphr2cart(x,y,z)]).T,
        surfacecolor=texture,
        coloraxis='coloraxis',
        hovertemplate =
        '<b>lon</b>: %{customdata[0]:,.3f}°<br>'+
        '<b>lat</b>: %{customdata[1]:,.3f}°<br>'+
        '<b>Mix</b>: %{surfacecolor:,.3f} ppb<br>',
    ))

    # plot basemap
    _json = json.loads(_basemap)
    lon, lat= ma.array(_json['x']), ma.array(_json['y'])

    bx, by, bz = cart2sph(lon, lat, 1)

    fig.add_trace(go.Scatter3d(
        name='',
        x=bx, y=by, z=bz, 
        mode='lines', 
        line=dict(color='white', width=3),
        hovertemplate='',
        text='',
    ))


    # plot layout
    p = np.arange(1000, -1, -10)[_pressure]
    # modified from @empet from https://chart-studio.plotly.com/~empet/14795/wind-speed-forecast/#/plot
    layout = go.Layout(
        title=f'Spherical CH4 Mixing Ratio @ {p} hPa Slice',
        scene=dict(
            xaxis=dict(ticks='', title='', showgrid=False, showline=False, 
                    zeroline=False, showbackground=False, showticklabels=False),
            yaxis=dict(ticks='', title='', showgrid=False, showline=False, 
                    zeroline=False, showbackground=False, showticklabels=False),
            zaxis=dict(ticks='', title='', showgrid=False, showline=False, 
                    zeroline=False, showbackground=False, showticklabels=False),
            aspectratio=dict(x=1, y=1, z=1),
            # camera = dict(
            #     up=dict(x=0, y=0, z=1),
            #     center=dict(x=0, y=0, z=0),
            #     eye=dict(x=-1.0, y=1.0, z=1.0)
            # ),
        ),
        # width=800,
        height=800,
         coloraxis=dict(
            colorscale='turbo',
            colorbar_title=dict(text='PPB', side= 'right'),
            cmin=0,
            # cmax=2600,
        )
    )

    fig.update_layout(layout)
    return fig



if __name__ == '__main__':
    # run dash app
    app.run_server(debug=False)