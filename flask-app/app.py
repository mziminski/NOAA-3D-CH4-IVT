import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import datatree as xt
import json

from plotly.utils import PlotlyJSONEncoder
import plotly.io as pio
import plotly.graph_objects as go

from shapely.geometry import Polygon
from flask import Flask, request, send_from_directory, jsonify, render_template
from werkzeug.utils import safe_join

from plotly2html.polys import to_coords

pio.templates.default = "plotly_dark"
### Global Vars

app = Flask(__name__)

mole_frac_mix = None
glb_base = None

### Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/volume', methods=['GET'])
def volume():
    # extract args from url query
    itime = request.args.get('itime', type=int)
    # get the plot
    json = create_volume_data(mole_frac_mix.isel(times=itime))
    return json

@app.route('/sphere', methods=['GET'])
def sphere():
    # extract args from url query
    itime = request.args.get('itime', type=int)
    level = request.args.get('level', type=int)
    # get the plot
    json = create_sphere_data(mole_frac_mix.isel(times=itime).isel(levels=level))
    return json

@app.route('/filter', methods=['GET'])
def filter():
    dt_filtered = None
    # extract args from url query
    itime = request.args.get('itime', type=int)
    layer = request.args.get('layer', type=int)
    # check which dataset to filter
    dt_filtered = mole_frac_mix.isel(times=itime).isel(levels=layer)
    # TODO: add others as need be
    return to_json(dt_filtered)


### Helper Functions
def load_mole_fracs():
    global mole_frac_mix
    # get data root directory path
    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    # get prior/posterior file paths
    mole_frac_fp = os.path.join(DATA_ROOT, 'mole_frac/mix_20160501.nc4')
    # read the data into datatree data structure
    mole_frac_dt = xt.open_datatree(mole_frac_fp, decode_times=False)
    # process the datatrees, and save results
    mole_frac_mix = process(mole_frac_dt)
    # print(">>> DEBUG --- loaded >> \n", mole_frac_mix, file=sys.stderr)

def load_glb_coastlines(res='i'):
    global glb_base
    # get data root directory path
    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    # get no_antartica/antartica file paths
    no_antartica = gpd.read_file(f'data/GSHHS_shp/{res}/GSHHS_{res}_L1.shp')
    antartica =  gpd.read_file(f'data/GSHHS_shp/{res}/GSHHS_{res}_L6.shp')
    # concat the two, and remove small areas
    world = pd.concat((no_antartica,antartica))
    world = world[world['area'] > 1e3]
    # convert to xy coords
    glb_base = get_xy(world.geometry)


def process(dt):
    dt_mix = dt['glb300x200/mix'].isel(tracers=0)

    # remove attributes
    dt_mix.attrs = ""

    return dt_mix
    
def to_json(dt):
    json_obj = {}

    to_dict = dt.to_dict()

    return jsonify(to_dict)


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

# Author: crgnam
# Source: https://community.plotly.com/t/applying-full-color-image-texture-to-create-an-interactive-earth-globe/60166
def texture2sphere(texture, size): 
    N_lat = int(texture.shape[0])
    N_lon = int(texture.shape[1])
    
    theta = np.linspace(0,2*np.pi,N_lat) - np.pi
    phi = np.linspace(0,np.pi,N_lon)
    
    # Set up coordinates for points on the sphere
    x0 = size * np.outer(np.cos(theta),np.sin(phi))
    y0 = size * np.outer(np.sin(theta),np.sin(phi))
    z0 = size * np.outer(np.ones(N_lat),np.cos(phi))
    
    # Set up trace
    return x0,y0,z0

# Author: rbrundritt
# Source: https://rbrundritt.wordpress.com/2008/10/14/conversion-between-spherical-and-cartesian-coordinates-systems/
def cart2sph(_lat, _lon, size):
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

   
def create_volume_data(ch4_mix=None):

    ch4_mix_vals = ch4_mix.values
    lyr, lat, lon = ch4_mix_vals.shape
    Z, Y, X = np.mgrid[0:25:lyr*1j, -90:90:lat*1j, -180:180:lon*1j]

    data = [
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=ch4_mix_vals.flatten(),
            colorscale='turbo',   # choose a colorscale
            # colorbar_len=0.8,
            opacity=0.2, # needs to be small to see through all surfaces
            surface_count=9, # needs to be a large number for good volume rendering
        ),
        go.Scatter3d(
            x=glb_base['x'], 
            y=glb_base['y'], 
            z=[0]*len(glb_base['x']), 
            mode='lines', 
            line=dict(color='white', width=3),
        )
    ]

    graphJSON = json.dumps(data, cls=PlotlyJSONEncoder)
    return graphJSON


def create_sphere_data(ch4_mix=None):

    texture = ch4_mix.values.T[:,::-1]

    earthRadius = 1; # radius in km
    x,y,z = texture2sphere(texture, earthRadius) 

    lon, lat= np.array(glb_base['x']), np.array(glb_base['y'])
    # lon, lat = lon[:np.where(np.isnan(llon))[0][0]], lat[:np.where(np.isnan(lat))[0][0]]

    bx, by, bz = cart2sph(lat, lon, 1)

    data = [
        go.Surface(
            x=x, y=y, z=z,
            surfacecolor=texture,
            colorscale='turbo',
        ),
        go.Scatter3d(
            x=bx, y=by, z=bz, 
            mode='lines', 
            line=dict(color='white', width=3),
        )
    ]

    graphJSON = json.dumps(data, cls=PlotlyJSONEncoder)
    return graphJSON



### MAIN block
if __name__ == '__main__':
    # print(">>> DEBUG --- ", WWW_ROOT, file=sys.stderr)
    load_mole_fracs()
    load_glb_coastlines()
    app.run(debug=True, host='127.0.0.1', port=8000)