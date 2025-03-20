import sys

import numpy as np
import pickle

import astropy.units as u
from astropy.table import unique
from astropy.io import fits
from astropy.coordinates import SkyCoord

from scipy.interpolate import interp1d

def pickle_to_file(data, fname):
    fh = open(fname, 'wb')
    pickle.dump(data, fh)
    fh.close()

def pickle_from_file(fname):
    fh = open(fname, 'rb')
    data = pickle.load(fh)
    fh.close()
    return data

def random_lonlat(lon_c, lat_c, radius, n, square=False):

    def sample_circle(N):
        '''
        Samples source in a circular image
        '''
        r = np.sqrt(np.random.uniform(size=N))
        theta = 2*np.pi*np.random.uniform(size=N)

        lon = lon_c + lon_radius * r * np.cos(theta)
        lat = lat_c + lat_radius * r * np.sin(theta)

        return lon, lat

    def sample_square(N):
        '''
        Sample sources in a square image
        '''
        lonmin = lon_c - lon_radius
        lonmax = lon_c + lon_radius
        latmin = lat_c - lat_radius
        latmax = lat_c + lat_radius

        lon_sample = np.random.uniform(np.deg2rad(lonmin), np.deg2rad(lonmax), N)
        p = np.random.uniform((np.sin(np.deg2rad(latmin)) + 1) / 2, (np.sin(np.deg2rad(latmax)) + 1) / 2, N)
        lat_sample = np.arcsin(2 * p - 1)

        return np.rad2deg(lon_sample), np.rad2deg(lat_sample)

    lat_radius = radius
    lon_radius = lat_radius/np.cos(np.deg2rad(lat_c))

    if square:
        lon, lat = sample_square(n)
    else:
        lon, lat = sample_circle(n)

    # Ensure proper boundaries
    lon[lon < 0.] += 360.
    lon[lon > 360.] -= 360

    return lon, lat

def slice_catalog(catalog, lon_c, lat_c, radius, square=False):

    def sample_circle(catalog, dlon, dlat):
        dist = np.sqrt(dlon**2 + dlat**2)

        dlon = dlon[dist < radius]
        dlat = dlat[dist < radius]
        catalog = catalog[dist < radius]

        return catalog, dlon, dlat

    def sample_square(catalog, dra, ddec):
        lon_dist = np.abs(dlon)
        lat_dist = np.abs(dlat)

        dlon = dlon[np.logical_and(lon_dist < radius, lat_dist < radius)]
        dlat = dlat[np.logical_and(lon_dist < radius, lat_dist < radius)]
        catalog = catalog[np.logical_and(lon_dist < radius, lat_dist < radius)]

        return catalog, dlon, dlat

    lon = catalog['lon']
    lat = catalog['lat']

    min_lon = lon.min() + radius
    min_lat = lat.min() + radius
    max_lon = lon.max() - radius
    max_lat = lat.max() - radius

    dlon = lon - np.random.uniform(min_lon, max_lon, 1)
    dlat = lat - np.random.uniform(min_lat, max_lat, 1)

    if square:
        catalog, dlon, dlat = sample_square(catalog, dlon, dlat)
    else:
        catalog, dlon, dlat = sample_circle(catalog, dlon, dlat)

    catalog['lon'] = dlon + lon_c
    catalog['lat'] = dlat + lat_c

    return catalog

def flag_artifacts(bright_sources, catalog):
    '''
    Identify artifacts near bright sources
    '''
    catalog_coord = SkyCoord(catalog['RA'], catalog['DEC'], unit='deg', frame='icrs')

    indices = []
    for source in bright_sources:
        source_coord = SkyCoord(source['RA'], source['DEC'], unit='deg', frame='icrs')

        d2d = source_coord.separation(catalog_coord)
        close = d2d < 5*source['Min']*u.deg

        indices.append(np.where(np.logical_and(close, catalog['Peak_flux'] < 0.1*source['Peak_flux']))[0])

    indices = np.concatenate(indices)
    return indices

def get_rms_coverage(rms_image):
    image = fits.open(rms_image)[0]
    rms_data = image.data.flatten()

    rms_range = np.logspace(np.log10(np.nanmin(rms_data)), np.log10(np.nanmax(rms_data)), 100)
    coverage = [np.sum([rms_data < rms])/np.count_nonzero(~np.isnan(rms_data)) for rms in rms_range]

    # Define a spline and interpolate the values
    data_spline = interp1d(rms_range, coverage, bounds_error=False, fill_value=(0,1))
    return data_spline, rms_range, coverage

def get_rms_radial(rms_image):
    image = fits.open(rms_image)[0]

    radialprofile = {}
    center = (image.header['CRPIX1'],image.header['CRPIX2'])
    data = np.squeeze(image.data)

    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    r_values, indices, inverse = np.unique(r.ravel(), return_index=True, return_inverse=True)
    rms = np.array([np.nanmedian(data.ravel()[inverse == r]) for r in r_values])

    r_values = r_values[~np.isnan(rms)]
    radialprofile['dist'] = r_values*max(image.header['CDELT1'],image.header['CDELT2'])
    radialprofile['rms'] = rms[~np.isnan(rms)]

    return radialprofile