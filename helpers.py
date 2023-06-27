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

def random_ra_dec(ra_c, dec_c, radius, n, square=False):

    def sample_circle(N):
        '''
        Samples source in a circular image
        '''
        r = np.sqrt(np.random.uniform(size=N))
        theta = 2*np.pi*np.random.uniform(size=N)

        ra = ra_c + ra_radius * r * np.cos(theta)
        dec = dec_c + dec_radius * r * np.sin(theta)

        return ra, dec

    def sample_square(N):
        '''
        Sample sources in a square image
        '''
        ramin = ra_c - ra_radius
        ramax = ra_c + ra_radius
        decmin = dec_c - dec_radius
        decmax = dec_c + dec_radius

        ra_sample = np.random.uniform(np.radians(ramin), np.radians(ramax), N)
        p = np.random.uniform((np.sin(np.radians(decmin)) + 1) / 2, (np.sin(np.radians(decmax)) + 1) / 2, N)
        dec_sample = np.arcsin(2 * p - 1)

        return np.degrees(ra_sample), np.degrees(dec_sample)

    dec_radius = radius
    ra_radius = dec_radius/np.cos(np.deg2rad(dec_c))

    if square:
        ra, dec = sample_square(n)
    else:
        ra, dec = sample_circle(n)

    # Ensure proper boundaries
    ra[ra < 0.] += 360.
    ra[ra > 360.] -= 360

    return ra, dec

def slice_catalog(catalog, ra_c, dec_c, radius, square=False):

    def sample_circle(catalog, dra, ddec):
        dist = np.sqrt(dra**2 + ddec**2)

        dra = dra[dist < radius]
        ddec = ddec[dist < radius]
        catalog = catalog[dist < radius]

        return catalog, dra, ddec

    def sample_square(catalog, dra, ddec):
        ra_dist = np.abs(dra)
        dec_dist = np.abs(ddec)

        dra = dra[np.logical_and(ra_dist < radius, dec_dist < radius)]
        ddec = ddec[np.logical_and(ra_dist < radius, dec_dist < radius)]
        catalog = catalog[np.logical_and(ra_dist < radius, dec_dist < radius)]

        return catalog, dra, ddec

    ra = catalog['right_ascension_1']
    dec = catalog['declination_1']

    min_ra = ra.min() + radius
    min_dec = dec.min() + radius
    max_ra = ra.max() - radius
    max_dec = dec.max() - radius

    dra = ra - np.random.uniform(min_ra, max_ra, 1)
    ddec = dec - np.random.uniform(min_dec, max_dec, 1)

    if square:
        catalog, dra, ddec = sample_square(catalog, dra, ddec)
    else:
        catalog, dra, ddec = sample_circle(catalog, dra, ddec)

    catalog['right_ascension_1'] = dra + ra_c
    catalog['declination_1'] = ddec + dec_c

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