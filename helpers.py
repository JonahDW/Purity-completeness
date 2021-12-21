import numpy as np
import pickle

import astropy.units as u
from astropy.coordinates import SkyCoord

def pickle_to_file(data, fname):
    fh = open(fname, 'wb')
    pickle.dump(data, fh)
    fh.close()

def pickle_from_file(fname):
    fh = open(fname, 'rb')
    data = pickle.load(fh)
    fh.close()
    return data

def random_ra_dec(ra_c, dec_c, radius, n):

    def sample(N):
        ra_sample = np.random.uniform(np.radians(ramin), np.radians(ramax), N)
        p = np.random.uniform((np.sin(np.radians(decmin)) + 1) / 2, (np.sin(np.radians(decmax)) + 1) / 2, N)
        dec_sample = np.arcsin(2 * p - 1)

        return np.degrees(ra_sample), np.degrees(dec_sample)

    def accept(ra, dec):
        PA = 0
        bool_points = ((np.cos(PA)*(ra-ra_c)
                  +np.sin(PA)*(dec-dec_c))**2
                  /(ra_radius)**2
                  +(np.sin(PA)*(ra-ra_c)
                  -np.cos(PA)*(dec-dec_c))**2
                  /(dec_radius)**2) <= 1
        return bool_points

    dec_radius = radius
    ra_radius = dec_radius/np.cos(np.deg2rad(dec_c))

    ramin = ra_c - ra_radius
    ramax = ra_c + ra_radius
    decmin = dec_c - dec_radius
    decmax = dec_c + dec_radius

    # Sample until all spots are filled
    ra_all, dec_all = sample(n)
    mask = accept(ra_all, dec_all)
    reject = np.where(~mask)[0]

    ra_all = ra_all[mask]
    dec_all = dec_all[mask]
    while len(reject) > 0:
        fill_ra, fill_dec = sample(len(reject))
        mask = accept(fill_ra, fill_dec)
        reject = reject[~mask]

        ra_all = np.append(ra_all, fill_ra[mask])
        dec_all = np.append(dec_all, fill_dec[mask])

    #Just to make sure
    ra_all[ra_all < 0.] += 360.

    return ra_all, dec_all

def flag_artifacts(bright_sources, catalog):
    catalog_coord = SkyCoord(catalog['RA'], catalog['DEC'], unit='deg', frame='icrs')

    indices = []
    for source in bright_sources:
        source_coord = SkyCoord(source['RA'], source['DEC'], unit='deg', frame='icrs')

        d2d = source_coord.separation(catalog_coord)
        close = d2d < 5*source['Maj']*u.deg

        indices.append(np.where(np.logical_and(close, catalog['Total_flux'] < 0.1*source['Total_flux']))[0])

    indices = np.concatenate(indices)
    return indices