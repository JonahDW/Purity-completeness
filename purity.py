import glob
import json
import ast
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, Column
from astropy.io import fits
import astropy.units as u

from argparse import ArgumentParser
import bdsf

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

plt.rcParams.update({'font.size': 14})

def invert_image(infile, outfile):
    image = fits.open(infile)
    image[0].data *= -1
    image.writeto(outfile, overwrite=True)

def run_bdsf(image, rms_map, mean_map):
    '''
    Run PyBDSF on an image

    Keyword arguments:
    image      -- Name of image
    rms_map    -- Name of rms image
    mean_map   -- Name of mean image
    '''
    imname = image.rsplit('.')[0]
    outcatalog = imname+'_bdsfcat.fits'

    path = Path(__file__).parent / 'parsets/bdsf_args.json'
    with open(path) as f:
        args_dict = json.load(f)

    # Fix json stupidness
    args_dict['process_image']['rms_box'] = ast.literal_eval(args_dict['process_image']['rms_box'])
    args_dict['process_image']['rms_box_bright'] = ast.literal_eval(args_dict['process_image']['rms_box_bright'])

    img = bdsf.process_image(image, rmsmean_map_filename=(mean_map,rms_map), **args_dict['process_image'])
    img.write_catalog(outfile = outcatalog, format='fits', **args_dict['write_catalog'])

    return outcatalog

def transform_cat(catalog, thresh_pix):
    '''
    Add distances from pointing center for analysis
    '''
    header = dict([x.split(' = ') for x in catalog.meta['comments'][5:]])
    catalog = catalog[catalog['Peak_flux']/catalog['Isl_rms'] > thresh_pix]

    pointing_center = SkyCoord(float(header['OBSRA'])*u.degree,
                               float(header['OBSDEC'])*u.degree)
    pointing_name = ['PT-'+header['OBJECT'].replace("'","")] * len(catalog)

    source_coord = SkyCoord([source['RA'] for source in catalog],
                            [source['DEC'] for source in catalog],
                            unit=(u.deg,u.deg))

    dra, ddec = pointing_center.spherical_offsets_to(source_coord)

    # Remove unnecessary columns
    catalog.remove_column('Source_id')
    catalog.remove_column('Isl_id')

    # Add columns at appropriate indices
    col_a = Column(pointing_name, name='Pointing_id')
    col_b = Column(dra, name='dRA')
    col_c = Column(ddec, name='dDEC')
    catalog.add_columns([col_a, col_b, col_c],
                         indexes=[0,2,4])

    return catalog

def plot_purity(inverse_catalog, full_catalog):
    '''
    Plot purity measures

    Keyword arguments:
    inverse_catalog -- Filename of catalog with source from inverted images
    full_catalog    -- Full catalog of sources from regular images
    '''
    output_dir = os.path.dirname(inverse_catalog)
    inverse = Table.read(inverse_catalog)
    full = Table.read(full_catalog)

    # Define S/N bins
    snr_bins = np.arange(4.5,7.6,0.1)
    snr_inverse = inverse['Peak_flux']/inverse['Isl_rms']
    snr_full = full['Peak_flux']/full['Isl_rms']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    n_inverse, _, _ = ax2.hist(snr_inverse, bins=snr_bins, edgecolor='k', facecolor='k', alpha=0.5)
    n_full, _, _ = ax2.hist(snr_full, bins=snr_bins, edgecolor='k', facecolor='none')

    ax1.plot((snr_bins[1:] + snr_bins[:-1]) / 2, n_inverse/n_full, color='r')
    ax1.set_xlabel('S/N')
    ax1.set_ylabel('Fraction')
    ax2.set_ylabel('Counts')
    plt.savefig(os.path.join(output_dir,'purity_snr.png'), dpi=300)
    plt.show()

    '''
    flux_bins = np.arange(7e-5,7e-4,1e-5)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    n_inverse, _, _ = ax2.hist(inverse['Total_flux'], bins=flux_bins, edgecolor='k', facecolor='k', alpha=0.5)
    n_full, _, _ = ax2.hist(full['Total_flux'], bins=flux_bins, edgecolor='k', facecolor='none')

    ax1.plot((flux_bins[1:] + flux_bins[:-1]) / 2, n_inverse/n_full, color='r')
    plt.savefig(os.path.join(output_dir,'purity_flux.png'), dpi=300)
    plt.show()
    '''

def main():
    parser = new_argument_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    full_catalog = args.full_catalog
    thresh_isl = args.thresh_isl
    thresh_pix = args.thresh_pix

    # Parse sourcefinding directories
    sf_dirs = sorted(glob.glob(os.path.join(input_dir,'*_pybdsf')))
    output_cat_file = os.path.join(input_dir,'inverse_sources.fits')

    if os.path.exists(output_cat_file):
        print('Reading table of inverse sources already present')
        full_inverse = Table.read(output_cat_file)
    else:
        full_inverse = Table()
        for sd in sf_dirs:
            source = os.path.basename(sd).split('_')[0]

            # Invert image
            im_in = os.path.join(os.path.dirname(sd), source+'.fits')
            im_out = os.path.join(sd, source+'_inverse.fits')
            invert_image(im_in, im_out)

            # Invert mean image for pybdsf
            mean_in = os.path.join(sd, source+'_mean.fits')
            mean_out = os.path.join(sd, source+'_inverse_mean.fits')
            invert_image(mean_in, mean_out)

            out_catalog = run_bdsf(im_out, source+'_rms.fits',
                                   source+'_inverse_mean.fits',
                                   thresh_isl, thresh_pix)

            try:
                inverse_catalog = Table.read(out_catalog)
                inverse_catalog = transform_cat(inverse_catalog)
            except FileNotFoundError:
                print('File not found, most likely no sources have been found in the image')

            # Clean up
            os.system(f'rm -f {out_catalog}')
            os.system(f'rm -f {im_out}')
            os.system(f'rm -f {mean_out}')
            os.system(f'rm -f {im_out}.pybdsf.log')

            full_inverse = vstack([full_inverse,inverse_catalog])
        full_inverse.write(output_cat_file, overwrite=True)

    plot_purity(output_cat_file, full_catalog)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("input_dir",
                        help="""Input directory containing pybdsf directories.""")
    parser.add_argument("full_catalog",
                        help="""Full catalog for comparison with inverse catalog.""")
    return parser

if __name__ == '__main__':
    main()