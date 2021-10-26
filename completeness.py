import glob
import json
import ast
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits, ascii
from astropy.wcs import WCS
import astropy.units as u

from astropy.convolution import convolve, Gaussian2DKernel
from astropy.modeling.models import Gaussian2D

from argparse import ArgumentParser
from pathlib import Path

import bdsf
import helpers

class Image:

    def __init__(self, image_file):
        self.image_file = image_file
        self.image_id = os.path.basename(image_file).split('_')[0]
        self.output_folder = os.path.dirname(image_file)

        self.image = fits.open(image_file)
        self.header = self.image[0].header

    def run_bdsf(self, input_image):
        '''
        Run PyBDSF on an image

        Keyword arguments:
        image   -- Name of image
        argfile -- Input json file containing arguments
                   for bdsf functions
        '''
        outcatalog = self.image_id+'_bdsfcat.fits'

        path = Path(__file__).parent / 'parsets/bdsf_args.json'
        with open(path) as f:
            args_dict = json.load(f)

        img = bdsf.process_image(input_image, rmsmean_map_filename=[self.image_id+'_mean.fits',self.image_id+'_rms.fits'],
                                 **args_dict['process_image'])
        img.write_catalog(outfile = outcatalog, format='fits', **args_dict['write_catalog'])

        return outcatalog

    def convolve_sources(self, catalog, flux_col):
        '''
        Convolve sources from input catalog with beam of specified image

        Keyword arguments:
        catalog   -- Input catalog containing sources
        flux_col  -- Desired column to extract flux from
        '''
        # Conversion factor from FWHM to std
        a = 2*np.sqrt(2*np.log(2))

        pix_scale = max(self.header['CDELT1'],self.header['CDELT2'])
        wcs = WCS(self.header, naxis=2)

        # Specify beam
        maj_std = self.header['BMAJ']/(a*pix_scale)
        min_std = self.header['BMIN']/(a*pix_scale)
        theta = np.deg2rad(self.header['BPA'])
        gauss_beam = Gaussian2DKernel(x_stddev = min_std,
                                      y_stddev = maj_std,
                                      theta=theta,
                                      x_size=51, y_size=51)
        gauss_beam_new = gauss_beam.array/gauss_beam.array.max()

        data_2d = np.zeros((self.header['NAXIS2'],self.header['NAXIS1']))
        for source in catalog:
            sky = SkyCoord(source['RA'], source['DEC'], unit=u.deg)
            X, Y = wcs.world_to_pixel(sky)

            # Check if point source and add accordingly
            if source['major_axis'] != 0.:
                source_maj_std = source['major_axis']/(a*pix_scale*3600)
                source_min_std = source['minor_axis']/(a*pix_scale*3600)
                source_theta = source['position_angle']
                gauss_source = Gaussian2D(x_stddev = source_min_std,
                                          y_stddev = source_maj_std,
                                          theta = source_theta)
                x = np.arange(-150, 151)
                y = np.arange(-150, 151)
                x, y = np.meshgrid(x, y)

                # Integrated flux should match
                flux_corr = np.sum(gauss_source(x,y))/10**source[flux_col]
                source = gauss_source(x,y)/flux_corr

                data_2d[int(Y)-150:int(Y)+151,int(X)-150:int(X)+151] += source
            else:
                data_2d[int(Y),int(X)] += 10**source[flux_col]

        data_2d[data_2d < 1e-10] = np.nan
        convolved_sources = convolve(data_2d, gauss_beam_new, 
                                     normalize_kernel=False, nan_treatment='fill')

        return convolved_sources

    def inject_sources(self, catalog, outfile, flux_col):
        '''
        Insert sources into an (empty) image from input catalog

        Keyword arguments:
        catalog    -- Catalog of sources to inject
        outfile    -- Filename of resulting image
        flux_col   -- Desired column to extract flux from
        '''
        new_image = fits.open(self.image_file)
        pixel_scale = max(self.header['CDELT1'],self.header['CDELT2'])
        imsize = min(new_image[0].data.shape[2:])

        # Generate random positions for sources
        radius = (imsize-400)*pixel_scale/2
        catalog['RA'], catalog['DEC'] = helpers.random_ra_dec(self.header['CRVAL1'],
                                                              self.header['CRVAL2'],
                                                              radius, 5000)

        image_data = new_image[0].data
        convolved_sources = self.convolve_sources(catalog, flux_col)
        image_data[0,0,:,:] += convolved_sources

        new_image[0].data = image_data
        new_image.writeto(outfile, overwrite=True)

        return catalog

    def match_completeness(self, catalog_in, catalog_out, flux_bins, flux_col):
        '''
        Measure completeness of detected sources by matching to the input catalog

        Keyword arguments:
        catalog_in  -- Input catalog
        catalog_out -- Output catalog generated by PyBDSF
        flux_bins   -- Flux bins
        flux_col    -- Flux column of input catalog
        '''
        match_dist = self.header['BMAJ']*3600 #arcsec

        c_in = SkyCoord(catalog_in['RA'], catalog_in['DEC'], unit=u.deg)
        c_out = SkyCoord(catalog_out['RA'], catalog_out['DEC'], unit=u.deg)

        idx, d2d, d3d = c_in.match_to_catalog_sky(c_out)
        sep_constraint = d2d < match_dist*u.arcsec
        catalog_in_matches = catalog_in[sep_constraint]
        catalog_out_matches = catalog_out[idx[sep_constraint]]

        n_in_matches, bins = np.histogram(10**catalog_in_matches[flux_col], bins=flux_bins)
        n_sources, bins = np.histogram(10**catalog_in[flux_col], bins=flux_bins)
        detected_fraction = n_in_matches/n_sources

        n_out_matches, bins = np.histogram(catalog_out_matches['Total_flux'], bins=flux_bins)
        flux_fraction = n_out_matches/n_sources

        return detected_fraction, flux_fraction

    def do_completeness(self, sim_cat, flux_col, flux_bins, n_sim, n_samples):
        '''
        Do one set of simulations on an image in order to measure the completeness

        Keyword arguments:
        sim_cat -- Catalog with simulated sources
        flux_col -- Flux column to use for simulated catalog
        flux_bins -- What flux bins to use
        n_sim -- Number of simulations to run
        '''
        detected_fraction = np.zeros((n_sim,len(flux_bins)-1))
        flux_fraction = np.zeros((n_sim,len(flux_bins)-1))
        for i in range(n_sim):
            # Draw random samples from flux catalog
            catalog_rows = np.random.choice(len(sim_cat), n_samples, replace=False)
            catalog_in = sim_cat[catalog_rows]

            # Inject sources and do sourcefinding
            out_image_file = os.path.join(self.output_folder,self.image_id+'_sim_'+str(i)+'.fits')
            catalog_in = self.inject_sources(catalog_in, out_image_file, flux_col)
            out_catalog_file = self.run_bdsf(out_image_file)

            catalog_out = Table.read(out_catalog_file)
            detected_fraction[i,:], flux_fraction[i,:] = self.match_completeness(catalog_in, catalog_out, 
                                                                                 flux_bins, flux_col)
            # Clean up
            os.system(f'rm -f {out_catalog_file}')
            os.system(f'rm -f {out_image_file}')
            os.system(f'rm -f {out_image_file}.pybdsf.log')

        return detected_fraction, flux_fraction

    def plot_results(self, flux_bins, detected_fraction_point, detected_fraction_ext):
        '''
        Plot the results from completeness measures

        Keyword arguments:
        flux_bins -- What flux bins to use
        detected_fraction_point -- Completeness fraction for point sources
        detected_fraction_ext -- Completeness fraction for extended sources
        '''
        # Point sources
        mean_fraction = np.mean(detected_fraction_point, axis=0)
        std_fraction = np.std(detected_fraction_point, axis=0)

        plt.plot(flux_bins[:-1], mean_fraction, color='k', label='Point sources')
        plt.fill_between(flux_bins[:-1],
                         mean_fraction-std_fraction,
                         mean_fraction+std_fraction,
                         color='k', alpha=0.5)

        # Extended sources
        mean_fraction = np.mean(detected_fraction_ext, axis=0)
        std_fraction = np.std(detected_fraction_ext, axis=0)

        plt.plot(flux_bins[:-1], mean_fraction, color='b', label='Extended sources')
        plt.fill_between(flux_bins[:-1],
                         mean_fraction-std_fraction,
                         mean_fraction+std_fraction,
                         color='b', alpha=0.5)

        plt.legend()
        plt.xscale('log')
        plt.xlabel('Flux density (Jy)')
        plt.ylabel('Completeness')

        plt.savefig(os.path.join(self.output_folder,self.image_id+'_completeness.png'), dpi=300)
        plt.close()

def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    image_file = args.image
    simulated_catalog = args.simulated_catalog
    flux_col = args.flux_col
    min_flux = args.min_flux
    max_flux = args.max_flux
    n_samples = args.n_samples
    n_flux_bins = args.flux_bins
    n_sim = args.n_sim

    # Define image object
    image = Image(image_file)

    # Get simulated catalog
    sim_cat = ascii.read(simulated_catalog)
    sim_flux = sim_cat[np.logical_and(sim_cat[flux_col] > min_flux, sim_cat[flux_col] < max_flux)]
    flux_bins = np.logspace(min_flux,max_flux,n_flux_bins)

    # Get completeness for point sources
    print('Measuring completeness for point sources')
    pickle = os.path.join(image.output_folder,'pickles',image.image_id+'_point_completeness.pkl')
    if os.path.exists(pickle):
        flux_bins, detected_fraction_point, flux_fraction_point = helpers.pickle_from_file(pickle)
    else:
        point_flux = sim_flux[sim_flux['major_axis'] == 0]
        detected_fraction_point, flux_fraction_point = image.do_completeness(point_flux, flux_col,
                                                                             flux_bins, n_sim, n_samples)

        if not os.path.exists(os.path.join(image.output_folder,'pickles')):
            os.mkdir(os.path.join(image.output_folder,'pickles'))
        helpers.pickle_to_file((flux_bins, detected_fraction_point, flux_fraction_point), fname=pickle)

    # Get completeness for extended sources
    print('Measuring completeness for extended sources')
    pickle = os.path.join(image.output_folder,'pickles',image.image_id+'_extended_completeness.pkl')
    if os.path.exists(pickle):
        flux_bins, detected_fraction_ext, flux_fraction_ext = helpers.pickle_from_file(pickle)
    else:
        extended_flux = sim_flux[sim_flux['major_axis'] > 0]
        detected_fraction_ext, flux_fraction_ext = image.do_completeness(extended_flux, flux_col,
                                                                         flux_bins, n_sim, n_samples)

        if not os.path.exists(os.path.join(image.output_folder,'pickles')):
            os.mkdir(os.path.join(image.output_folder,'pickles'))
        helpers.pickle_to_file((flux_bins, detected_fraction_ext, flux_fraction_ext), fname=pickle)

    image.plot_results(flux_bins, detected_fraction_point, detected_fraction_ext)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("image",
                        help="""Image to add sources to and measure completeness,
                                should contain no sources, only residuals.""")
    parser.add_argument("simulated_catalog",
                        help="""Name of catalog containing sources, flux to be drawn
                                from for input catalogs""")
    parser.add_argument("flux_col",
                        help="""Name of column in catalog containing log flux values""")
    parser.add_argument('--min_flux', default=-4.3,
                        help="""Log of minimum probed flux in Jansky.""")
    parser.add_argument('--max_flux', default=-2.3,
                        help="""Log of maximum probed flux in Jansky.""")
    parser.add_argument('--n_samples', default=5000,
                        help="""Number of sources to be drawn""")
    parser.add_argument('--flux_bins', default=30,
                        help="""Amount of flux bins""")
    parser.add_argument('--n_sim', default=10,
                        help="""Number of simulations to run""")

    return parser

if __name__ == '__main__':
    main()