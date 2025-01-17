import copy
import glob
import json
import ast
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from numpyencoder import NumpyEncoder

from astropy.coordinates import SkyCoord
from astropy.table import Table, join, unique
from astropy.io import fits, ascii
from astropy.wcs import WCS
import astropy.units as u

from astropy.convolution import convolve, Gaussian2DKernel
from astropy.modeling.models import Gaussian2D

from argparse import ArgumentParser
from pathlib import Path

import bdsf
import helpers

def save_load_results(func):
    def wrapper(self, *args, **kwargs):
        # Define output json file storing completeness
        outfile = self.image_id+f"_{kwargs['s_type']}_completeness.json"
        filename = os.path.join(self.output_folder, 
                                'completeness', outfile)

        # Check if data is there and open
        if os.path.exists(filename):
            with open(filename) as f:
                data = json.load(f)
        # Otherwise run function
        else:
            data = func(self, *args, **kwargs)
            # Save data to a json file
            with open(filename, 'w') as outfile:
                json.dump(data, outfile,
                          indent=4, sort_keys=True,
                          separators=(',', ': '),
                          ensure_ascii=False,
                          cls=NumpyEncoder)
        return data
    return wrapper

class Image:

    def __init__(self, image_name, outdir=None):
        self.image_id = os.path.basename(image_name)
        self.image_dir = os.path.dirname(image_name)
        self.image_file = image_name+'_gaus_resid.fits'

        if outdir is None:
            self.output_folder = self.image_dir
        else:
            self.output_folder = outdir

        if self.image_id == 'empty':
            path = Path(__file__).parent / 'parsets/empty_image.json'
            with open(path) as f:
                image_dict = json.load(f)

            self.header = image_dict['header']

            # Convert stuff to degrees
            self.header['CDELT1'] /= 3600
            self.header['CDELT2'] /= 3600
            self.header['BMAJ']   /= 3600
            self.header['BMIN']   /= 3600

            # Create image
            image_data = self.empty_noise_image(image_dict['properties']['RMS'])
            hdu = fits.PrimaryHDU(data=image_data, header=fits.Header(self.header))
            self.hdulist = fits.HDUList([hdu])

        else:
            self.hdulist = fits.open(self.image_file)
            self.header = self.hdulist[0].header

        # Initialize catalog of matches
        self.matched_catalog = Table()

    def empty_noise_image(self, rms):
        '''
        Make an empty image with uniform noise based on input image header

        Keyword arguments:
        rms -- RMS noise value
        '''
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

        # Initialize image
        image = np.random.normal(0, rms, size=(self.header['NAXIS2'],self.header['NAXIS1']))
        convolved_image = convolve(image, gauss_beam_new, 
                                   normalize_kernel=False, nan_treatment='fill')

        return convolved_image[np.newaxis,np.newaxis,:,:]

    def run_bdsf(self, input_image):
        '''
        Run PyBDSF on an image

        Keyword arguments:
        image   -- Name of image
        argfile -- Input json file containing arguments
                   for bdsf functions
        '''
        outcatalog = self.image_id+'_sim_bdsfcat.fits'

        path = Path(__file__).parent / 'parsets/bdsf_args.json'
        with open(path) as f:
            args_dict = json.load(f)

        # Make sure tuples are correctly parsed
        if 'rms_box' in args_dict['process_image']:
            if args_dict['process_image']['rms_box'] is not None:
                args_dict['process_image']['rms_box'] = ast.literal_eval(args_dict['process_image']['rms_box'])

        if self.image_id == 'empty':
            img = bdsf.process_image(input_image, **args_dict['process_image'])
        else:
            img = bdsf.process_image(input_image,
                                     advanced_opts=True,
                                     rmsmean_map_filename=[self.image_id+'_mean.fits',
                                                           self.image_id+'_rms.fits'],
                                     **args_dict['process_image'])

        # Uncomment this if you want to check what source finder is doing
        #img.show_fit()

        img.write_catalog(outfile = outcatalog, format='fits', **args_dict['write_catalog'])

        return outcatalog

    def convolve_sources(self, catalog, image):
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

        data_2d = np.zeros(np.squeeze(image).shape)
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
                flux_corr = np.sum(gauss_source(x,y))/10**source['flux']
                source = gauss_source(x,y)/flux_corr

                # Make sure the source is not out of bounds of the image
                minY = min(int(Y)-150, 0)
                minX = min(int(X)-150, 0)
                maxY = max(int(Y)+151-data_2d.shape[0], 0)
                maxX = max(int(X)+151-data_2d.shape[1], 0)

                data_2d[int(Y)-150-minY:int(Y)+151-maxY,
                        int(X)-150-minX:int(X)+151-maxX] += source[-minY:301-maxY,-minX:301-maxX]
            else:
                try: 
                    data_2d[int(Y),int(X)] += 10**source['flux']
                except IndexError:
                    print('Source was somehow out of bounds, it will be skipped!')
                    continue

        data_2d[data_2d < 1e-10] = np.nan
        convolved_sources = convolve(data_2d, gauss_beam_new, 
                                     normalize_kernel=False, nan_treatment='fill')

        return convolved_sources

    def inject_sources(self, catalog, outfile, realistic_counts, imsize, square):
        '''
        Insert sources into an (empty) image from input catalog

        Keyword arguments:
        catalog    -- Catalog of sources to inject
        outfile    -- Filename of resulting image
        flux_col   -- Desired column to extract flux from
        '''
        new_image = copy.deepcopy(self.hdulist)

        if imsize is None:
            pixel_scale = max(self.header['CDELT1'],self.header['CDELT2'])
            imlen = min(np.squeeze(new_image[0].data).shape)
            imsize = imlen*pixel_scale

        # Generate random positions for sources
        radius = imsize/2
        if realistic_counts:
            if self.image_id == 'empty':
                catalog = helpers.slice_catalog(catalog, self.header['CRVAL1'],
                                                self.header['CRVAL2'], radius, square=True)
            else:
                catalog = helpers.slice_catalog(catalog, self.header['CRVAL1'],
                                                self.header['CRVAL2'], radius, square=square)
            catalog.rename_column('right_ascension_1', 'RA')
            catalog.rename_column('declination_1', 'DEC')
        else:
            if self.image_id == 'empty':
                catalog['RA'], catalog['DEC'] = helpers.random_ra_dec(self.header['CRVAL1'],
                                                                      self.header['CRVAL2'], 
                                                                      radius, len(catalog),
                                                                      square=True)
            else:
                catalog['RA'], catalog['DEC'] = helpers.random_ra_dec(self.header['CRVAL1'],
                                                                      self.header['CRVAL2'],
                                                                      radius, len(catalog),
                                                                      square=square)

        image_data = new_image[0].data
        convolved_sources = self.convolve_sources(catalog, image_data)

        image_data[0,0,:,:] += convolved_sources

        new_image[0].data = image_data
        new_image.writeto(outfile, overwrite=True)

        return catalog

    def match_catalog(self, catalog_in, catalog_out):
        '''
        Match to the input catalog to catalog of recovered sources

        Keyword arguments:
        catalog_in  -- Input catalog
        catalog_out -- Output catalog generated by PyBDSF
        flux_bins   -- Flux bins
        flux_col    -- Flux column of input catalog
        '''
        match_dist = self.header['BMAJ']*3600 #arcsec

        c_in = SkyCoord(catalog_in['RA'], catalog_in['DEC'], unit=u.deg)
        c_out = SkyCoord(catalog_out['RA'], catalog_out['DEC'], unit=u.deg)

        idx, d2d, d3d = c_out.match_to_catalog_sky(c_in)
        sep_constraint = d2d < match_dist*u.arcsec

        catalog_in['idx']  = np.arange(len(catalog_in))
        catalog_out['idx'] = np.inf
        catalog_out['idx'][sep_constraint] = idx[sep_constraint]

        catalog_out_matches = catalog_out[sep_constraint]
        matched_catalog = join(catalog_in, catalog_out_matches, join_type='left', keys='idx')

        return matched_catalog

    def get_fractions(self, matched_catalog, flux_bins, n_sources):
        '''
        Get completeness fractions
        '''
        matched_sources = matched_catalog[matched_catalog['Source_id'] < n_sources]

        n_total_sources, _ = np.histogram(10**matched_catalog['flux'], bins=flux_bins)
        n_in_matches, _ = np.histogram(10**matched_sources['flux'], bins=flux_bins)
        detected_fraction = n_in_matches/n_total_sources

        n_out_matches, _ = np.histogram(matched_sources['Total_flux'], bins=flux_bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            flux_fraction = n_out_matches/n_total_sources

        return detected_fraction, flux_fraction

    @save_load_results
    def do_completeness(self, flux_bins, n_sim, n_sources, sim_cat, realistic_counts,
                        imsize, square, no_delete, s_type):
        '''
        Do one set of simulations on an image in order to measure the completeness

        Keyword arguments:
        sim_cat -- Catalog with simulated sources
        flux_col -- Flux column to use for simulated catalog
        flux_bins -- What flux bins to use
        n_sim -- Number of simulations to run
        s_type -- Source type (extended or point)
        '''
        detected_fraction = np.zeros((n_sim,len(flux_bins)-1))
        flux_fraction = np.zeros((n_sim,len(flux_bins)-1))
        for i in range(n_sim):
            print(f'Running simulation {i}')
            matched_cat_file = os.path.join(self.output_folder,'completeness',
                                            f'{self.image_id}_sim_{i}_{s_type}_catalog.fits')

            # Draw sample of sources and flux densities
            if os.path.exists(matched_cat_file):
                matched_cat = Table.read(matched_cat_file)
            else:
                if realistic_counts:
                    catalog_in = sim_cat
                elif s_type == 'point':
                    fluxes = np.random.uniform(np.log10(flux_bins[0]),
                                               np.log10(flux_bins[-1]),
                                               n_sources)
                    catalog_in = Table()
                    catalog_in['flux'] = fluxes
                    catalog_in['major_axis'] = 0
                    catalog_in['minor_axis'] = 0
                else:
                    # Draw from simulated catalogue and equal bins
                    digitized_flux = np.digitize(10**sim_cat['flux'], flux_bins)
                    catalog_rows = []
                    for j in range(1,len(flux_bins)):
                        samples_bin = int(n_sources/(len(flux_bins)-1))
                        try:
                            choice = np.random.choice(np.where(digitized_flux == j)[0], 
                                                      samples_bin, replace=False)
                        except ValueError:
                            print('Not enough sources in bin, selecting with replacement')
                            choice = np.random.choice(np.where(digitized_flux == j)[0], 
                                                      samples_bin, replace=True)
                        catalog_rows.append(choice)

                    catalog_rows = np.concatenate(catalog_rows, axis=None)
                    catalog_in = sim_cat[catalog_rows]

                # Inject sources and do sourcefinding
                out_image_file = os.path.join(self.image_dir,f'{self.image_id}_sim_{i}.fits')
                catalog_in = self.inject_sources(catalog_in, out_image_file, 
                                                 realistic_counts, imsize, square)
                out_catalog_file = self.run_bdsf(out_image_file)

                catalog_out = Table.read(out_catalog_file)
                matched_cat = self.match_catalog(catalog_in, catalog_out)
                matched_cat.write(matched_cat_file, overwrite=True)

                # Clean up
                os.system(f'rm -f {out_catalog_file}')
                if not no_delete:
                    os.system(f'rm -f {out_image_file}')
                    os.system(f'rm -f {out_image_file}.pybdsf.log')

            detected_fraction[i,:], flux_fraction[i,:] = self.get_fractions(matched_cat, flux_bins, n_sources)

        completeness = {'flux_bins': flux_bins,
                        'detected_fraction': detected_fraction,
                        'flux_fraction': flux_fraction}

        return completeness

    def plot_results(self, completeness, s_type):
        '''
        Plot basic results from completeness measures

        Keyword arguments:
        flux_bins -- What flux bins to use
        detected_fraction_point -- Completeness fraction for point sources
        detected_fraction_ext -- Completeness fraction for extended sources
        '''

        # Plot recovered fraction
        mean_fraction = np.mean(completeness['detected_fraction'], axis=0)
        std_fraction = np.std(completeness['detected_fraction'], axis=0)

        plt.plot(completeness['flux_bins'][:-1], mean_fraction, 
                 label='Detection completeness', color='k')
        plt.fill_between(completeness['flux_bins'][:-1],
                         mean_fraction-std_fraction,
                         mean_fraction+std_fraction,
                         color='k', alpha=0.5)

        # Plot flux recovery
        mean_fraction = np.mean(completeness['flux_fraction'], axis=0)
        std_fraction = np.std(completeness['flux_fraction'], axis=0)

        plt.plot(completeness['flux_bins'][:-1], mean_fraction, 
                 label='Flux completeness', color='crimson')
        plt.fill_between(completeness['flux_bins'][:-1],
                         mean_fraction-std_fraction,
                         mean_fraction+std_fraction,
                         color='crimson', alpha=0.5)

        plt.axhline(1,0,1, color='k', ls=':')

        plt.legend()
        plt.xscale('log')
        plt.xlabel('Flux density (Jy)')
        plt.ylabel('Completeness')

        outfile = os.path.join(self.output_folder,
                               'completeness',
                               f'{self.image_id}_{s_type}_completeness.png')
        plt.savefig(outfile, dpi=300)
        plt.close()

def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    image_name = args.image_name
    sources = args.sources

    # Simulation options
    realistic_counts = args.realistic_counts
    n_sources = args.n_sources
    n_flux_bins = args.flux_bins
    n_sim = args.n_sim
    min_flux = args.min_flux
    max_flux = args.max_flux

    # Catalog options
    sim_cat = args.sim_cat
    flux_col = args.flux_col

    # Image options
    imsize = args.imsize
    square = args.square

    # Output options
    outdir = args.outdir
    no_delete = args.no_delete

    # Define image object
    image = Image(image_name, outdir)

    flux_bins = np.logspace(min_flux,max_flux,n_flux_bins)

    # Get simulated catalog, or otherwise produce a uniform flux density distribution
    if sim_cat:
        sim_cat = ascii.read(simulated_catalog)
        sim_cat.rename_column(flux_col, 'flux')
        sim_flux = sim_cat[np.logical_and(sim_cat['flux'] > min_flux, sim_cat['flux'] < max_flux)]

    if not os.path.exists(os.path.join(image.output_folder, 'completeness')):
        os.mkdir(os.path.join(image.output_folder, 'completeness'))

    if sources == 'point' and realistic_counts:
        sim_cat = sim_flux[sim_flux['major_axis'] == 0]

    if sources == 'extended':
        sim_cat = sim_flux[np.logical_and(sim_flux['major_axis'] > 0,sim_flux['minor_axis'] > 0)]

    if sources == 'realistic':
        sim_cat = sim_flux[np.logical_xor(sim_flux['major_axis'] > 0,sim_flux['minor_axis'] == 0.)]

    completeness = image.do_completeness(flux_bins, n_sim, n_sources,
                                         sim_cat, realistic_counts,
                                         imsize, square, no_delete, 
                                         s_type=sources)

    image.plot_results(completeness, s_type=sources)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("image_name",
                        help="""Image to measure completeness on, without any extension.
                                PyBDSF file structure is assumed for the residual 
                                ('image_name_gaus_resid.fits'), rms ('image_name_rms.fits'),
                                and mean ('image_name_mean.fits').
                                An empty image containing only noise can also be created
                                from scratch by typing 'empty' instead of a filename. 
                                In this case the header and other properties 
                                will be read from parsets/empty_image.json""")
    parser.add_argument("sources",
                        help="""Type of sources to use for simulations. Options are:
                                'point': Only injects point sources
                                'extended': Only injects sources with a nonzero size
                                'realistic': Inject sources of all sizes, with realistic
                                size distribution. If nonzero source sizes are used in
                                any way, sources must be drawn from specified
                                simulated catalogue.""")
    parser.add_argument("--realistic_counts", action="store_true",
                        help="""Use a realistic flux density and spatial distribution 
                                instead of uniform per bin and in space. Simulated 
                                catalogue must be specified to draw from. (default = False)""")
    parser.add_argument("--flux_bins", default=30, type=int,
                        help="Amount of flux bins")
    parser.add_argument("--n_sim", default=10, type=int,
                        help="Number of simulations to run")
    parser.add_argument("--sim_cat", default=None, type=str,
                        help="""Name of catalog containing sources, flux to be drawn
                                from for input catalogs""")
    parser.add_argument("--flux_col", default=None, type=str,
                        help="Name of column in catalog containing log flux values")
    parser.add_argument("--min_flux", default=-5, type=float,
                        help="Log of minimum probed flux in Jansky.")
    parser.add_argument("--max_flux", default=-2, type=float,
                        help="Log of maximum probed flux in Jansky.")
    parser.add_argument("--n_sources", default=5000, type=int,
                        help="Number of sources to be drawn")
    parser.add_argument("--imsize", default=None, type=float,
                        help="""Specify size of the image in degrees for generating 
                                source positions. If the image is circular this is the 
                                diameter. By default this is read from the image header.""")
    parser.add_argument("--square", action="store_true",
                        help="Generate sources positions in a square rather than circle.")
    parser.add_argument("--outdir", default=None, type=str,
                        help="""Directory to store output, by default output is stored
                              in the directory of the input image""")
    parser.add_argument("--no_delete", action="store_true",
                        help="Do not delete pybdsf logs and images")

    return parser

if __name__ == '__main__':
    main()