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

class Image:

    def __init__(self, image_file, outdir=None):
        self.image_file = image_file
        self.image_id = os.path.basename(image_file).split('_')[0]
        self.image_dir = os.path.dirname(image_file)

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
            self.header['BMAJ'] /= 3600
            self.header['BMIN'] /= 3600

            # Create image
            image_data = self.empty_noise_image(image_dict['properties']['RMS'])
            hdu = fits.PrimaryHDU(data=image_data, header=fits.Header(self.header))
            self.hdulist = fits.HDUList([hdu])

        else:
            self.hdulist = fits.open(image_file)
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

        if self.image_id == 'empty':
            img = bdsf.process_image(input_image, **args_dict['process_image'])
        else:
            img = bdsf.process_image(input_image, rmsmean_map_filename=[self.image_id+'_mean.fits',self.image_id+'_rms.fits'],
                                     **args_dict['process_image'])

        #img.show_fit()
        img.write_catalog(outfile = outcatalog, format='fits', **args_dict['write_catalog'])

        return outcatalog

    def convolve_sources(self, catalog, flux_col, image):
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
                flux_corr = np.sum(gauss_source(x,y))/10**source[flux_col]
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
                    data_2d[int(Y),int(X)] += 10**source[flux_col]
                except IndexError:
                    print('Source was somehow out of bounds, it will be skipped!')
                    continue

        data_2d[data_2d < 1e-10] = np.nan
        convolved_sources = convolve(data_2d, gauss_beam_new, 
                                     normalize_kernel=False, nan_treatment='fill')

        return convolved_sources

    def inject_sources(self, catalog, outfile, flux_col, n_samples, orig_counts, imsize, square):
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
        if orig_counts:
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
        convolved_sources = self.convolve_sources(catalog, flux_col, image_data)

        image_data[0,0,:,:] += convolved_sources

        new_image[0].data = image_data
        new_image.writeto(outfile, overwrite=True)

        return catalog

    def match_completeness(self, catalog_in, catalog_out):
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

        catalog_out['idx'] = np.arange(len(catalog_out))
        catalog_in['idx'] = np.inf
        catalog_in['idx'][sep_constraint] = idx[sep_constraint]

        catalog_out_matches = catalog_out[np.unique(idx[sep_constraint])]
        matched_catalog = join(catalog_in, catalog_out_matches, join_type='left', keys='idx')

        return matched_catalog

    def get_fractions(self, matched_catalog, flux_bins, flux_col, n_samples):
        '''
        Get completeness fractions
        '''
        matched_sources = matched_catalog[matched_catalog['idx'] < n_samples]

        n_sources, _ = np.histogram(10**matched_catalog[flux_col], bins=flux_bins)
        n_in_matches, _ = np.histogram(10**matched_sources[flux_col], bins=flux_bins)
        detected_fraction = n_in_matches/n_sources

        n_out_matches, _ = np.histogram(matched_sources['Total_flux'], bins=flux_bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            flux_fraction = n_out_matches/n_sources

        return detected_fraction, flux_fraction

    def do_completeness(self, sim_cat, flux_col, flux_bins, n_sim, n_samples, imsize, square, no_delete, s_type):
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
            print(i)
            matched_catalog_file = os.path.join(self.output_folder,'completeness',f'{self.image_id}_sim_{i}_{s_type}_catalog.fits')

            if os.path.exists(matched_catalog_file):
                matched_catalog = Table.read(matched_catalog_file)
            else:
                # Draw random samples from flux catalog
                if s_type == 'orig':
                    orig_counts = True
                    catalog_in = sim_cat
                else:
                    orig_counts = False

                    # Draw from bins equally
                    digitized_flux = np.digitize(10**sim_cat[flux_col], flux_bins)
                    catalog_rows = []
                    for j in range(1,len(flux_bins)):
                        samples_bin = int(n_samples/(len(flux_bins)-1))
                        try:
                            choice = np.random.choice(np.where(digitized_flux == j)[0], samples_bin, replace=False)
                        except ValueError:
                            print('Not enough sources in bin, selecting with replacement')
                            choice = np.random.choice(np.where(digitized_flux == j)[0], samples_bin, replace=True)
                        catalog_rows.append(choice)

                    catalog_rows = np.concatenate(catalog_rows, axis=None)
                    catalog_in = sim_cat[catalog_rows]

                # Inject sources and do sourcefinding
                out_image_file = os.path.join(self.image_dir,self.image_id+'_sim_'+str(i)+'.fits')
                catalog_in = self.inject_sources(catalog_in, out_image_file, flux_col, n_samples, orig_counts, imsize, square)
                out_catalog_file = self.run_bdsf(out_image_file)

                catalog_out = Table.read(out_catalog_file)
                matched_catalog = self.match_completeness(catalog_in, catalog_out)
                matched_catalog.write(matched_catalog_file, overwrite=True)

                # Clean up
                os.system(f'rm -f {out_catalog_file}')
                if not no_delete:
                    os.system(f'rm -f {out_image_file}')
                    os.system(f'rm -f {out_image_file}.pybdsf.log')

            detected_fraction[i,:], flux_fraction[i,:] = self.get_fractions(matched_catalog, flux_bins, flux_col, n_samples)

        print(detected_fraction)
        return detected_fraction, flux_fraction

    def plot_results(self, flux_bins, detected_fraction_point=None, detected_fraction_ext=None, detected_fraction_orig=None):
        '''
        Plot the results from completeness measures

        Keyword arguments:
        flux_bins -- What flux bins to use
        detected_fraction_point -- Completeness fraction for point sources
        detected_fraction_ext -- Completeness fraction for extended sources
        '''

        if detected_fraction_point is not None:
            # Point sources
            mean_fraction = np.mean(detected_fraction_point, axis=0)
            std_fraction = np.std(detected_fraction_point, axis=0)

            plt.plot(flux_bins[:-1], mean_fraction, color='k', label='Point sources')
            plt.fill_between(flux_bins[:-1],
                             mean_fraction-std_fraction,
                             mean_fraction+std_fraction,
                             color='k', alpha=0.5)

        if detected_fraction_ext is not None:
            # Extended sources
            mean_fraction = np.mean(detected_fraction_ext, axis=0)
            std_fraction = np.std(detected_fraction_ext, axis=0)

            plt.plot(flux_bins[:-1], mean_fraction, color='b', label='Extended sources')
            plt.fill_between(flux_bins[:-1],
                             mean_fraction-std_fraction,
                             mean_fraction+std_fraction,
                             color='b', alpha=0.5)

        if detected_fraction_orig is not None:
            # Original number counts
            mean_fraction = np.mean(detected_fraction_orig, axis=0)
            std_fraction = np.std(detected_fraction_orig, axis=0)

            plt.plot(flux_bins[:-1], mean_fraction, color='b', label='SKADS sources')
            plt.fill_between(flux_bins[:-1],
                             mean_fraction-std_fraction,
                             mean_fraction+std_fraction,
                             color='b', alpha=0.5)

        plt.legend()
        plt.xscale('log')
        plt.xlabel('Flux density (Jy)')
        plt.ylabel('Completeness')

        plt.savefig(os.path.join(self.output_folder,'completeness',self.image_id+'_completeness.png'), dpi=300)
        plt.close()

def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    image_file = args.image
    simulated_catalog = args.simulated_catalog
    flux_col = args.flux_col
    outdir = args.outdir
    min_flux = args.min_flux
    max_flux = args.max_flux
    n_samples = args.n_samples
    n_flux_bins = args.flux_bins
    n_sim = args.n_sim
    orig_counts = args.orig_counts
    imsize = args.imsize
    square = args.square
    no_delete = args.no_delete

    # Define image object
    image = Image(image_file, outdir)

    # Get simulated catalog
    sim_cat = ascii.read(simulated_catalog)
    sim_flux = sim_cat[np.logical_and(sim_cat[flux_col] > min_flux, sim_cat[flux_col] < max_flux)]
    flux_bins = np.logspace(min_flux,max_flux,n_flux_bins)

    if not os.path.exists(os.path.join(image.output_folder, 'completeness')):
        os.mkdir(os.path.join(image.output_folder, 'completeness'))

    if orig_counts:
        print('Measuring completeness using SKADS number counts')
        data_file = os.path.join(image.output_folder, 'completeness', image.image_id+'_completeness.json')
        if os.path.exists(data_file):
            with open(data_file) as f:
                completeness = json.load(f)

        else:
            sim_cat = sim_flux[np.logical_xor(sim_flux['major_axis'] > 0,sim_flux['minor_axis'] == 0.)]
            detected_fraction, flux_fraction = image.do_completeness(sim_cat, flux_col,
                                                                     flux_bins, n_sim, n_samples,
                                                                     imsize, square,
                                                                     no_delete, s_type='orig')

            completeness = {'flux_bins': flux_bins,
                            'detected_fraction': detected_fraction,
                            'flux_fraction': flux_fraction}

            # Save data to a json file
            with open(data_file, 'w') as outfile:
                json.dump(completeness, outfile,
                          indent=4, sort_keys=True,
                          separators=(',', ': '),
                          ensure_ascii=False,
                          cls=NumpyEncoder)

        image.plot_results(flux_bins, detected_fraction_orig=detected_fraction)

    else:
        # Get completeness for point sources
        print('Measuring completeness for point sources')
        data_file = os.path.join(image.output_folder, 'completeness',image.image_id+'_point_completeness.json')
        if os.path.exists(data_file):
            with open(data_file) as f:
                point_completeness = json.load(f)
        else:
            point_flux = sim_flux[sim_flux['major_axis'] == 0]
            detected_fraction_point, flux_fraction_point = image.do_completeness(point_flux, flux_col,
                                                                                 flux_bins, n_sim, n_samples,
                                                                                 imsize, square,
                                                                                 no_delete, s_type='point')
            point_completeness = {'flux_bins': flux_bins,
                                  'detected_fraction': detected_fraction_point,
                                  'flux_fraction': flux_fraction_point}

            # Save data to a json file
            with open(data_file, 'w') as outfile:
                json.dump(point_completeness, outfile,
                          indent=4, sort_keys=True,
                          separators=(',', ': '),
                          ensure_ascii=False,
                          cls=NumpyEncoder)

        # Get completeness for extended sources
        print('Measuring completeness for extended sources')
        data_file = os.path.join(image.output_folder, 'completeness', image.image_id+'_extended_completeness.json')
        if os.path.exists(data_file):
            with open(data_file) as f:
                ext_completeness = json.load(f)
        else:
            extended_flux = sim_flux[np.logical_and(sim_flux['major_axis'] > 0,sim_flux['minor_axis'] > 0)]
            detected_fraction_ext, flux_fraction_ext = image.do_completeness(extended_flux, flux_col,
                                                                             flux_bins, n_sim, n_samples,
                                                                             imsize, square,
                                                                             no_delete, s_type='ext')

            ext_completeness = {'flux_bins': flux_bins,
                                'detected_fraction': detected_fraction_ext,
                                'flux_fraction': flux_fraction_ext}

            # Save data to a json file
            with open(data_file, 'w') as outfile:
                json.dump(ext_completeness, outfile,
                          indent=4, sort_keys=True,
                          separators=(',', ': '),
                          ensure_ascii=False,
                          cls=NumpyEncoder)

        image.plot_results(flux_bins, detected_fraction_point=point_completeness['detected_fraction'], 
                           detected_fraction_ext=ext_completeness['detected_fraction'])

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("image",
                        help="""Image to add sources to and measure completeness,
                                should contain no sources, only residuals. An empty
                                image containing only noise can also be create from scratch
                                by typing 'empty' instead of a filename. In this case
                                the header and other properties will be read from 
                                parsets/empty_image.json""")
    parser.add_argument("simulated_catalog",
                        help="""Name of catalog containing sources, flux to be drawn
                                from for input catalogs""")
    parser.add_argument("flux_col",
                        help="Name of column in catalog containing log flux values")
    parser.add_argument("--outdir", default=None, type=str,
                        help="""Directory to store output, by default output is stored
                              in the directory of the input image""")
    parser.add_argument("--min_flux", default=-5, type=float,
                        help="Log of minimum probed flux in Jansky.")
    parser.add_argument("--max_flux", default=-2, type=float,
                        help="Log of maximum probed flux in Jansky.")
    parser.add_argument("--n_samples", default=5000, type=int,
                        help="Number of sources to be drawn")
    parser.add_argument("--flux_bins", default=30, type=int,
                        help="Amount of flux bins")
    parser.add_argument("--n_sim", default=10, type=int,
                        help="Number of simulations to run")
    parser.add_argument("--orig_counts", action="store_true",
                        help="""Instead of choosing how many sources are in the image,
                                preserve the number counts from the simulated catalog
                                and sources are injected from a randomly chose patch 
                                of the catalog. (default = False)""")
    parser.add_argument("--imsize", default=None, type=float,
                        help="""Specify size of the image in degrees for generating 
                                source positions. If the image is circular this is the 
                                diameter. By default this is read from the image header.""")
    parser.add_argument("--square", action="store_true",
                        help="Generate sources positions in a square rather than circle.")
    parser.add_argument("--no_delete", action="store_true",
                        help="Do not delete pybdsf logs and images")

    return parser

if __name__ == '__main__':
    main()