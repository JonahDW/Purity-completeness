# Purity-completeness

The purpose of this module is to assess the purity and completeness of a radio astronomical image. This code makes extensive use `astropy` and uses [PyBDSF](https://github.com/lofar-astron/PyBDSF) and its dependencies for its sourcefinding. Usage of this module assumes that regular sourcefinding has already been performed with PyBDSF, with certain outputs from that process expected to be present (residual, mean and rms images).

## completeness.py

Assess the completeness of your image by injecting fake sources, either as randomly drawn point sources, or more realistic sources drawn from a catalog of simulated sources into the desired image, performing sourcefinding on the image, and seeing how many sources are retrieved. Recovered sources are matched to the original input catalog, enabling a range of verification and consistency tests of the sourcefinding procedure.

```
usage: completeness.py [-h] [--flux_bins FLUX_BINS] [--n_sim N_SIM]
                       [--min_flux MIN_FLUX] [--max_flux MAX_FLUX]
                       [--n_sources N_SOURCES] [--sim_cat SIM_CAT]
                       [--flux_col FLUX_COL] [--real_flux_dist]
                       [--real_sky_dist] [--pos_col POS_COL] [--imsize IMSIZE]
                       [--square] [--outdir OUTDIR] [--no_delete]
                       image_name sources

positional arguments:
  image_name            Image to measure completeness on, without any
                        extension. PyBDSF file structure is assumed for the
                        residual ('image_name_gaus_resid.fits'), rms
                        ('image_name_rms.fits'), and mean
                        ('image_name_mean.fits'). An empty image containing
                        only noise can also be created from scratch by typing
                        'empty' instead of a filename. In this case the header
                        and other properties will be read from
                        parsets/empty_image.json
  sources               Type of sources to use for simulations. Options are:
                        'point': Only injects point sources 'extended': Only
                        injects sources with a nonzero size 'realistic':
                        Inject sources of all sizes, with realistic size
                        distribution. If nonzero source sizes are used in any
                        way, sources must be drawn from specified simulated
                        catalogue.

optional arguments:
  -h, --help            show this help message and exit
  --flux_bins FLUX_BINS
                        Amount of flux density bins
  --n_sim N_SIM         Number of simulations to run
  --min_flux MIN_FLUX   Log of minimum probed flux in Jansky.
  --max_flux MAX_FLUX   Log of maximum probed flux in Jansky.
  --n_sources N_SOURCES
                        Number of sources to be drawn
  --sim_cat SIM_CAT     Name of catalog containing sources, flux to be drawn
                        from for input catalogs
  --flux_col FLUX_COL   Name of column in catalog containing log flux values
  --real_flux_dist      Use the flux density distribution from the simulated
                        catalogue instead of a (log) uniform one. (default =
                        False)
  --real_sky_dist       Use the spatial distribution from the simulated
                        catalog instead of a uniform sky distribution. Instead
                        of randomly generating source positions, a region of
                        sky corresponding to the image size is extracted.
                        (default = False)
  --pos_col POS_COL     Column names of source positions (lat,lon) to use in
                        case realistic spatial distribution is specified.
                        These are assumed to be in the same coordinate system
                        of the image, no conversion is made. (default =
                        'RA,DEC')
  --imsize IMSIZE       Specify size of the image in degrees for generating
                        source positions. If the image is circular this is the
                        diameter. By default this is read from the image
                        header.
  --square              Generate sources positions in a square rather than
                        circle.
  --outdir OUTDIR       Directory to store output, by default output is stored
                        in the directory of the input image
  --no_delete           Do not delete pybdsf logs and images
```

## purity.py 

Assess the purity of sourcefinding by extracting the amount of fake sources in an image, accomplished by inverting the regular image and performing sourcefinding on it. This can be compared to the regular catalog of sources already obtained.

```
usage: purity.py [-h] [--full_catalog FULL_CATALOG] [--clean_up] input_dir

positional arguments:
  input_dir             Input directory containing pybdsf directories.

optional arguments:
  -h, --help            show this help message and exit
  --full_catalog FULL_CATALOG
                        Full catalog for comparison with inverse catalog.
  --clean_up            Clean up output for individual images leaving only the
                        combined catalog
```
