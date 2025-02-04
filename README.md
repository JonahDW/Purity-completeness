# Purity-completeness

The purpose of this module is to assess the purity (how many of my sources are real?) and completeness (how many sources are detected?) of an image or series of images. This code makes extensive use `astropy` and uses PyBDSF and its dependencies for its sourcefinding, which can be found [here](https://github.com/lofar-astron/PyBDSF). At the moment, the usage of this module assumes that regular sourcefinding has already been performed using the scripts in the [Image-processing](https://github.com/JonahDW/Image-processing) module, as certain output from that process is expected to be present (directory structure and output image).

## completeness.py

Assess the completeness of your image by injecting fake sources drawn from a catalog of simulated sources into the desired image (ideally a residual image), performing sourcefinding on the image, and seeing how many sources are retrieved.

```
usage: completeness.py [-h] [--realistic_counts] [--flux_bins FLUX_BINS]
                       [--n_sim N_SIM] [--sim_cat SIM_CAT]
                       [--flux_col FLUX_COL] [--min_flux MIN_FLUX]
                       [--max_flux MAX_FLUX] [--n_sources N_SOURCES]
                       [--imsize IMSIZE] [--square] [--outdir OUTDIR]
                       [--no_delete]
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
  --realistic_counts    Use a realistic flux density and spatial distribution
                        instead of uniform per bin and in space. Simulated
                        catalogue must be specified to draw from. (default =
                        False)
  --flux_bins FLUX_BINS
                        Amount of flux bins
  --n_sim N_SIM         Number of simulations to run
  --sim_cat SIM_CAT     Name of catalog containing sources, flux to be drawn
                        from for input catalogs
  --flux_col FLUX_COL   Name of column in catalog containing log flux values
  --min_flux MIN_FLUX   Log of minimum probed flux in Jansky.
  --max_flux MAX_FLUX   Log of maximum probed flux in Jansky.
  --n_sources N_SOURCES
                        Number of sources to be drawn
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
