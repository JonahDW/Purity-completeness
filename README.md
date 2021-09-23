# Purity-completeness

The purpose of this module is to assess the purity (how many of my sources are real?) and completeness (how many sources are detected?) of an image or series of images. This code makes extensive use `astropy` and uses PyBDSF and its dependencies for its sourcefinding, which can be found [here](https://github.com/lofar-astron/PyBDSF). At the moment, the usage of this module assumes that regular sourcefinding has already been performed using the scripts in the [Image-processing](https://github.com/JonahDW/Image-processing) module, as certain output from that process is expected to be present (directory structure and output image).

## purity.py 

Assess the purity of sourcefinding by extracting the amount of fake sources in an image, accomplished by inverting the regular image and performing sourcefinding on it. This can be compared to the regular catalog of sources already obtained.

```
usage: purity.py [-h] input_dir full_catalog

positional arguments:
  input_dir     Input directory containing pybdsf directories.
  full_catalog  Full catalog for comparison with inverse catalog.

optional arguments:
  -h, --help    show this help message and exit
```


## completeness.py

Assess the completeness of your image by injecting fake sources drawn from a catalog of simulated sources into the desired image (ideally a residual image), performing sourcefinding on the image, and seeing how many sources are retrieved.

```
usage: completeness.py [-h] [--min_flux MIN_FLUX] [--max_flux MAX_FLUX]
                       [--n_samples N_SAMPLES] [--flux_bins FLUX_BINS]
                       [--n_sim N_SIM]
                       image simulated_catalog flux_col

positional arguments:
  image                 Image to add sources to and measure completeness,
                        should contain no sources, only residuals.
  simulated_catalog     Name of catalog containing sources, flux to be drawn
                        from for input catalogs
  flux_col              Name of column in catalog containing log flux values

optional arguments:
  -h, --help            show this help message and exit
  --min_flux MIN_FLUX   Log of minimum probed flux in Jansky.
  --max_flux MAX_FLUX   Log of maximum probed flux in Jansky.
  --n_samples N_SAMPLES
                        Number of sources to be drawn
  --flux_bins FLUX_BINS
                        Amount of flux bins
  --n_sim N_SIM         Number of simulations to run
```
