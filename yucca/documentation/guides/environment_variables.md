# Setting up environment variables
What's the purpose of this?

To the maximum extent possible we don't want hardcoded paths. It is simply bad practice and completely lacks scalability. Using environment variables allows us to limit hardcoded paths to the Task Conversion scripts.

## Yucca installed locally using git clone 
To setup environment variables add the following `.env` file to the root Yucca folder:

```
YUCCA_SOURCE=<path-to-non-converted-datasets>
YUCCA_RAW_DATA=<path-to-yucca-data-diretory>/raw_data
YUCCA_PREPROCESSED_DATA=<path-to-yucca-data-diretory>/preprocessed_data
YUCCA_MODELS=<path-to-yucca-data-diretory>/models
YUCCA_RESULTS=<path-to-yucca-data-diretory>/results
```
As an example `<path-to-yucca-data-diretory>` could be substituted by `/data/yucca`.


## Yucca installed as a package with conda environments
From the official conda [guide](#https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux):

1. Create the following directories and files 
```
conda activate YOUR_YUCCA_ENV_HERE
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```
2. Edit the ./etc/conda/activate.d/env_vars.sh file:
```
export YUCCA_SOURCE=<path-to-non-converted-datasets>
export YUCCA_RAW_DATA=<path-to-yucca-data-diretory>/raw_data
export YUCCA_PREPROCESSED_DATA=<path-to-yucca-data-diretory>/preprocessed_data
export YUCCA_MODELS=<path-to-yucca-data-diretory>/models
export YUCCA_RESULTS=<path-to-yucca-data-diretory>/results
```
3. Edit the ./etc/conda/deactivate.d/env_vars.sh
```
unset YUCCA_RAW_DATA
unset YUCCA_PREPROCESSED_DATA
unset YUCCA_MODELS
unset YUCCA_RESULTS
unset YUCCA_SOURCE
```
4. For HPC users working with workload managers such as Slurm or Spectrum the following might
be necessary to add in your job scripts.
```
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate YOUR_YUCCA_ENV_HERE
```
## Yucca installed as a package with poetry environments
