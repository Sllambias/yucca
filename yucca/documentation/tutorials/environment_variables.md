# Setting up environment variables

Edit the .bashrc file (found in the home folder)

and add the following lines (editing the paths to your liking):
```
export yucca_raw_data="/path/to/YuccaData/yucca_raw_data"
export yucca_preprocessed="/path/to/YuccaData/yucca_preprocessed"
export yucca_models="/path/to/YuccaData/yucca_models"
export yucca_results="/path/to/YuccaData/yucca_results"
```

What's the purpose of this?

To the maximum extent possible we don't want hardcoded paths. It is simply bad practice and completely lacks scalability. Using environment variables allows us to limit hardcoded paths to the Task Conversion scripts. 
