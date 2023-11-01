# Setting up environment variables

To setup environment variables add the following `.env` file to the root Yucca folder:

```
YUCCA_RAW_DATA=/path/to/yucca_data/raw_data
YUCCA_PREPROCESSED="/path/to/yucca_data/preprocessed"
YUCCA_MODELS="/path/to/yucca_data/models"
YUCCA_RESULTS="/path/to/yucca_data/results"
```

What's the purpose of this?

To the maximum extent possible we don't want hardcoded paths. It is simply bad practice and completely lacks scalability. Using environment variables allows us to limit hardcoded paths to the Task Conversion scripts.
