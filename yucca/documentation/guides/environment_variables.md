# Setting up environment variables

To setup environment variables add the following `.env` file to the root Yucca folder:

```
YUCCA_SOURCE=<path-to-non-converted-datasets>
YUCCA_RAW_DATA=<path-to-yucca-data-diretory>/raw_data
YUCCA_PREPROCESSED_DATA=<path-to-yucca-data-diretory>/preprocessed_data
YUCCA_MODELS=<path-to-yucca-data-diretory>/models
YUCCA_RESULTS=<path-to-yucca-data-diretory>/results
```
As an example `<path-to-yucca-data-diretory>` could be substituted by `/data/yucca`.

What's the purpose of this?

To the maximum extent possible we don't want hardcoded paths. It is simply bad practice and completely lacks scalability. Using environment variables allows us to limit hardcoded paths to the Task Conversion scripts.
