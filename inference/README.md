# Inference integration

`catvton_masked.py:run_pair` is the PropVTON call boundary for mask selection and CatVTON execution. It accepts an already initialized CatVTON pipeline, PIL person/garment images, split and pair IDs, a mask source, and standard pipeline keyword arguments. See [BAM-Net mask integration](../docs/BAMNET_INTEGRATION.md) for paths and conventions.

There is not yet a full dataset inference command or checkpoint-loading runner. No real inference has been run from this integration.
