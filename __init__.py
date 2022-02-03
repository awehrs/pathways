download_config = tfds.download_config(
    extract_dir=extract_dir,
    manual_dir=manual_dir,
)
ds = tfds.load(
    "imagenet2012:5.*.*",
    data_dir=data,
    download_config=None,
)
