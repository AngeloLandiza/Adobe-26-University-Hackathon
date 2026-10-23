# Crello reference data

`data/` held the four raw parquet shards of the public Crello dataset's test split
(`test-0000{0..3}-of-00004.parquet`, ~365 MB each). They are not in git: each is over
GitHub's 100 MB file limit. Re-download them from the Crello dataset on Hugging Face
(`cyberagent/crello`, test split) into `data/`. The challenge2/challenge3 and crello_dev
sets were built from them (`chameleon/dev/distill/make_real_challenge.py`,
`make_challenge3.py`, `make_crello_dev.py`). Crello's licence for redistribution is unverified.
