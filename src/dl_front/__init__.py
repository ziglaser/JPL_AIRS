"""DL-FRONT replication (Biard & Kunkel 2019) + dryline extension.

Sibling of ``front_finder`` (the FrontFinder/UNET3+ track); shares its
label loaders, neighborhood evaluation, degradation operators and stage
A/B/C curriculum so both models ride the same reanalysis -> degraded ->
real-AIRS fine-tune framework.
"""
