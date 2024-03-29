import nibabel as nib
from pathlib import Path

PATH = "/home/bertan/TractSeg/data/HCP/987983/tracts/"

paths = list(Path(PATH).iterdir())

for path in paths:
    p = str(path)
    trk = nib.streamlines.load(p)
    nib.streamlines.save(trk.tractogram, f"{p[:-4]}.tck")
