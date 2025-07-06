# Audio Idenfitication

Implementation of Philips algorithm [1], including:
- Fingerprinting
- Search algorithm

[1] Haitsma, J., & Kalker, T. (2002, October). A highly robust audio fingerprinting system. In Ismir (Vol. 2002, pp. 107-115).


## Running Fingerprinting
1. Ensure data is in data folders
`database_recordings/` - database wav files should be placed here
`query_recordings/` - query wav files should be placed here

2. Run Fingerprinting:
```
from fingerprint import fingerprintBuilder

db_path = "database_recordings"
fingerprint_path = "fingerprints"

fpb = fingerprintBuilder()
fpb.overlap_factor=64
fpb(db_path, fingerprint_path)
```
The database of fingerprints will be stored in `fingerprint_path` as pickle files.

3. Run Audio Idenfitication:
```
from audio_identification import audioIdentification

fingerprint_path = "fingerprints"
query_path = "query_recordings"
output_path = "query_output.txt"

# Run audio identifier
aid = audioIdentification()
aid.overlap_factor=32
aid(query_path, fingerprint_path, output_path)
```
The query output will be stored as "<search type>_<overlap_factor>_<output_path>" (e.g. `simple_32_query_output.txt`)

## Directory description
- `fingerprinting_philips.ipynb` - notebook for developing the philips algorithm
- `audio_id_eval.ipynb` - notebook for running the philips algorithm and evaluation
- `fingerprint.py` - script containing Fingerprinting component
- `audio_identification.py` - script containing Audio Identification component
