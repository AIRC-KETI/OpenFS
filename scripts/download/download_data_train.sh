mkdir -p data/fingerspelling
mkdir -p data/fingerspelling_fsboard

## For training
### Raw CFSW & CFSWP
gdown --id 1PqK0flwJ4RVX6ZxVDTVZQYOoYRnnFDGU -O data/fingerspelling_raw.zip
unzip data/fingerspelling_raw.zip -d data/
rm data/fingerspelling_raw.zip
mv data/fingerspelling_raw/*.npz data/fingerspelling
rm -r data/fingerspelling_raw
rm -r data/__MACOSX
### Processed CFSW & CFSWP
gdown --id 1hHjM37AQQrdIepzqJeYEDBfsWRhI_6S2 -O data/fingerspelling_proc.zip
unzip data/fingerspelling_proc.zip -d data/
rm data/fingerspelling_proc.zip
mv data/fingerspelling_proc/*.npz data/fingerspelling
rm -r data/fingerspelling_proc
rm -r data/__MACOSX
### Processed & clean CFSW & CFSWP
gdown --id 1FEk_PCZJtV35AUx45WybsgfD-BNnjgoE -O data/fingerspelling_proc_clean.zip
unzip data/fingerspelling_proc_clean.zip -d data/
rm data/fingerspelling_proc_clean.zip
mv data/fingerspelling_proc_clean/*.npz data/fingerspelling
rm -r data/fingerspelling_proc_clean
rm -r data/__MACOSX
### Generated training data
gdown --id 1GxmfpftQ4Z-bQ9Y88MyqP8wb056wFAtL -O data/fingerspelling_alpha.zip
unzip data/fingerspelling_alpha.zip -d data/
rm data/fingerspelling_alpha.zip
mv data/fingerspelling_alpha/*.npz data/fingerspelling
rm -r data/fingerspelling_alpha
rm -r data/__MACOSX
### Raw FSboard
gdown --id 1M6jOtBfsoyGdhP4--8qimGA2BVqhwtTa -O data/fingerspelling_fsboard_raw.zip
unzip data/fingerspelling_fsboard_raw.zip -d data/
rm data/fingerspelling_fsboard_raw.zip
mv data/fingerspelling_fsboard_raw/*.npz data/fingerspelling_fsboard
rm -r data/fingerspelling_fsboard
rm -r data/__MACOSX
### Processed FSboard
gdown --id 13jTUsYEEz0lYI1YjYduF46ssPRvGlD3Q -O data/fingerspelling_fsboard_proc.zip
unzip data/fingerspelling_fsboard_proc.zip -d data/
rm data/fingerspelling_fsboard_proc.zip
mv data/fingerspelling_fsboard_proc/*.npz data/fingerspelling_fsboard
rm -r data/fingerspelling_fsboard_proc
rm -r data/__MACOSX
### Processed & clean FSboard
gdown --id 1dhcrbZeb3-5IshIkzatVmNO3MOTVFjJW -O data/fingerspelling_fsboard_proc_clean.zip
unzip data/fingerspelling_fsboard_proc_clean.zip -d data/
rm data/fingerspelling_fsboard_proc_clean.zip
mv data/fingerspelling_fsboard_proc_clean/*.npz data/fingerspelling_fsboard
rm -r data/fingerspelling_fsboard_proc_clean
rm -r data/__MACOSX