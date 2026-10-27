mkdir -p data/fingerspelling
mkdir -p data/fingerspelling_fsboard

gdown 13aKZvmjJOk-jfNX5iSZV-hklt24gUVNS -O data/CustomChicagoFSWildPlusFinal.csv
gdown 1RNZ6fXoAuMrmFefYSoQc5Erw8Yq4Mb-K -O data/posenet_signhand_miss.txt

## For evaluation
### CFSW & CFSWP
gdown 10AdHfAEK9jOLLaouBvBDnj1spKnn9srt -O data/fingerspelling/ChicagoFSWild_test.npz
gdown 1Ri50TU86ULzqvAy-inn6Nk0RmSbrTxfI -O data/fingerspelling/ChicagoFSWildPlus_test.npz
### FSNeo
gdown 1y9MaEQdU9zpUIGJ-Le0KX-LwqU8BBYCE -O data/fsneo.zip
unzip data/fsneo.zip -d data/
rm data/fsneo.zip
mv data/fsneo/*.npz data/fingerspelling
rm -r data/fsneo
rm -r data/__MACOSX
### FSboard
gdown 1CjNvkhOx4uUBYraaFIQZ-fXpotiXEdof -O data/fingerspelling_fsboard/fsboard_test.npz
### English words text files
gdown 1jWxUBcRuQI1HcLbPrDZjINrV-J5dDCjs -O data/english_word.zip
unzip data/english_word.zip -d data/
rm data/english_word.zip
rm -r data/__MACOSX