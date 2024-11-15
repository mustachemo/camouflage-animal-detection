#!/bin/bash

# Download the file using curl
curl -L "https://drive.usercontent.google.com/download?id=1lLDZwQ0JiUM9FxTPGUGNQJhzBEkgm7x4&confirm=xxx" -o CAMO-V1.0.zip
curl -L "https://drive.usercontent.google.com/download?id=1pRbZVVWRbS3Czqmr7kaQqaSGCiOfEMNr&confirm=xxx" -o CAMO-COCO-V1.0.zip

# Unzip the downloaded file
unzip CAMO-V1.0.zip
unzip CAMO-COCO-V1.0.zip

# Delete the zip file
rm CAMO-V1.0.zip
rm CAMO-COCO-V1.0.zip

echo "Download and extraction completed."