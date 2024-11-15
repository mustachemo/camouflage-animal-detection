#!/bin/bash

# Download the file using curl
curl -L "https://drive.usercontent.google.com/download?id=1vRYAie0JcNStcSwagmCq55eirGyMYGm5&confirm=xxx" -o COD10K-v3.zip

# Unzip the downloaded file
unzip COD10K-v3.zip

# Delete the zip file
rm COD10K-v3.zip

echo "Download and extraction completed."