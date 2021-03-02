#/bin/sh

echo "Downloading reais bankbnote dataset. This may take a while"
wget -c -q --show-progress -O ./datasets.zip https://www.dropbox.com/s/jq4naabiu0a1z9r/data.zip?dl=0
echo "Extracting train and validation dataset..."
unzip datasets.zip; rm datasets.zip

echo "Finished."
