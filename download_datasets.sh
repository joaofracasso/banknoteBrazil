echo "Downloading reais bankbnote dataset. This may take a while"
wget -c -q --show-progress -O ./dataset.zip https://www.dropbox.com/s/jq4naabiu0a1z9r/data.zip?dl=0

echo "Extracting Sid dataset..."
unzip dataset.zip; rm dataset.zip

echo "Finished."
