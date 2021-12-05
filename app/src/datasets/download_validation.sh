#/bin/sh

echo "Downloading reais bankbnote dataset. This may take a while"
wget -c -q --show-progress -O ./validation.zip https://www.dropbox.com/s/est8xcd9blthj44/validation.zip?dl=0
echo "Extracting validation dataset..."
unzip validation.zip; rm validation.zip

echo "Finished."
