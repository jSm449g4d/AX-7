
apt install p7zip-full
echo "download from \"Index of /~taesung_park/CycleGAN/datasets\""
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip
echo $URL
wget -N $URL
7z x apple2orange.zip
rm apple2orange.zip
