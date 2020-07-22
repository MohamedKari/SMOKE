identify im_in.png

# individual
mkdir -p cropped
convert in/0.png -crop 50%x100%+0+0 cropped/0.png

mkdir -p padded
convert cropped/0.png -gravity center -background black -extent 1242x375 padded/0.png

# bulk
rm -rf cropped
rm -rf padded

cp -r in cropped
mogrify -crop 50%x100%+0+0 cropped/*

cp -r cropped padded
mogrify -gravity center -background black -extent 1242x375 padded/*



# waymo to kitti

# identify 000134.png
# 000134.png PNG 1242x375 2484x375+0+0 8-bit sRGB 419286B 0.010u 0:00.001

cd ../waymo
mogrify -format png *.jpg


# https://www.imagemagick.org/discourse-server/viewtopic.php?t=13175
mogrify -resize "375^>" *.png

mogrify -gravity center -background black -extent 1242x375 *.png

mkdir ../png && mv *.png ../png