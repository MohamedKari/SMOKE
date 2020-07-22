# carves out a single tracklet from the kitti 3d tracking dataset looking like the kitti 3d detection dataset
# carveout.sh 0002

TESTING_ID="$1"

if [ -z "$TESTING_ID" ]; then
echo Please pass the testing sequence id! >&2
exit 1
fi

rm -rf track
mkdir track
cp -r testing/ImageSets/"$TESTING_ID" track/ImageSets
cp -r testing/calib/"$TESTING_ID" track/calib
cp -r testing/image_02/"$TESTING_ID" track/image_2 