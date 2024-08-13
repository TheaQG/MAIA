#!/bin/bash
# NOTE : Quote it else use array to avoid problems #
FILES="/Volumes/1TB-FREECOM/MAIA_data/*"
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  cdo timselmean,6 $f $f"_6hr_mean.nc"
  #cat "$f"
done