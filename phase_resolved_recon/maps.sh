#! /bin/bash
echo "Generating coil sensitivy maps from soft-gated data"

if [ $# -lt 1 ]
then
    echo "Not enough arguments supplied"
    echo "Usage: maps.sh input_root"
    echo " input_root files required include _traj, _data, _sg_dcf"
    exit 113
fi

in=$1

export DEBUG_LEVEL=5
set -x

c=24

dims=(`bart estdims $in"_traj"`)
bart fmac $in"_data_pr" $in"_dcf_pr" $in"_dcfdata"

# Iterative gridding on center

bart nufft -d 208:128:160 -a $in"_traj_pr" $in"_dcfdata" $in"_calib"

bart fft 7 $in"_calib" $in"_kcalib"

# ecalib
time bart ecalib -S -c 0.6 -m 1 -r $c $in"_kcalib" $in"_maps_pr"

rm $in"_calib".*
rm $in"_kcalib".*
rm $in"_dcfdata".*
