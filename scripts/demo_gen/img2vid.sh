#!/bin/bash

# usage: ./img2vid.sh <input_pattern> <output_video> [fps]

if [ $# -lt 2 ]; then
    echo "usage: $0 <input_pattern> <output_video> [fps]"
    exit 1
fi

in_pattern=$1      # e.g., frame_%05d.png
out_vid=$2         # e.g., video.mp4
fps=${3:-30}       # default 30fps

ffmpeg -framerate "$fps" -i "$in_pattern" -c:v libx264 -pix_fmt yuv420p "$out_vid"