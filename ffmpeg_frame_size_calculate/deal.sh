#!/bin/sh
 

 
Folder_A="F:/youtube_video/bitrate/football/1100k"
for file_a in ${Folder_A}/*
do
    out_filename=`basename $file_a`
    in_filename="_CIDI_"${out_filename}
    python -m ffmpeg_bitrate_stats  -a time -c 30 -of csv F:/youtube_video/bitrate/football/1100k/$out_filename
done
