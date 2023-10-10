#!/bin/sh
 

 
Folder_A="/data/dyn123/youtube_video/dyn/1500k训练集/264/concert"
for file_a in ${Folder_A}/*
do
    out_filename=`basename $file_a`
    in_filename="_CIDI_"${out_filename}
    python -m ffmpeg_bitrate_stats  -a time -c 30 -of csv /data/dyn123/youtube_video/dyn/1500k训练集/264/concert/$out_filename
done

