#!/bin/bash
set -x
set -e

OUT_PATH="temp_files" # VG80K/test

N_OBJ=0 # number of object categories
N_REL=0 # number of relationship categories

H5=VG-SGG.h5
JSON=VG-SGG-dicts.json
FRAC=1
IMDB=
OBJECTS=
RELS=
IMAGES_LIST=
MERGED_BOXES=./tmp/merged_boxes.json
FILTERED_FILE=./tmp/filtered_rel_data.json

python vg_to_roidb.py \
    --imdb $IMDB \
    --json_file $OUT_PATH/$JSON \
    --h5_file $OUT_PATH/$H5 \
    --load_frac $FRAC \
    --object_input $OBJECTS \
    --relationship_input $RELS \
    --num_objects $N_OBJ \
    --num_predicates $N_REL \
    --class_selection False 
    #--class_selection
    #--merged_boxes $MERGED_BOXES 
    #--object_list  $OBJECTS \
    #--predicate_list $PREDICATES
    #--images_list $IMAGES_LIST \
    #    --object_alias VG/object_alias_indoor.txt \
