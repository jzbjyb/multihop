#!/usr/bin/env bash

root=$1

python prep.py --task combine_multi_targets --input ${root}/train.target
python prep.py --task combine_multi_targets --input ${root}/val_all.target
head -n 100 ${root}/val_all.source > ${root}/val.source
head -n 100 ${root}/val_all.target > ${root}/val.target
pushd ${root}
ln -s val.source test.source
ln -s val.target test.target
popd
