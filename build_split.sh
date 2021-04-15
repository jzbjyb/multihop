#!/usr/bin/env bash

root=$1

python prep.py --task combine_multi_targets --input ${root}/train.target
python prep.py --task combine_multi_targets --input ${root}/val_all.target
head -n 5 ${root}/val_all.source > ${root}/val.source
head -n 5 ${root}/val_all.target > ${root}/val.target
pushd ${root}
rm test.source
rm test.target
ln -s val.source test.source
ln -s val.target test.target
popd
