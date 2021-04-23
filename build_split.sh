#!/usr/bin/env bash

root=$1

python prep.py --task combine_multi_targets --input ${root}/train.target
python prep.py --task combine_multi_targets --input ${root}/val_all.target
pushd ${root}
#rm val.source
#rm val.target
head -n 5 val_all.source > val.source
head -n 5 val_all.target > val.target
#rm test.source
#rm test.target
ln -s val.source test.source
ln -s val.target test.target
popd
