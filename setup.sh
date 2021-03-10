#!/usr/bin/env bash

conda create -n knowlm python=3.7
conda activate knowlm

pip install http://www.jbox.dk/sling/sling-2.0.0-py3-none-linux_x86_64.whl
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
