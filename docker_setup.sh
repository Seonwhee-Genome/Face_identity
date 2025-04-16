#!/bin/bash
apt-get update && apt-cache madison python3-pip
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get install -y python3.9 python3-pip
apt-get install -y libpython3.9-dev
rm /usr/bin/python3 && ln -s /usr/bin/python3.9 /usr/bin/python3
python3 --version
python3 -m pip install --upgrade pip
pip install notebook
apt-get install -y git
apt-get install -y libgl1-mesa-glx
pip install tf-keras
pip install tf-nightly[and-cuda]
