#!/bin/bash
# Install
sudo python setup.py install
# Deleting trash files
sudo rm -r build/
sudo rm -r dist/
sudo rm -r pythonUtils.egg-info/

