#!/bin/bash

# get the directory that this script is in so you can call it from anywhere
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# your output folder, where sub0MyAgent.py.zip will be placed
# TODO: EDIT
out_dir="$script_dir/submissions/latest"

# locations of your files
# TODO: EDIT
model="$script_dir/results/experiment_2/checkpoint_210_eps.pth"
agent="$script_dir/MyAgent.py"

# remake temp dir
rm -rf "$out_dir"
mkdir -p "$out_dir"

# copy model to temp dir
# TODO: Might need to edit, make sure this is not in a subdirectory of $out_dir
cp "$model" "$out_dir/model.pth"

# cd into temp dir
cd "$out_dir"

# merge dependencies of MyAgent.py
# TODO: Might need to edit python-paths
stickytape "$agent" --add-python-path "$script_dir" --add-python-path "$script_dir/" --output-file "MyAgent.py"

# create zip
zip -r "sub0MyAgent.py.zip" ./*

# status
printf "\033[92mCreated Submission\033[0m: \033[90m$out_dir/sub0MyAgent.py.zip\033[0m\n"
