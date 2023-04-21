#!/usr/bin/env bash

echo "Formatting files..."

find -type f -name '*.py' -exec autopep8 --in-place --aggressive --aggressive '{}' \;
find -type f -name '*.py' -exec pycodestyle --first '{}' \;