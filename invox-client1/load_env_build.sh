#!/usr/bin/env bash

# Print shell input lines as they are read
set -v

# Print command traces before executing command
set -x

# Don't exit on errors
set -o errexit

# Don't error on missing variables
set -o nounset

# Export args
if [ $# -ne 0 ]; then
  for var in "$@"
  do
    export "$var"
  done
fi

# Print env variables
# env

# Add node executables to path in order to have npm available
export PATH=/app/node_modules/.bin:$PATH
# Add global node executables
export PATH=$HOME/.node_modules_global/bin:$PATH

# Build node app with base path
npm run build

# Ensure target directory exists
mkdir -p /usr/share/nginx/html

# clean previous build
rm -rf /usr/share/nginx/html/*

# Move generated files to QA subdirectory
mv -f /app/dist/* /usr/share/nginx/html

# keep active
trap : TERM INT; sleep infinity & wait

