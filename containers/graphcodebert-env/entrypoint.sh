#!/bin/bash
source /opt/conda/bin/activate
conda activate base
exec "$@"
