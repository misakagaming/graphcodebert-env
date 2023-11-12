#!/bin/bash
conda init bash
conda activate base
exec "$@"
