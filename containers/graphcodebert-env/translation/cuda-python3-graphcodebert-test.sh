#!/bin/bash
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate


source=java
target=cs
lr=1e-4
batch_size=32
beam_size=10
source_length=320
target_length=256
output_dir=saved_models/$source-$target/
train_file=data/train.java-cs.txt.$source,data/train.java-cs.txt.$target
dev_file=data/valid.java-cs.txt.$source,data/valid.java-cs.txt.$target
epochs=100
pretrained_model=microsoft/graphcodebert-base
current=/users/eray.erer/CodeBERT-master/CodeBERT-master/GraphCodeBERT/translation

mkdir -p saved_models/java-cs/
python current/run.py --do_train --do_eval --model_type roberta --source_lang java --model_name_or_path microsoft/graphcodebert-base --tokenizer_name microsoft/graphcodebert-base --config_name microsoft/graphcodebert-base --train_filename current/data/train.java-cs.txt.java,current/data/train.java-cs.txt.cs --dev_filename current/data/valid.java-cs.txt.java,data/valid.java-cs.txt.cs --output_dir saved_models\java-cs --max_source_length 320 --max_target_length 256 --beam_size 10 --train_batch_size 4 --eval_batch_size 4 --learning_rate 1e-4 --num_train_epochs 50 2>&1 | tee "saved_models\java-cs\train.log"