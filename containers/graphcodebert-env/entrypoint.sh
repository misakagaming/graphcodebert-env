#!/bin/bash
source /opt/python3/venv/base/bin/activate
echo "nice"
cd translation
cd parser
bash build.sh
cd ..
python3 run.py --do_train --do_eval --model_type roberta --source_lang java --model_name_or_path microsoft/graphcodebert-base --tokenizer_name microsoft/graphcodebert-base --config_name microsoft/graphcodebert-base --train_filename data/train.java-cs.txt.java,data/train.java-cs.txt.cs --dev_filename data/valid.java-cs.txt.java,data/valid.java-cs.txt.cs --output_dir saved_models/java-cs --max_source_length 320 --max_target_length 256 --beam_size 10 --train_batch_size 4 --eval_batch_size 4 --learning_rate 1e-4 --num_train_epochs 50 2>&1 | tee "saved_models/java-cs/train.log"
exec "$@"
