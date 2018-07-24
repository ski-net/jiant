#!/bin/bash

# DO NOT COMMIT CHANGES TO THIS FILE! Make a local copy and follow the
# instructions below.

# Copy this to /etc/profile.d/ to auto-set environment vars on login.
# Or, make a copy of this, customize, and run immediately before the training
# binary:
# cp path_config.sh ~/my_path_config.sh
# source ~/my_path_config.sh; python main.py --config ../config/demo.conf \
#   --overrides "do_train = 0"

# Default environment variables for JSALT code. May be overwritten by user.
# See https://github.com/jsalt18-sentence-repl/jiant for more.

##
# Example of custom paths for a local installation:
# export JIANT_PROJECT_PREFIX=/Users/Bowman/Drive/JSALT
# export JIANT_DATA_DIR=/Users/Bowman/Drive/JSALT/jiant/glue_data
# export JIANT_DATA_DIR=/home/raghu1991_p_gmail_com/
# export WORD_EMBS_FILE=~/glove.840B.300d.txt
# export FASTTEXT_MODEL_FILE=None
# export FASTTEXT_EMBS_FILE=None

export NFS_PROJECT_PREFIX=/nfs/jsalt/exp/nkim/edgeprobe
export JIANT_PROJECT_PREFIX=${NFS_PROJECT_PREFIX}
export JIANT_DATA_DIR=/nfs/jsalt/home/nkim
export JIANT_DATA_DIR=/nfs/jsalt/share/glue_data
export NFS_DATA_DIR=${JIANT_DATA_DIR}
export MODEL_DIR=/nfs/jsalt/exp/nkim/models
export NOELMO_MODEL_DIR=/nfs/jsalt/exp/rtmccoy4-worker120/main-dissentbwb/noelmo-do2-sd1
export ELMO_MODEL=${MODEL_DIR}/mnli-elmo-do2-sd1/model_state_best.th
export VN_NOELMO_MODEL=/nfs/jsalt/exp/nkim/vn/train_test/noelmo-do2/model_state_eval_best.th
export VN_ELMO_MODEL=/nfs/jsalt/exp/nkim/vn/train_test/elmo-do2/model_state_eval_best.th
#export NOELMO_MODEL=${MODEL_DIR}/mnli-noelmo-do2-sd1/model_state_main_epoch_23.best_macro.th
export NOELMO_MODEL=${NOELMO_MODEL_DIR}/model_state_eval_best.th
export ELMO_PARAMS_FILE=${MODEL_DIR}/mnli-elmo-do2-sd1/params.conf
export NOELMO_PARAMS_FILE=${MODEL_DIR}/mnli-noelmo-do2-sd1/params.confi
export NOELMO_PARAMS_FILE=${NOELMO_MODEL_DIR}/params.conf
export MODELS_PATH=/nfs/jsalt/share/models_to_probe

export NLI=${MODELS_PATH}/nli_do2_noelmo
export LM=${MODELS_PATH}/lm_do2_noelmo
export MT=${MODELS_PATH}/mt_do2_noelmo
export SYNTAX=${MODELS_PATH}/syntax_do2_noelmo
export IMG=${MODELS_PATH}/natural_images_do2_noelmo
export DISSENT=${NOELMO_MODEL_DIR}
export FRACAS=/nfs/jsalt/exp/nkim/pretrain-fracas/train-fracas-eval-glue
export COLA=${MODEL_DIR}/cola-noelmo-do2-sd1
export VN=/nfs/jsalt/exp/nkim/vn/train_0718/noelmo-do2-f1
export REDDIT=${MODELS_PATH}/reddit_noelmo
export RANDOMINIT=${MODEL_DIR}/random-noelmo

export CODE=/nfs/jsalt/home/nkim/code/jiant
