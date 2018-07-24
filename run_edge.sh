#!/bin/bash

# Script to run an edge-probing task on an existing trained model.
# Based on prob_example_run.sh

# NOTE: don't be startled if you see a lot of warnings about missing parameters,
# like:
#    Parameter missing from checkpoint: edges-srl-conll2005_mdl.proj2.weight
# This is normal, because the probing task won't be in the original checkpoint.

declare -a core_paths=(${NLI} ${LM} ${MT} ${SYNTAX} ${IMG} ${REDDIT})
declare -a core_paths_str=("NLI" "LM" "MT" "SYNTAX" "IMG" "REDDIT")

#MODEL_DIR=$1 # directory of checkpoint to probe,
             # e.g: /nfs/jsalt/share/models_to_probe/nli_do2_noelmo
#PROBING_TASK=${2:-"edges-all"}  # probing task name(s)
                                # "edges-all" runs all as defined in
                                # preprocess.ALL_EDGE_TASKS

#PROBING_TASK="edges-all"
PROBING_TASK=edges-srl-conll2005
#,edges-spr2,edges-dep-labeling
for i in "${!core_paths[@]}"
do
MODEL_DIR="${core_paths["$i"]}"
path_str="${core_paths_str["$i"]}"

EXP_NAME="edgeprobe-${path_str}"  # experiment name
RUN_NAME="run_0724"                     # name for this run

PARAM_FILE="${MODEL_DIR}/params.conf"
MODEL_FILE="${MODEL_DIR}/model_state_eval_best.th"

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", eval_tasks = ${PROBING_TASK}"

#pushd "${PWD%jiant*}jiant"

# Load defaults.conf for any missing params, then model param file,
# then eval_existing.conf to override paths & eval config.
# Finally, apply custom overrides defined above.
#python main.py -c config/defaults.conf ${PARAM_FILE} config/edgeprobe_existing.conf -o "${OVERRIDES}" --remote_log
done

# Non-core models
#declare -a noncore_paths=(${DISSENT} ${FRACAS} ${COLA} ${VN} ${RANDOMINIT})
declare -a noncore_paths=(${VN})
declare -a noncore_paths_str=("VN")
#declare -a noncore_paths_str=("DISSENT" "FRACAS" "COLA" "VN" "RANDOMINIT")

for i in "${!noncore_paths[@]}"
do
MODEL_DIR=${noncore_paths["$i"]}
path_str="${noncore_paths_str["$i"]}"

EXP_NAME=edgeprobe-${path_str}
RUN_NAME="run_0724"
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", eval_tasks = ${PROBING_TASK}"

python main.py -c config/defaults.conf ${PARAM_FILE} config/edgeprobe_existing.conf -o "${OVERRIDES}" --remote_log
done


