#! /bin/bash
 
#SBATCH --job-name=glue_baseline
#SBATCH --output=/data/nlp/gogamza/logs/log_target_evals.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=28-00:00:00


export JIANT_PATH="/nfs/jiant"


#docker run --runtime=nvidia --rm -v "/data/nlp/gogamza/jiant:/nfs/jsalt" jiant:latest \
#       -e "NFS_PROJECT_PREFIX=/nfs/jsalt/exp/docker" \
#       -e "JIANT_PROJECT_PREFIX=/nfs/jsalt/exp/docker" \
#       python $JIANT_PATH/main.py --config_file $JIANT_PATH/conf/demo.conf \
#       --overrides "exp_name = my_exp, run_name = foobar, d_hid = 256"


echo "MY ID is ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DOCKER_ID=$DOCKER_ID"


srun docker run --runtime=nvidia  --rm  -v "/data/nlp/gogamza/jiant:/nfs/jsalt" -v "/data/nlp/gogamza/jiant_git:$JIANT_PATH" BDP-TBRAIN-GPU01:5000/jiant:0.001 \
       python $JIANT_PATH/main.py --config_file $JIANT_PATH/config/glue_exp_02.conf \
       --overrides "exp_name = glue_02, run_name = f01"

wait

