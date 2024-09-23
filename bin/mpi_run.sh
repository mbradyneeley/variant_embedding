#!/bin/bash

module load openmpi/gcc/64/4.1.5
module load cuda12.2/toolkit/12.2.2
#module load anaconda3/3.11
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/home/matthewn/projects/GPT_Diagnosis/variant_embedding/bin
MAIN_MACHINE="sabercore-a100-002"
MAIN_PROCESS_IP=$(host $MAIN_MACHINE | cut -d" " -f 4)
HOST="$MAIN_MACHINE"
N_MACHINE=$(echo $HOST | tr ',' '\n' | wc -l)
N_PROCESS=$N_MACHINE

echo $N_PROCESS
echo $N_MACHINE
echo $MAIN_PROCESS_IP
echo $MAIN_MACHINE

mpirun -np $N_PROCESS \
  --output-filename mpi_out \
  --prefix /cm/shared/apps/openmpi4/gcc/4.1.5 \
  --host $HOST \
  -x MAIN_PROCESS_IP=$MAIN_PROCESS_IP \
  -x N_MACHINE=$N_MACHINE \
  -x N_PROCESS=$N_PROCESS \
  -x CUDA_HOME \
  -wd $WORKDIR \
  bash -c '
    /home/matthewn/.conda/envs/kuda/bin/accelerate launch \
    --config_file ./accelerate_config_singlegpu_brady.yaml \
    --main_process_ip $MAIN_PROCESS_IP \
    --num_machines $N_MACHINE \
    --num_processes $N_PROCESS \
    --machine_rank $OMPI_COMM_WORLD_RANK \
    --mixed_precision bf16 \
    generate_textual_variants.py
#!/bin/bash

#module load openmpi/gcc/64/4.1.5
#module load cuda12.2/toolkit/12.2.2
##module load anaconda3/3.11
#export PYTHONUNBUFFERED=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
#WORKDIR=/home/matthewn/projects/GPT_Diagnosis/variant_embedding/bin
#MAIN_MACHINE="sabercore-a100-002"
#MAIN_PROCESS_IP=$(host $MAIN_MACHINE | cut -d" " -f 4)
#HOST="$MAIN_MACHINE"
#N_MACHINE=$(echo $HOST | tr ',' '\n' | wc -l)
#N_PROCESS=$N_MACHINE
#
#echo $N_PROCESS
#echo $N_MACHINE
#echo $MAIN_PROCESS_IP
#echo $MAIN_MACHINE
#
#mpirun -np $N_PROCESS \
#  --output-filename mpi_out \
#  --prefix /cm/shared/apps/openmpi4/gcc/4.1.5 \
#  --host $HOST \
#  -x MAIN_PROCESS_IP=$MAIN_PROCESS_IP \
#  -x N_MACHINE=$N_MACHINE \
#  -x N_PROCESS=$N_PROCESS \
#  -x CUDA_HOME \
#  -wd $WORKDIR \
#  bash -c '
#    # Activate vllm within the Python script using a subprocess or wrapper
#    export PYTHONPATH=/home/matthewn/.conda/envs/vllm/lib/python3.11/site-packages:$PYTHONPATH
#
#    /home/matthewn/.conda/envs/kuda/bin/accelerate launch \
#    --config_file ./accelerate_config_singlegpu_brady.yaml \
#    --main_process_ip $MAIN_PROCESS_IP \
#    --num_machines $N_MACHINE \
#    --num_processes $N_PROCESS \
#    --machine_rank $OMPI_COMM_WORLD_RANK \
#    --mixed_precision bf16 \
#    vllm_generate_textual_variants.py
#  '
##  '
