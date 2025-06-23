#!/bin/tcsh
# Combined multinode training script

# Environment setup
module load LSF/mtkgpu
module load openmpi/4.0.3
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

# Check GPU availability
bhosts GPU_3090_4
# bhosts GPU_A6000_8

# Create temporary LSF script
echo "Creating temporary LSF script..."
set lsf_script = /tmp/multinode_$$.lsf
echo "LSF script path: $lsf_script"

cat > $lsf_script << 'END_SCRIPT'
#!/bin/bash
# Required Directives
#BSUB -q gpu
#BSUB -n num=32*
#BSUB -R "span[ptile=8]"
#BSUB -J test
#BSUB -o test.out
#BSUB -e test.err
#BSUB -W HH:MM
#BSUB -g /ML_GPU
#BSUB -app PyTorch

# Optional Directives
#BSUB -J multinode job
#BSUB -o test.err
#BSUB -e test.err
#BSUB -n test

MPI_PREFIX=/mktoss/openmpi/4.0.3.ubuntu22/x86-64

# Parse LSB_MCPU_HOSTS to get GPU and CPU hosts
CPU_HOSTS=""
GPU_HOSTS=""
echo "LSB_MCPU_HOSTS: $LSB_MCPU_HOSTS"
re='^[0-9]+$'
for loop in $LSB_MCPU_HOSTS
do
    if [[ ${loop:3:3} == 'alx' || ${loop:3:3} == 'hlx' ]]
    then
        myhost='CPU'
        myname=$loop
    elif [[ ${loop:3:3} == 'glx' ]]
    then
        myhost='GPU'
        myname=$loop
        GPUname=$myname
        MPI_PREFIX=/mktoss/openmpi/4.0.3.ubuntu22/x86-64
    elif [[ ${loop:3:3} == 'hgx' ]]
    then
        myhost='GPU'
        myname=$loop
        GPUname=$myname
        MPI_PREFIX=/mktoss/openmpi/4.0.3.ubuntu22/x86-64
    elif [[ ${loop:3:3} == 'ur' ]]
    then
        myhost='CPU'
        myname=$loop
        MPI_PREFIX=/mktoss/openmpi/4.0.3.ubuntu22/x86-64
    fi

    if [[ $loop =~ $re ]]
    then
        number=$loop
        if [[ $myhost == 'CPU' ]]
        then
            if [ -z "${CPU_HOSTS// }" ]
            then
                CPU_HOSTS="$myname:$number"
            else
                CPU_HOSTS="$CPU_HOSTS,$myname:$number"
            fi
        elif [[ $myhost == 'GPU' ]]
        then
            if [ -z "${GPU_HOSTS// }" ]
            then
                GPU_HOSTS="$myname:$number"
                masternode="$myname"
            else
                GPU_HOSTS="$GPU_HOSTS,$myname:$number"
            fi
        fi
    fi
done

echo "CPU: $CPU_HOSTS, GPU:$GPU_HOSTS"
nproc_per_node=$number
nnodes=$(echo $GPU_HOST | tr "," "\n" | wc -l)
masterport="9487"
rdzv_id="$RANDOM"

GPU_cores=$(($nproc_per_node * $nnodes))

PYTHON_SCRIPT="trainer.py fit --config configs/train_ew.xfrm.yml"
GPU_SCRIPT="torchrun --nnodes=$nnodes --nproc_per_node=$nproc_per_node --max_restart=3 --rdzv_id $rdzv_id --rdzv-backend=c10d --rdzv-endpoint=$myname:$masterport $PYTHON_SCRIPT"

# Define MPI options
HOST_N_COMMAND=$(echo $GPU_HOST | sed 's/[0-9]\+//1g')
MPI_SHARED_OPTIONS="-x PATH -x LD_LIBRARY_PATH --tag-output --bind-to none -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0 -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_DEBUG=INFO --report-bindings"
MPI_NON_SHARED_OPTIONS="--map-by node -n $nodesH -host $HOST_N_COMMAND"
MPI_PIN_SHARED_OPTIONS="--map-by slot -np $GPU_core -H $GPU_HOST -x MASTER_ADDR=$myname -x MASTER_PORT=$masterport"

echo -e "HOST_N COMMAND submitted:\n$HOST_N_COMMAND"
USER_CMD="$GPU_SCRIPT"
echo -e "USER CMD submitted:\n$USER_CMD"
MPI_OPTIONS="$MPI_SHARED_OPTIONS $MPI_NON_SHARED_OPTIONS"
echo -e "MPI OPTIONS submitted:\n$MPI_OPTIONS"
echo -e "MPI_PREFIX submitted:\n$MPI_PREFIX"

# Execute with mpirun
mpirun --prefix $MPI_PREFIX $MPI_OPTIONS $FINAL_SCRIPT
mpirun --prefix $MPI_PREFIX $MPI_OPTIONS $PODMAN_SCRIPT
mpirun --prefix $MPI_PREFIX $MPI_OPTIONS $GPU_SCRIPT
END_SCRIPT

echo "LSF script created successfully"
echo "Script contents:"
head -20 $lsf_script

# Submit the job
echo "Submitting multinode training job..."
bsub < $lsf_script

# Clean up temporary file
echo "Cleaning up temporary file..."
rm -f $lsf_script

echo "Job submitted successfully!" 