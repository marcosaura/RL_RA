#!/bin/bash --login
#$ -cwd
#$ -l v100=2         # A 4-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)
#$ -pe smp.pe 16      # Our MPI code only uses one MPI process (hence one CPU core) per GPU
#$ -l s_rt=24:00:00
#$ -ac nvmps
# MPI library (which also loads the cuda modulefile)
module load libs/cuda
module load tools/gcc/cmake/3.23.0
module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
module load compilers/gcc/9.3.0
module avail libs/cuDNN
module avail libs/nccl
echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"

set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
## In this example we start one MPI process per GPU. We could use $NSLOTS or $NGPUS (both = 4)
# It is assume the application will ensure each MPI process uses a different GPU. For example
# MPI rank 0 will use GPU 0, MPI rank 1 will use GPU 1 and so on

#Install dependencies
pip3 install --upgrade pip --user && pip3 install --upgrade setuptools --user && pip3 install opencv-python --user && pip3 install -r ./EfficientZero_RA/requirements.txt --user && python3 -m atari_py.import_roms ./Roms/ && cd ./EfficientZero_RA/core/ctree && bash make.sh 
#Run benchmark environment
python3 ./main.py --env AsterixNoFrameskip-v4 --case atari --opr train --amp_type torch_amp --use_max_priority --use_priority