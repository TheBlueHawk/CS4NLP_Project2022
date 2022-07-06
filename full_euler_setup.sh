# Usage: sh full_euler_setup.sh <username> <sweepID>

cd ..

cd ..

cd scratch/$1

env2lmod

module load gcc/8.2.0

module load python_gpu/3.8.5

module load eth_proxy

python -m venv venv 

source venv/bin/activate 

pip install -r requirements.txt

wandb login 

bsub -W 23:00 -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" wandb agent --count 25 $2