# Usage: sh quick_euler_setup.sh

env2lmod

module load gcc/8.2.0

module load python_gpu/3.8.5

module load eth_proxy 

source venv/bin/activate 

pip install -r requirements.txt

wandb login 