export CONDA_ALWAYS_YES=true
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true


ENV_NAME="spec" 

source ~/.bashrc

source activate base

if conda env list | grep -q "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' found. Removing..."
  conda env remove --name ${ENV_NAME} -y
  echo "Conda environment '${ENV_NAME}' removed."
else
  echo "Conda environment '${ENV_NAME}' not found. No action taken."
fi

conda create --name ${ENV_NAME} python=3.10 
source activate ${ENV_NAME}

# Show current conda environment
conda info --envs

conda install -c conda-forge libstdcxx-ng --update-deps


pip install --upgrade pip
pip install tqdm==4.67.1
pip install hydra-core==1.3.2
pip install datasets==4.0.0
pip install transformers==4.55.2
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install wandb==0.21.1
pip install math-verify==0.8.0
pip install vllm==0.10.1
pip install trl==0.21.0
pip install peft==0.17.0
pip install accelerate==1.10.0
pip install deepspeed==0.17.4