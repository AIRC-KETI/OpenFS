conda create -n openfs python=3.10 -y
conda activate openfs

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt