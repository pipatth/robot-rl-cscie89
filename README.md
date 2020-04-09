# CSCI-E89 Deep Learning: Final Project
# Case Study in Robotics

### Installation


#### Install dependencies 

- Follow the instruction to install NVIDIA CUDA here: https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_101
- Install the following packages in Ubuntu
- Create a conda environment and install the packages from Anaconda and pip
```
apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev wget bzip2 git patchelf ffmpeg
conda env create -f environment.yml
pip install -r requirements.txt
```

#### Install Mujoco
- Download Mujoco 200 from https://www.roboti.us/index.html 
- Unzip the downloaded mjpro200 directory into ~/.mujoco/mjpro200 
- Place your license key (the mjkey.txt file from your email) at ~/.mujoco/mjkey.txt.
- Run the following code to install mujoco-py
```
git clone https://github.com/openai/mujoco-py
cd mujoco-py
pip install .
```
- Append the following lines to .bashrc 
```
export MUJOCO_HOME=$HOME/.mujoco/mjpro200
export LD_LIBRARY_PATH=$MUJOCO_HOME/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384
```

#### How to run
Due to the bug in mujoco-py when running with NVIDIA GPU, the workaround is to prepend the following before running a python command
```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so python fetchreach.py
```
