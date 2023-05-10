# gym-rltrading


## Installation

### Prerequisites

This repository requires [Baselines](https://github.com/openai/baselines), which requires python3 (>=3.5) with the development headers.
You'll also need system packages CMake, OpenMPI and zlib.
Those can be installed as follows

#### Ubuntu 
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

### Install Baselines

- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, you may use
    ```bash 
    pip install tensorflow-gpu==1.14 # if you have a CUDA-compatible gpu and proper drivers
    ```
    or 
    ```bash
    pip install tensorflow==1.14
    ```
    to install Tensorflow 1.14, which is the latest version of Tensorflow supported by the master branch. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install baselines package
    ```bash
    pip install -e .
    ```

### Install additional requirements using pip
```bash
pip install -r requirements.txt
```


## Usage

Test logger functionality
```bash
python main.py --mode=test
```

Train SMA Cross Strategy on BTC/USDT  
Make sure you check `htop` and `nvidia-smi` before running below. The GPU device can be set by defining the environment variable `CUDA_VISIBLE_DEVICES` as the number of the device, `[GPU_NUM]`.
```bash
CUDA_VISIBLE_DEVICES=[GPU_NUM] python main.py --mode=train --n_env=4 --env_id=btc/usdt-smacross-v0 --total_step=100000
```
  
### Use tensorboard to monitor your training progress in remote-server!  
If you're using VSCode, you just need to run tensorboard on remote server via VSC terminal.
VSC will automatically forward port for you.
```bash
tensorboard --logdir [LOG_DIR] --port [PORT_NUM]
```
Default log_dir is [./logs] and port number is [6006].
You have to choose 4 digits number if the port is already taken.
```bash
tensorboard --logdir ./logs --port 6006
```
You can see your tensorboard in  
http://localhost:6006/  
  
If you're not using VSC, you can mannually forward port when accessing ssh or just set it in your local ssh config.  
On your local device,
```bash
vi ~/.ssh/config
```
add below on your config
```bash
Host tensorboard # Can be any name you want to call
        HostName [remote_server_address]
        User [username]
        Port [port]
        LocalForward [forward_port] localhost:[forward_port]
```
Then you will be able to access to remote server via
```bash
ssh tensorboard
```
Run tensorboard,
```bash
tensorboard --logdir [LOG_DIR] --port [PORT_NUM]
```
You can see your tensorboard in  
http://localhost:6006/  
  
See more details in  
https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server

### Using --tag command
You can give a tag when you run script via
```bash
python main.py --mode=train --env_id=btc/usdt-smacross-v0 --tag [tag_name]
```
This will give a prefix on your log file as [tag]/[datetime]. (e.g. foo/20210123015919)  
It will store all log files with same tag in [./logs/foo] dir.  
This can be useful to recognize which log file you should monitor in tensorboard.  
Also, you can manage your log directory by using multiple tags.  
```bash
python main.py --mode=train --env_id=btc/usdt-smacross-v0 --tag [tag_1]/[tag_2]
```

#### Example  
Train foo strategy in btc and eth market.  
```bash
python main.py --mode=train --env_id=btc/usdt-foo-v0 --tag foo/btc
python main.py --mode=train --env_id=eth/usdt-foo-v0 --tag foo/eth
```
Train bar strategy in btc and eth market.  
```bash
python main.py --mode=train --env_id=btc/usdt-bar-v0 --tag bar/btc
python main.py --mode=train --env_id=eth/usdt-bar-v0 --tag bar/eth
```

You can enjoy clean tensorboard by specifing [log_dir] when you activate the tensorboard.
```bash
tensorboard --logdir ./logs/foo
```
Only foo/[mkt]/[datetime] will be visible.
```bash
tensorboard --logdir ./logs/bar
```
Only bar/[mkt]/[datetime] will be visible.