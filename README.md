# Kevin


This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

To install dependencies simply run
```
python3 -m venv venv
source venv/bin/activate
sudo apt install \
  git libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev libglfw3 libgl1-mesa-dev libglu1-mesa-dev \
  python3-pyaudio portaudio19-dev ffmpeg flac python3.8-venv python3-pip
pip install -r requirements.txt
```
in an environment of your choosing.



for Jetson had to deal with versions
```
pip install numba==0.54


sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons libopenblas-dev

export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

python3 -m pip install --upgrade pip; python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3' export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --no-cache $TORCH_INSTALL


```
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.

```
        sense     (sensor input)
      perceive    (deep learning encode)
    emote         (calculate internal low level motivations like battery, connect, help )
  concern         (attention over world model with possible futures)
trust             (core policy. score futures. select goal)
  act             (evaluate actions to acheive goal)
    try           (select behavior)
      orchestrate (track progress and emit actions)
        react     (actuators output)
```


def invert()
spectator


tick
lambda: #Root
  lambda: #Chooser
    SensePainCondition() and HandlePainAction() or\
    IsMovingCondition()  and  or\
    FindMove()
