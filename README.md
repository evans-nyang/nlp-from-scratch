# PyTorch Fundamentals

## Issues

When trying to run torch with the initial dependency versions as below:

```txt
torch==2.1.2+rocm5.6
torchvision==0.16.2+cu121
```

I got the following error:

```shell
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
Cell In[1], line 1
----> 1 import torch
      2 import numpy as np

File ~/development/mlenv/lib/python3.10/site-packages/torch/__init__.py:237
    235     if USE_GLOBAL_DEPS:
    236         _load_global_deps()
--> 237     from torch._C import *  # noqa: F403
    239 # Appease the type checker; ordinarily this binding is inserted by the
    240 # torch._C module initialization code in C
    241 if TYPE_CHECKING:

ImportError: libcudnn.so.8: cannot open shared object file: No such file or directory
```

The error indicates that the system cannot locate libcudnn.so.8, which is a critical library for CUDA-based operations in PyTorch. The error is likely due to the fact that the CUDA version on your linux system is different from the one used to compile the PyTorch version in the requirements.txt file.

## Solution

Check the CUDA version on your system by running the following command.

```shell
nvcc --version
```

Check if libcudnn.so.8 exists in your system.

```shell
ls /usr/local/cuda/lib64 | grep libcudnn
```

If libcudnn.so.8 does not exist, you need to install the correct version of libcudnn for your CUDA version. You can download the libcudnn library from the [NVIDIA cuDNN Archive](https://developer.nvidia.com/cudnn-archive) and install it following the instructions provided in the libcudnn documentation.

Installation steps:

1. Download the CuDNN .deb Package. You can download the CuDNN .deb package from the NVIDIA cuDNN Archive. The package name should be similar to `cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb`.

2. Install the .deb Package. Use dpkg to install the `.deb` file.

```sh
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
```

3. Add the GPG Key for the repository to your system. You can do this by copying the keyring file to the `/usr/share/keyrings/` directory.

```sh
sudo cp /var/cudnn-local-repo-*/cudnn-local-*.gpg /usr/share/keyrings/
```

4. Update the package list to include the new repository.

```sh
sudo apt-get update
```

5. Install CuDNN Libraries Install the runtime, developer, and documentation libraries.

```sh
sudo apt-get install libcudnn8 libcudnn8-dev libcudnn8-samples
```

6. Verify that the CuDNN libraries have been installed correctly.

```sh
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

7. Check if `libcudnn.so.8` is available.

```sh
ls /usr/lib/x86_64-linux-gnu/ | grep libcudnn
```

**Note:** The above process applies only when you don't intend to use image functionality from torchvision.io. Else you will encounter an exception.

You will encounter this exception when running the initial code cell:

```shell
/home/mine/development/mlenv/lib/python3.10/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 'libc10_cuda.so: cannot open shared object file: No such file or directory'If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
  warn(
```

## Workaround Solution

The error above indicates that torchvision is trying to load its image-handling extension, but it failed because the CUDA-related shared library `libc10_cuda.so` could not be loaded.

This is likely due to one of the following reasons:

- CUDA Library Not Found:
The library `libc10_cuda.so` is part of PyTorch's CUDA backend. If you don't have a proper CUDA installation or your PyTorch is not installed with GPU support, this error may occur.

- Dependencies Missing:
The warning also mentions `libjpeg` and `libpng`. These are libraries used for handling image files. If they were not present during the build of torchvision (or are not installed now), the error could occur.

- Environment Issues:
If the environment where you're running the code isn't properly set up (e.g., conflicting packages or incorrect paths), it can also cause the error.

Check your cuda version:

```sh
nvcc --version
```

Install the correct cuda version from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive). Preferrably CUDA 11.8 / 12.1 / 12.4 which are compatible with the stable versions of torch, torchaudio and torchvision as documented in this [link](https://pypi.org/project/torch/2.5.1/#nvidia-cuda-support).

You can [select a supported version of CUDA from our support matrix here.](https://pytorch.org/get-started/locally/)

**Tip:** To install the correct version of CUDA (CUDA Toolkit 12.4), follow the instructions provided in the [CUDA Installation Guide](https://developer.nvidia.com/cuda-12-4-0-download-archive).

The installation process might get stuck or fail because of incomplete package retrieval or unmet dependencies when running the command `sudo apt-get -y install cuda-toolkit-12-4`. Here’s how you can troubleshoot and resolve this issue:

1. Clear the package manager's cache to remove any corrupted files.

```sh
sudo apt-get clean
sudo apt-get autoremove
```

2. Update the package list to ensure the system is aware of the latest available packages.

```sh
sudo apt-get update
```

3. Install missing dependencies by manually attempting to fix any broken dependencies.

```sh
sudo apt-get install -f
```

4. Retry the installation of the CUDA Toolkit 12.4 . Ensure your internet connection is stable.

```sh
sudo apt-get -y install cuda-toolkit-12-4
```

5. Verify the installation by checking the CUDA version.

```sh
nvcc --version
```

Install dependencies from [requirements.txt](requirements.txt) file using the command below:

```sh
pip install -r requirements.txt
```

## Extras

To remove the cuda insallation:

1. List cuda installation.

```sh
dpkg -l | grep cuda
```

2. Run the following command to remove all CUDA-related packages.

```sh
sudo apt-get --purge remove '*cuda*' '*nvidia*'
```

3. Remove any leftover configuration files and directories.

```sh
sudo rm -rf /usr/local/cuda*
```

4. Update your package lists and clean up any unnecessary dependencies.

```sh
sudo apt-get autoremove
sudo apt-get autoclean
```

5. Verify Removal, run this command to ensure nvcc is no longer available.

```sh
nvcc --version
```

## Documentation Guides

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN Archive](https://developer.nvidia.com/cudnn-archive)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
