# Base Image
ARG PARENT_IMAGE
FROM $PARENT_IMAGE

# Args need to be below FROM!
ARG USER_ID
ARG GROUP_ID
ARG PYTHON_VERSION=3.10

ENV NVIDIA_DRIVER_CAPABILITIES all

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    ffmpeg \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    # lib for SAPIEN rendering
    libglvnd-dev \
    && rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# install Simpler env
WORKDIR /
RUN git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
WORKDIR SimplerEnv
# install dependencies
RUN pip install tensorflow==2.15.0
COPY requirements_full_install.txt ./
RUN pip install tensorflow[and-cuda]==2.15.1
RUN pip install git+https://github.com/nathanrooy/simulated-annealing
# install simpler_env
RUN pip install -e ./ManiSkill2_real2sim
RUN pip install -r requirements_full_install.txt
RUN pip install -e .

# https://github.com/haosulab/ManiSkill/issues/9
COPY docker/simpler/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/simpler/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY docker/simpler/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# rlds converter
# WORKDIR /
# RUN git clone https://github.com/kpertsch/rlds_dataset_builder.git
# WORKDIR rlds_dataset_builder
# RUN conda env create -f environment_ubuntu.yml

# install gsutil
# Add the Google Cloud SDK distribution URI as a package source
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# Import the Google Cloud public key
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
# Update and install the Google Cloud SDK (includes gsutil)
RUN apt-get update && apt-get install -y google-cloud-sdk

# install LIBERO
WORKDIR /
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
WORKDIR LIBERO
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install robosuite==1.4.1 bddl easydict
RUN pip install -e .

# install our package
RUN pip install --upgrade pip
# RUN pip install --upgrade "jax[cuda12]"
RUN pip uninstall -y jax jaxlib
RUN pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
COPY ./ /user/hypervla
WORKDIR /user/hypervla
RUN pip install -e .

# Change permissions
RUN groupadd -g ${GROUP_ID} user
RUN useradd --shell /bin/bash -u ${USER_ID} -g ${GROUP_ID} -o -d /user user
RUN chown -R user:user /user && chmod -R u+w /user
# RUN chown -R user:user / && chmod -R u+w /

# Set python ENV variables
ENV PYTHONUNBUFFERED=1