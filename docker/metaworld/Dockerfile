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
RUN pip install -r requirements_full_install.txt
RUN pip install tensorflow[and-cuda]==2.15.1

# install meta world
WORKDIR /
RUN pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld

# install our package
RUN pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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