FROM tensorflow/tensorflow:2.4.1-gpu

WORKDIR /root/

COPY requirements.txt .

# RUN apt-get update \
#     && apt-get install -y wget gnupg2 \
#     && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/InRelease | apt-key add - \
#     && apt-get update \
#     && apt-get upgrade \
#     && apt-get install -y \
#     sudo \
#     ffmpeg \
#     cmake \
#     git \
#     && apt-get clean \
#     && apt-get autoremove -y \
#     && rm -rf /var/lib/apt/lists/*

# RUN python -m pip install --upgrade pip

# RUN pip install -r requirements.txt