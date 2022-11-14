# syntax=docker/dockerfile:1.2
# use me with:
#   cd tools/dockerfiles
#   docker build --pull -t ${DST_IMAGE} -f dockerfile.contribute.sxn.tf --build-arg RELEASE=false --build-arg RMM_VER=vnightly --build-arg CUDF_VER=vnightly --build-arg NVTAB_VER=vnightly --build-arg HUGECTR_DEV_MODE=true --no-cache .
#   docker run --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host -v /home/songxiaoniu/hugectr_dev:/hugectr_dev -v /nvme:/nvme --name sxn-dev -it tf-plugin zsh

# ARG MERLIN_VERSION=22.08
# ARG TRITON_VERSION=22.07
ARG TENSORFLOW_VERSION=22.07

ARG RELEASE=false
# ARG RMM_VER=vnightly
# ARG CUDF_VER=vnightly
# ARG NVTAB_VER=vnightly
ARG HUGECTR_DEV_MODE=true 

# ARG DLFW_IMAGE=nvcr.io/nvidia/tensorflow:${TENSORFLOW_VERSION}-tf2-py3
# ARG FULL_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
# ARG BASE_IMAGE=nvcr.io/nvstaging/merlin/merlin-base:${MERLIN_VERSION}
ARG BASE_IMAGE=nvcr.io/nvidia/merlin/merlin-tensorflow:22.07

# FROM ${DLFW_IMAGE} as dlfw
# FROM ${FULL_IMAGE} as triton
FROM ${BASE_IMAGE} as base

ENV http_proxy="http://10.1.8.10:34560"
ENV https_proxy="http://10.1.8.10:34560"
ENV all_proxy="http://10.1.8.10:34560"
ENV HTTP_PROXY="http://10.1.8.10:34560"
ENV HTTPS_PROXY="http://10.1.8.10:34560"
ENV ALL_PROXY="http://10.1.8.10:34560"

ENV LANG=en_US.UTF-8
ENV USER=root

# # Triton TF backends
# COPY --chown=1000:1000 --from=triton /opt/tritonserver/backends/tensorflow2 backends/tensorflow2/

# # Tensorflow dependencies (only)
# # Pinning to pass hugectr sok tests
# RUN pip install tensorflow-gpu==2.9.2 \
#     && pip uninstall tensorflow-gpu keras -y

# # DLFW Tensorflow packages
# COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/tensorflow /usr/local/lib/python3.8/dist-packages/tensorflow/
# COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/keras /usr/local/lib/python3.8/dist-packages/keras/
# COPY --chown=1000:1000 --from=dlfw /usr/local/lib/tensorflow/ /usr/local/lib/tensorflow/
# COPY --chown=1000:1000 --from=dlfw /usr/local/lib/python3.8/dist-packages/horovod /usr/local/lib/python3.8/dist-packages/horovod/
# COPY --chown=1000:1000 --from=dlfw /usr/local/bin/horovodrun /usr/local/bin/horovodrun

# Install dependencies for hps tf plugin
RUN chmod 1777 /tmp
RUN apt update -y --fix-missing && \
    apt install -y --no-install-recommends \
        #   Required to build RocksDB.
            libgflags-dev \
            zlib1g-dev libbz2-dev libsnappy-dev liblz4-dev libzstd-dev \
        #   Required to build RdKafka.
            zlib1g-dev libzstd-dev \
            libssl-dev libsasl2-dev && \
    apt install -y --no-install-recommends zsh tmux htop &&\
    apt clean && \
    rm -rf /var/lib/apt/lists/*


# Install HugeCTR
ENV LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    SOK_COMPILE_UNIT_TEST=ON

# Arguments "_XXXX" are only valid when $HUGECTR_DEV_MODE==false
ARG HUGECTR_DEV_MODE=true
# ARG _HUGECTR_REPO="github.com/NVIDIA-Merlin/HugeCTR.git"
ARG _CI_JOB_TOKEN=""
ARG HUGECTR_VER=master

# RUN mkdir -p /usr/local/nvidia/lib64 && \
#     ln -s /usr/local/cuda/lib64/libcusolver.so /usr/local/nvidia/lib64/libcusolver.so.10

# RUN ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so

RUN pip install ninja numba

# COPY . /hugectr_dev

# RUN if [ "$HUGECTR_DEV_MODE" == "false" ]; then \
#         git clone --branch ${HUGECTR_VER} --depth 1 https://${_CI_JOB_TOKEN}${_HUGECTR_REPO} /hugectr && \
#         pushd /hugectr && \
# 	pip install ninja && \
# 	git submodule update --init --recursive && \
#         # Install SOK
#         cd sparse_operation_kit && \
#         python setup.py install && \
#         # Install HPS TF plugin
#         cd ../hierarchical_parameter_server && \
#         python setup.py install && \
#         popd; \
#     fi

# # Install distributed-embeddings
# ARG INSTALL_DISTRIBUTED_EMBEDDINGS=true
# RUN if [ "$INSTALL_DISTRIBUTED_EMBEDDINGS" == "true" ]; then \
#         git clone https://github.com/NVIDIA-Merlin/distributed-embeddings.git /distributed_embeddings/ && \
#         cd /distributed_embeddings && git checkout ${TFDE_VER} && \
#         make pip_pkg && pip install artifacts/*.whl && make clean; \
#     fi
