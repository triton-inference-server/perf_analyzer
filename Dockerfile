ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.08-py3-min
FROM ${BASE_IMAGE} AS build_base

RUN DEBIAN_FRONTEND="noninteractive" apt-get update -qq && \
      apt-get install -y \
        build-essential \
        libb64-dev \
        libcurl4-openssl-dev \
        python3-dev \
        python3-pip \
        rapidjson-dev

RUN pip3 install cmake==3.30.2 ninja==1.11.1.1

FROM build_base AS cache

RUN curl -k -L -O https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz \
        && tar -xf ccache-4.10.2-linux-x86_64.tar.xz \
        && mv ccache-4.10.2-linux-x86_64 /usr/local/ \
        && ln -s /usr/local/ccache-4.10.2-linux-x86_64/ccache /usr/local/bin/ccache

ARG CCACHE_REMOTE_STORAGE

ENV CMAKE_CXX_COMPILER_LAUNCHER="ccache"
ENV CMAKE_C_COMPILER_LAUNCHER="ccache"
ENV CCACHE_DEBUG=1

RUN [[ -v CCACHE_REMOTE_STORAGE ]] && \
      ccache --set-config=remote_only=true ; \
      ccache --set-config=remote_storage=${CCACHE_REMOTE_STORAGE} ; \
      ccache --set-config=log_file=/tmp/ccache.log ;

RUN ccache -p

FROM cache AS build

ARG GRPC_VERSION=v1.54.3
RUN git clone --recursive -b ${GRPC_VERSION} https://github.com/grpc/grpc.git /tmp/grpc
RUN cmake \
      -G Ninja \
      -DgRPC_INSTALL=ON \
      -Dprotobuf_INSTALL=ON \
      -Dabseil_INSTALL=ON \
      -Dre2_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -S /tmp/grpc -B /tmp/grpc/cmake/build
RUN cmake --build /tmp/grpc/cmake/build/ -t install

WORKDIR /perf_analyzer
COPY . .
ARG TRITON_COMMON_REPO_TAG=main
ARG TRITON_CLIENT_REPO_TAG=main
ARG TRITON_CORE_REPO_TAG=main

RUN cmake \
    -G Ninja \
    -DTRITON_COMMON_ENABLE_PROTOBUF=ON \
    -DTRITON_COMMON_ENABLE_GRPC=ON \
    -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
    -DTRITON_CLIENT_REPO_TAG=${TRITON_CLIENT_REPO_TAG} \
    -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCMAKE_PREFIX_PATH="/usr/include/$(uname -m)-linux-gnu;/usr/lib/$(uname -m)-linux-gnu" \
    -S src \
    -B /tmp/build
RUN cmake --build /tmp/build --target install -v

WORKDIR /perf_analyzer/genai-perf
RUN pip3 wheel .

WORKDIR /workspace/artifacts/
RUN cp /perf_analyzer/genai-perf/genai_perf*whl . && \
    cp /tmp/build/perf_analyzer* .

RUN tar -czvf perf_analyzer-${BUILD_ARCH}.tar.gz *perf*
