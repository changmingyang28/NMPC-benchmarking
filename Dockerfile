FROM ubuntu:focal
LABEL Description="Build environment"

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME /root

SHELL ["/bin/bash", "-c"]

## Essentials
RUN apt-get update && apt-get -y --no-install-recommends install \
    build-essential \
    clang \
    cmake cmake-gui\
    gdb \
    wget \
    git \
    curl \
    nano \
    llvm\
    libgtest-dev\
    && apt-get clean

## Allow GIT clone
RUN apt-get install -y --reinstall ca-certificates

## Python3
RUN apt-get update && apt-get install -y python3.8 python3.8-tk &&\
    apt-get install -y pip

## install Python dependencies via pip
RUN python3 -m pip install numpy matplotlib scipy ipykernel meshcat

## construct direcotry for dependencies
RUN cd ${HOME} &&\
    mkdir workspace

ENV DEP_DIR ${HOME}/workspace

## install eigen
RUN apt update && apt install -y libeigen3-dev libboost-all-dev liburdfdom-dev && apt-get clean

## Define symbolic link for Eigen library https://stackoverflow.com/questions/23284473/fatal-error-eigen-dense-no-such-file-or-directory
RUN cd /usr/include &&\
    ln -sf eigen3/Eigen Eigen &&\
    ln -sf eigen3/unsupported unsupported

## install CPPAD
RUN cd ${DEP_DIR} &&\
    git clone --recursive https://github.com/coin-or/CppAD.git cppad &&\
    cd cppad && mkdir build && cd build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fPIC" &&\
    make && make install

## install CPPAD code gen
RUN cd ${DEP_DIR} &&\
    git clone https://github.com/joaoleal/CppADCodeGen.git CppADCodeGen &&\
    cd CppADCodeGen && mkdir build && cd build &&\
    cmake .. && make && make install

RUN apt install -qqy lsb-release gnupg2 curl &&\
    echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | tee /etc/apt/sources.list.d/robotpkg.list &&\
    curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | apt-key add - &&\
    apt update &&\
    apt install -qqy robotpkg-py38-eigenpy robotpkg-py38-qt5-gepetto-viewer-corba

ENV PATH=/opt/openrobots/bin:$PATH
ENV PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH 
ENV CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH

## Install Pinocchio from source
RUN cd ${DEP_DIR} &&\
    git clone --recursive https://github.com/stack-of-tasks/pinocchio &&\
    cd pinocchio  &&\
    git checkout master &&\
    mkdir build && cd build

RUN cd ${DEP_DIR}/pinocchio/build &&\
    cmake .. \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \ 
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DBUILD_WITH_CODEGEN_SUPPORT=ON

RUN cd ${DEP_DIR}/pinocchio/build &&\
    make -j4 && make install

# define path for cmake and python
ENV PATH=/usr/local/bin:$PATH
ENV PKG_CONFIG_PATH =/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:$PYTHONPATH
ENV CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH

# install robotoc
RUN cd ${DEP_DIR}&&\
    git clone https://github.com/mayataka/robotoc robotoc &&\
    cd robotoc &&\
    mkdir build && cd build 

RUN cd ${DEP_DIR}/robotoc/build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release -DOPTIMIZE_FOR_NATIVE=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 &&\
    make -j4 && make install

## TODO move up front
RUN apt install -qqy lsb-release gnupg2 curl &&\
    echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | tee /etc/apt/sources.list.d/robotpkg.list &&\
    curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | apt-key add - &&\
    apt update &&\
    apt install -qqy robotpkg-py38-example-robot-data

# install corcoddyl
RUN cd ${DEP_DIR} &&\
    git clone --recursive https://github.com/loco-3d/crocoddyl.git crocoddyl &&\
    cd crocoddyl && git checkout master &&\
    mkdir build

RUN cd ${DEP_DIR}/crocoddyl/build &&\
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPYTHON_EXECUTABLE=/usr/bin/python3\ 
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARK=OFF \
    -DBUILD_WITH_IPOPT=OFF

RUN cd ${DEP_DIR}/crocoddyl/build &&\
    make -j2 && make install

## minor adjustments 
# TODO place upfront
RUN apt update && apt-get install -y locales &&\
    locale-gen en_US en_US.UTF-8 &&\
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV NO_AT_BRIDGE=1

# TODO place after robotoc installation
ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages:$PYTHONPATH 