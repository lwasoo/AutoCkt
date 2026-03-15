# Build:
#   docker build -t autockt:py310-centos7 .
# Run:
#   docker run -it --name autockt-py310 -p 6006:6006 autockt:py310-centos7
# TensorBoard inside container:
#   tensorboard --logdir /root/ray_results --host=0.0.0.0 --port=6006

FROM centos:7

# CentOS7 mirror fix (pin to TUNA centos-vault for better stability in CN networks)
RUN printf '%s\n' \
'[base]' \
'name=CentOS-7.9.2009 - Base' \
'baseurl=https://mirrors.tuna.tsinghua.edu.cn/centos-vault/7.9.2009/os/$basearch/' \
'gpgcheck=1' \
'enabled=1' \
'gpgkey=https://mirrors.tuna.tsinghua.edu.cn/centos-vault/7.9.2009/os/$basearch/RPM-GPG-KEY-CentOS-7' \
'' \
'[updates]' \
'name=CentOS-7.9.2009 - Updates' \
'baseurl=https://mirrors.tuna.tsinghua.edu.cn/centos-vault/7.9.2009/updates/$basearch/' \
'gpgcheck=1' \
'enabled=1' \
'gpgkey=https://mirrors.tuna.tsinghua.edu.cn/centos-vault/7.9.2009/os/$basearch/RPM-GPG-KEY-CentOS-7' \
'' \
'[extras]' \
'name=CentOS-7.9.2009 - Extras' \
'baseurl=https://mirrors.tuna.tsinghua.edu.cn/centos-vault/7.9.2009/extras/$basearch/' \
'gpgcheck=1' \
'enabled=1' \
'gpgkey=https://mirrors.tuna.tsinghua.edu.cn/centos-vault/7.9.2009/os/$basearch/RPM-GPG-KEY-CentOS-7' \
> /etc/yum.repos.d/CentOS-Base.repo \
    && rpm --import https://mirrors.tuna.tsinghua.edu.cn/centos-vault/7.9.2009/os/x86_64/RPM-GPG-KEY-CentOS-7 \
    && echo "retries=10" >> /etc/yum.conf \
    && echo "timeout=120" >> /etc/yum.conf \
    && echo "minrate=1" >> /etc/yum.conf \
    && yum clean all \
    && yum makecache

# Base build dependencies
RUN yum install -y --setopt=timeout=120 --setopt=retries=10 \
    wget \
    tar \
    make \
    gcc \
    gcc-c++ \
    perl \
    zlib-devel \
    bzip2-devel \
    xz-devel \
    libffi-devel \
    readline-devel \
    sqlite-devel \
    ncurses-devel \
    ca-certificates \
    && yum clean all

# Build OpenSSL 1.1.1 (Python 3.10 requires newer OpenSSL than CentOS7 default)
ENV OPENSSL_VERSION=1.1.1w
RUN cd /tmp \
    && wget https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz \
    && tar -xzf openssl-${OPENSSL_VERSION}.tar.gz \
    && cd openssl-${OPENSSL_VERSION} \
    && ./config --prefix=/opt/openssl --openssldir=/opt/openssl shared zlib \
    && make -j"$(nproc)" \
    && make install_sw \
    && rm -rf /tmp/openssl-${OPENSSL_VERSION}*

# Build Python 3.10 from source
ENV PYTHON_VERSION=3.10.14
RUN cd /tmp \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && LD_RUN_PATH=/opt/openssl/lib ./configure \
        --prefix=/opt/python/${PYTHON_VERSION} \
        --with-openssl=/opt/openssl \
        --with-openssl-rpath=auto \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/Python-${PYTHON_VERSION}*

ENV PATH=/opt/python/${PYTHON_VERSION}/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/openssl/lib:$LD_LIBRARY_PATH

# Install NGSPICE 2.7
RUN cd /tmp \
    && wget https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/27/ngspice-27.tar.gz \
    && tar -xzf ngspice-27.tar.gz \
    && cd ngspice-27 \
    && ./configure --prefix=/usr/local --enable-xspice --disable-debug --without-x \
    && make -j"$(nproc)" \
    && make install \
    && cd /tmp \
    && rm -rf ngspice-27*

# Project
WORKDIR /app/AutoCkt
COPY . /app/AutoCkt

# venv + python deps
RUN PYTHON_BIN=python3.10 VENV_DIR=/opt/venv INSTALL_TORCH_CPU=1 \
    bash /app/AutoCkt/scripts/setup_venv.sh \
    && /opt/venv/bin/pip check
ENV PATH=/opt/venv/bin:$PATH

# Optional runtime dir for ray outputs
RUN mkdir -p /root/ray_results
ENV RAY_DISABLE_DASHBOARD=1
ENV PYTHONPATH=/app/AutoCkt

COPY scripts/docker_entrypoint.sh /usr/local/bin/docker_entrypoint.sh
COPY scripts/cleanup_cktda.sh /usr/local/bin/cleanup_cktda.sh
RUN chmod +x /usr/local/bin/docker_entrypoint.sh /usr/local/bin/cleanup_cktda.sh

ENTRYPOINT ["/usr/local/bin/docker_entrypoint.sh"]
CMD ["/bin/bash"]
