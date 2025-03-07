# 构建镜像指令：docker build -t autockt .
# 首次运行创建容器：docker run -it -p 6006:6006 --name autockt-container autockt:latest
# (创建容器并挂载到本地目录：docker run -it -p 8080:8080 -v <本地路径>(如 C:\Users\ASUS\Documents\project):<容器路径>(/app) --name autockt-container-mount autockt:latest)
# 再次进入或容器未开启 先启动容器：docker start autockt-container
# 再次进入容器：docker exec -it autockt-container /bin/bash
# 退出容器：exit
# 删除容器：docker rm -f autockt-container
# 删除镜像：docker rmi -f autockt

# M芯片目前不可用
# M芯片：docker build --platform linux/amd64 -t autockt .
# M芯片运行镜像：docker run --platform linux/amd64 -it --name autockt-container autockt:latest

# （MARK: 长命令反斜杠后面不要加注释，docker会解析错误）

# tensorboard --logdir /root/ray_results/train_45nm_ngspice/ --host=0.0.0.0 --port=6006
# 外部启用：http://localhost:6006

# 使用 CentOS 7 基础镜像，并安装 Miniconda
FROM centos:7

# 安装基础工具和依赖 + 换阿里云镜像源（centos7官方不维护）
# 如果无法解析，添加到deamon.json "dns": [ "8.8.8.8", "8.8.4.4"]

# --------------- 第一步：替换 CentOS 7 软件源为阿里云镜像 ---------------
RUN mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.backup \
    && curl -o /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo \
    && sed -i 's/mirror.centos.org/mirrors.aliyun.com/g' /etc/yum.repos.d/CentOS-Base.repo \
    && yum clean all \
    && yum makecache

# --------------- 第二步：安装系统依赖 ---------------
RUN yum install -y \
    wget \
    gcc \
    gcc-c++ \
    make \
    libtool \
    automake \
    mesa-libGL \
    libXext \
    libXrender \
    libSM \
    readline-devel \
    libX11-devel \
    && yum clean all

RUN mkdir -p /mnt/raid_p310/isrl03ws10a/workspace310a && chmod 777 /mnt/raid_p310/isrl03ws10a/workspace310a

ARG USERNAME=autockt_user
ARG USER_HOME=/mnt/raid_p310/isrl03ws10a
RUN useradd -m -d $USER_HOME -s /bin/bash $USERNAME

# 安装 Miniconda（手动指定 Python 3.5 版本）
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p $CONDA_DIR \
    && rm miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 安装 NGSPICE 2.7（根据 CentOS 依赖调整）
# --enable-xspice \    # 启用 XSPICE 扩展 
# --disable-debug \    # 禁用调试模式（减少体积）
# --without-x \        # 禁用图形界面（纯命令行）
RUN wget https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/27/ngspice-27.tar.gz \
    && tar -xzf ngspice-27.tar.gz \
    && cd ngspice-27 \
    && ./configure --prefix=/usr/local \
        --enable-xspice \
        --disable-debug \
        --without-x \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf ngspice-27*

# 复制 当前（AutoCkt） 文件夹到容器中
COPY . $USER_HOME/workspace310a/AutoCkt
RUN chown -R $USERNAME:$USERNAME $USER_HOME


USER $USERNAME
WORKDIR $USER_HOME/workspace310a/AutoCkt

# 创建 Conda 环境
RUN conda env create -f $USER_HOME/workspace310a/AutoCkt/environment.yml

# --------------- 激活环境并设置默认命令 ---------------
# centos7 需要先初始化 conda
RUN conda init bash

# 激活环境并设置默认命令
RUN echo "conda activate autockt" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# 设置容器默认命令
CMD ["/bin/bash"]
