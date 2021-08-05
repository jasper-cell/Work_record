## Docker

1. 安装成功之后运行: docker version 或者 docker info
2. docker是服务器客户端架构，命令行运行docker的时候，需要本机有Docker服务，可以采用以下命令启动docker:
   1. sudo service docker start
   2. sudo systemctl start docker
3. image文件：
   1. **docker image ls**
   2. **docker image rm 【imageName】**
4. 实例：Hello World：
   1. **docker image pull hello-world**
   2. **docker image ls**
   3. **docker container run hello-world**
   4. **docker container kill 【containerID】**
5. 容器文件
   1. 一旦容器生成，就会同时存在两个文件：image文件和容器文件，而且关闭容器并不会删除容器文件，只是停止容器运行而已
   2. **docker container ls**  # 列出本机正在运行的容器
   3. **docker container ls -all**  # 列出本机所有容器，包括终止运行的容器
   4. 终止运行的容器文件，依然会占据硬盘空间，可以使用**docker container rm 【containerID】**来删除
   5. docker container start [containerID] 该命令用来启动已经生成、已经停止运行的容器文件
   6. docker container stop [containerID] 比skill更加优雅的停止容器的运行
   7. docker container logs 用来查看docker容器的输出，即容器里面shell的标准输出。如果docker run命令运行容器的时候，没有使用-it参数，就要用这个命令查看输出
   8. docker exec -it [containerID] /bin/bash 该命令用于进入一个正在运行的docker容器，如果docker run命令没有加上-it参数，就要用这个命令进入容器，一旦进入容器，就可以在容器的shell执行命令了
   9. docker container cp [containerID]:[/path/to/file]

