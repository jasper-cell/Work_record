## Git

1. 创建版本库
   1. 初始化一个Git仓库，使用**git init** 命令
   2. 添加文件到Git仓库，分两步
      1. 使用**git add <file>**，注意，可反复多次使用，添加多个文件
      2. 使用命令**git commit -m <message>** 完成提交。
2. 时光穿梭机
   1. 要随时掌握工作区的状态，使用git status命令
   2. 如果git status告诉文件有被修改过，使用git diff 可以查看修改内容
3. 版本回退
   1. HEAD指向的就是当前版本，因此，Git允许我们在版本的历史之间进行穿梭，使用命令**git reset --hard commit_id**
   2. 穿梭前，使用**git log**可以查看提交历史，以便确定要退回到哪个版本
   3. 要重返未来，使用**git reflog** 查看命令历史，以便确定要回到未来的哪个版本
4. 工作区和暂存区
5. 管理修改
   1. 每次修改如果不用git add到暂存区，那就不会加入到commit中
6. 撤销修改
   1. 当你改乱了工作区的某个文件的内容，想直接丢弃工作区的修改时，用命令**git checkout -- <file>**
   2. 当你不但改乱了工作区的某个文件的内容，还添加到了暂存区，想丢弃修改时，分两步：
      1. **git reset HEAD <file>**
      2. **git checkout -- <file>**
   3. 已经提交不合适的修改到版本库时，想撤销本次提交，可以使用git reset --hard commit_id
7. 删除文件
   1. 命令**git rm** 用于删除一个文件，如果一个文件已经被提交到版本库，那么永远不用担心误删，但是要小心，你只能回复文件到最新的版本，你会丢失最后一次提交后你修改的内容。
8. 远程仓库
   1. 要关联一个远程仓库，使用命令**git remote add origin git@server-name"path/repo-name.git**
   2. 关联一个远程仓库时必须给远程仓库指定一个名字，origin是默认命名习惯
   3. 关联后，使用命令**git push -u origin master**第一次推送master分支的所有内容
   4. 此后，每次本地提交后，只要有必要就可以用**git push origin master**推送最新的修改版本
   5. 本地仓库与远程仓库不一致的时候可以使用，**git pull --rebase origin master** 进行同步
9. 远程克隆一个仓库
   1. **git clone**，支持https和ssh等协议
10. 分支管理
    1. 查看分支： **git branch**
    2. 创建分支：**git branch <name>**
    3. 切换分支: **git checkout <name>** 或者 **git switch <name>**
    4. 创建+切换分支： **git checkout -b <name>**
    5. 合并某分支到当前分支: **git merge <name>**
    6. 删除分支: **git branch -d <name>**
11. 解决冲突
    1. 当git无法自动合并的时候首先要解决冲突，再合并
    2. **git log --graph**可看到分支合并图

