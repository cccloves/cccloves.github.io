# 入门

[Git常用操作总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/404642045)

[Git使用教程,最详细，最傻瓜，最浅显，真正手把手教 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/30044692)

## ssh设定

获取ssh并粘贴，将项目clone到本地

```python
#获取ssh
ssh-keygen -o -t rsa -b 4096 -C "yuyihan@bilibili.com"
y
cd ~/.ssh
cat id_rsa.pub #复制粘贴

#创建需要clone到的文件目录
cd /Users/yihan/Desktop/
mkdir pcdn
cd pcdn
git clone (git@git.bilibili.co:luyuangen/MakeQualityBetter.gi)#gitlab上copy下来的

```



## 修改文件

```python
cd MakeQualityBetter #进入主目录
git fetch
git pull #更新
git checkout master #切到主分支中
git checkout -b feature/pcdn #创建个人分支
open . #手动操作
cd /rpa/web_platform/dist/pcdn #进入需要修改的目录下
git status #查看状态
git add pcdn_alarm.md pcdn_bandwidth.md pcdn_box.md pcdn_player.md pcdn_supernode.md #将所有操作的文件add起来（包括添加修改删除等
git commit -m "pcdn的web网页修改" #commit
git push #将commit进行push，第一次：git push --set-upstream origin feature/pcdn
```

之后需要提交merge request，批准后就合入啦



## 调用软件



```python
#pycharm
cd /snap/pycharm-community/380/bin/
sh pycharm.sh
#jupyter notebook
jupyter notebook

cd /mnt/nfs/
```



```
sudo mount 192.168.110.199:/zp-raid1/nfs /mnt/nfs 
```

