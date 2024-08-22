

## 环境配置

### 虚拟环境激活

```linux
conda info --env
conda activate dino
conda deactivate
```



### Jupyter Notebook

[MobaXterm实现代理功能及把内网服务器，用公网地址转发出去。_mobaxterm设置代理-CSDN博客](https://blog.csdn.net/weixin_43606975/article/details/119958628)

[Jupyter Notebook 7+ 远程访问配置_jupyter notebook 远程文件访问-CSDN博客](https://blog.csdn.net/qq_46046560/article/details/139123105)

无法使用第三方包的问题：

[关于在终端能到import模块 而在jupyter notebook无法导入的问题_vue项目中能集成 jupyter notebook 吗-CSDN博客](https://blog.csdn.net/qq_34650787/article/details/83304080)

```
ipython kernelspec list
```



启用很简单：打开

![image-20240701174311880](C:\Users\08\AppData\Roaming\Typora\typora-user-images\image-20240701174311880.png)

终端进入虚拟环境后输入：jupyter notebook



### Pycharm

```
cd /home/cl/yyh/pycharm-community-2024.1.4/bin/

./pycharm.sh
```







## Linux常用命令

### 文件移动

1. 本地文件上传：拖动即可
2. 虚拟机文件下载：左上角下载符号







## Docker



```
sudo nvidia-docker exec -it deepstream7 bash
sudo nvidia-docker cp deepstream7:/home/yyh/Grounding-Dino-FineTuning-main/weights/model_weights2000.pth  /home/cl/yyh/Grounding-Dino-FineTuning-main/weights/

```

