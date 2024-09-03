---
layout: post
title: RCNN 笔记
category: Paper
tags: 目标检测
keywords: RCNN, Selective-Search, NMS, 迁移学习
description:
---

<center>
# RCNN
<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/RCNN/1.png">


</center>

## 1. 技术路线

**Selective Search + CNN + SVMs**

<center>
<img src="https://raw.githubusercontent.com/cccloves/cccloves.github.io/master/img/RCNN/6.png">

</center>

## 2. 算法流程

    1. 候选区域生成：利用Selective Search对每张图像生成约2K个候选区域
    
    2. 特征提取：对每个候选区域，使用CNN提取特征
    
    3. 类别判断：特征送入每一类的SVM 分类器，判别是否属于该类
    
    4. 位置精修：使用 Bounding-Box Regression 精细修正候选框位置　

## 3. 候选框搜索——Selective Search

R-CNN目标检测首先需要每张图像生成约2000-3000个候选区域，能够生成候选区域的方法很多，比如：

1. objectness
2. selective search
3. category-independen object proposals
4. constrained parametric min-cuts(CPMC)
5. multi-scale combinatorial grouping
6. Ciresan
   R-CNN 采用的是 **Selective Search** 算法。简单来说就是通过一些传统图像处理方法将图像分成很多小尺寸区域，然后根据小尺寸区域的特征合并小尺寸得到大尺寸区域，以实现候选区域的选取。

基本思路如下：

1. 使用一种过分割手段，将图像分割成小区域；

2. 查看现有小区域，合并可能性最高的两个区域。重复直到整张图像合并成一个区域位置,
优先合并以下四种区域：

        颜色（颜色直方图）相近的；
        
        纹理（梯度直方图）相近的；
        
        合并后总面积小的----保证合并操作的尺度较为均匀，避免一个大区域陆续“吃掉”其他小区域；
        
        合并后，总面积在其BBOX中所占比例大的----保证合并后形状规则；

3. 输出所有曾经存在过的区域，所谓候选区域；

## 4. 特征提取——AlexNet CNN

### 4.1 预处理

+ 通过 Selective Search 产生的候选区域大小不一，为了与 Alexnet 兼容，R-CNN 采用了非常暴力的手段，那就是无视候选区域的大小和形状，统一变换到 227x227 的尺寸。

+ 图像缩放：

    各向异性缩放：不关心图像的长宽比例，全部归一化为227×227；

    各向同性缩放：

    + 把Bounding Box的边界扩展成正方形，然后裁剪。超出边界部分就用Bounding Box颜色均值填充。

    + 先把Bounding Box裁剪出来，然后用Bounding Box颜色均值填充背景

    PS：外扩的尺寸大小，形变时是否保持原比例，对框外区域直接截取还是补灰。会轻微影响性能。

### 4.2 网络结构

- 采用训练好的AlexNet模型进行PASCAL VOC 2007样本集下的微调，学习率=0.001（PASCAL VOC 2007样本集上既有图像中物体类别标签，也有图像中物体位置标签）
- mini-batch为32个正样本和96个负样本（由于正样本太少）
- 修改了原来的1000为类别输出，改为21维【20类+背景】输出。

### 4.3 Pre-train

使用ILVCR 2012的全部数据进行训练，输入一张图片，输出1000维的类别标号；

<center>
<img src="https://raw.githubusercontent.com/cccloves/cccloves.github.io/master/img/RCNN/3.png">

</center>

### 4.4 Fine-running

+ 使用上述网络，将最后一层换成4096->21的全连接网络，在PASCAL VOC 2007的数据集（目标数据集）上进行检测训练；

+ 使用通过selective search之后的region proposal 作为网络的输入。输出21维的类别标号，表示20类+背景；

+ 如果当前region  proposal的IOU大于0.5，把他标记为positive，其余的是作为negtive，去训练detection网络。

+ 学习率0.001，每一个batch包含32个正样本（属于20类）和96个背景。

<center>
<img src="https://raw.githubusercontent.com/cccloves/cccloves.github.io/master/img/RCNN/5.png">

</center>

## 5. 类别判断——SVM

对每一类目标，使用一个线性SVM二类分类器进行判别。

CNN得到的4096维特征输入到SVM进行分类，看看这个feature vector所对应的region proposal是需要的物体还是无关的实物(background) 。 排序，canny边界检测之后就得到了我们需要的bounding-box。

由于负样本很多，使用hard negative mining方法。

+ 正样本：本类的真值标定框。

+ 负样本：考察每一个候选框，如果和本类所有标定框的重叠都小于0.3，认定其为负样本

## 6. 位置精修

### 6.1 回归器

对每一类目标，使用一个线性脊回归器进行精修。正则项λ=10000。输入为深度网络pool5层的4096维特征，输出为xy方向的缩放和平移。

### 6.2 训练样本

判定为本类的候选框中，和真值重叠面积大于0.6的候选框。

<center>

<img src="https://raw.githubusercontent.com/cccloves/cccloves.github.io/master/img/RCNN/4.png">

</center>

## 7. 测试阶段

1. 对给定的一张图片，通过Selective Search得到2000个region Proposals，将每个region proposals归一化到227*227；

2. 每一个Proposal都经过已经训练好的CNN网络， 得到fc7层的features,4096-dimension，即2000*4096；

3. 用SVM分类器(4096\*K)得到相应的score，即2000\*K；

4. 用CNN中pool5的特征，利用已经训练好的权值，得到bounding box的修正值，原先的proposal经过修正得到新的proposal的位置；

5. 对每一类别的scores，采用非极大值抑制（NMS），去除相交的多余的框；

    1. 对于2000*K中的每一列，进行nms；

    2. 对于特定的这一列（这一类），选取值最大的对应的proposal，计算其他proposal跟此proposal的IOU，剔除那些重合很多的proposal；

    3. 再从剩下的proposal里选取值最大的，然后再进行剔除，如此反复进行，直到没有剩下的proposal；

    4. K列（K类）都进行这样的操作，即可得到最终的bounding box和每一个bounding box对应的类别及其score值；









# Fast-RCNN

***除了Proposal阶段，Fast RCNN基本实现了end-to-end的CNN对象检测模型***

<center>
<img src="https://raw.githubusercontent.com/cccloves/cccloves.github.io/master/img/Fast-RCNN/6.png">

</center>

## 1. R-CNN、SPP-net的缺点

1. R-CNN和SPP-Net的训练过程类似，分多个阶段进行，实现过程较复杂。这两种方法首先选用Selective Search方法提取proposals,然后用CNN实现特征提取，最后基于SVMs算法训练分类器，在此基础上还可以进一步学习检测目标的boulding box。

2. 训练时间和空间开销大。SPP-Net在特征提取阶段只需要对整图做一遍前向CNN计算，然后通过空间映射方式计算得到每一个proposal相应的CNN特征；区别于前者，RCNN在特征提取阶段对每一个proposal均需要做一遍前向CNN计算，考虑到proposal数量较多（~2000个），因此RCNN特征提取的时间成本很高。R-CNN和SPP-Net用于训练SVMs分类器的特征需要提前保存在磁盘，考虑到2000个proposal的CNN特征总量还是比较大，因此造成空间代价较高。

3. R-CNN检测速度很慢。RCNN在特征提取阶段对每一个proposal均需要做一遍前向CNN计算，如果用VGG进行特征提取，处理一幅图像的所有proposal需要47s；

4. 特征提取CNN的训练和SVMs分类器的训练在时间上是先后顺序，两者的训练方式独立，因此SVMs的训练Loss无法更新SPP-Layer之前的卷积层参数，因此即使采用更深的CNN网络进行特征提取，也无法保证SVMs分类器的准确率一定能够提升。

## 2. Fast-RCNN 改进

1. 训练的时候，pipeline是隔离的，先提proposal，然后CNN提取特征，之后用SVM分类器，最后再做bbox regression。Fast RCN实现了end-to-end的joint training(提proposal阶段除外)；

2. 训练时间和空间开销大。RCNN中ROI-centric的运算开销大，所以Fast RCN用了image-centric的训练方式来通过卷积的share特性来降低运算开销；RCNN提取特征给SVM训练时候需要中间要大量的磁盘空间存放特征，Fast RCN去掉了SVM这一步，所有的特征都暂存在显存中，就不需要额外的磁盘空间了;

3. 测试时间开销大。依然是因为ROI-centric的原因(whole image as input->ss region映射)，这点SPP-Net已经改进，Fast RCN进一步通过single scale(pooling->spp just for one scale) testing和SVD(降维)分解全连接来提速。

## 3. 网络框架

### 3.1 训练过程

1. selective search在一张图片中得到约2k个建议窗口（Region proposal）；

2. 将整张图片输入CNN，进行特征提取；

3. 把建议窗口映射到CNN的最后一层卷积feature map上；

4. 通过一个Rol pooling layer（SSP layer的特殊情况）使每个建议窗口生成固定尺寸的feature map；

5. 利用Softmax Loss(探测分类概率) 和Smooth L1 Loss(探测边框回归)对分类概率和边框回归(Bounding box regression)联合训练（测试时候，在4之后做一个NMS）；

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Fast-RCNN/1.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Fast-RCNN/2.png">

</center>

### 3.2 RoI pooling layer

这是SPP pooling层的一个简化版，只有一级“金字塔”，输入是N个特征映射和一组R个RoI，R>>N。N个特征映射来自于最后一个卷积层，每个特征映射都是H x W x C的大小。
每个RoI是一个元组(n, r, c, h, w)，n是特征映射的索引，n∈{0, ... ,N-1}，(r, c)是RoI左上角的坐标，(h, w)是高与宽。输出是max-pool过的特征映射，H' x W' x C的大小，H'≤H，W'≤W。对于RoI，bin-size ~ h/H' x w/W'，这样就有H'W'个输出bin，bin的大小是自适应的，取决于RoI的大小。

#### 3.2.1 作用

+ 将image中的rol定位到feature map中对应patch

+ 用一个单层的SPP layer将这个feature map patch下采样为大小固定的feature再传入全连接层。即RoI pooling layer来统一到相同的大小－> (fc)feature vector 即－>提取一个固定维度的特征表示。

#### 3.2.2 Roi Pooling Test Forward

Roi_pool层将每个候选区域均匀分成M×N块，对每块进行max pooling。将特征图上大小不一的候选区域转变为大小统一的数据，送入下一层。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Fast-RCNN/3.png">

</center>

#### 3.2.3 Roi Pooling Training Backward

首先考虑普通max pooling层。设 $x_{i}$ 为输入层的节点，$y_{j}$ 为输出层的节点。

<div>
$$\frac{\partial L}{\partial x_{i}} =\begin{cases}
0 & \delta \left ( i, j \right )= \text { false }\\
\frac{\partial L}{\partial y_{j}} & \delta \left ( i, j \right )= \text { true }
\end{cases}$$
</div>

其中判决函数 $\delta \left ( i, j \right )$ 表示 i 节点是否被 j 节点选为最大值输出。不被选中有两种可能：$x_{i}$ 不在 $y_{j}$ 范围内，或者 $x_{i}$ 不是最大值。

对于roi max pooling，一个输入节点可能和多个输出节点相连。设 $x_{i}$ 为输入层的节点，$y_{rj}$ 为第 r 个候选区域的第 j 个输出节点

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Fast-RCNN/4.png">

</center>

$$\frac{\partial L}{\partial x_{i}} = \sum _{r,j} \delta \left ( i, r, j \right ) \frac{\partial L}{\partial y_{rj}}$$

判决函数 $\delta \left ( i, r,j \right )$ 表示 i 节点是否被候选区域 r 的第 j 个节点选为最大值输出。代价对于 $x_{i}$ 的梯度等于所有相关的后一层梯度之和。

## 4. 训练过程

### 4.1 Pre-trained

用了3个预训练的ImageNet网络（CaffeNet/VGG_CNN_M_1024/VGG16）。预训练的网络初始化Fast RCNN要经过三次变形：

1. 最后一个max pooling层替换为RoI pooling层，设置H’和W’与第一个全连接层兼容。(SPPnet for one scale -> arbitrary input image size )

2. 最后一个全连接层和softmax（原本是1000个类）-> 替换为softmax的对K+1个类别的分类层，和bounding box 回归层。 (Cls and Det at same time)

3. 输入修改为两种数据：一组N个图形，R个RoI，batch size和ROI数、图像分辨率都是可变的。

### 4.2 Fine-tuning

#### 4.2.1 Multi-task loss

两个输出层，一个对每个RoI输出离散概率分布：

$$p = \left ( p_{0},\cdots , p_{K} \right )$$

一个输出bounding box回归的位移：

$$t^{k} = \left ( t_{x}^{k},t_{y}^{k},t_{w}^{k},t_{h}^{k}\right )$$

k 表示类别的索引，前两个参数是指相对于 object proposal 尺度不变的平移，后两个参数是指对数空间中相对于 object proposal 的高与宽。把这两个输出的损失写到一起：

$$L \left ( p,k^{\ast},t,t^{\ast} \right ) = L_{cls}\left ( p,k^{\ast} \right ) + \lambda \left [ k^{\ast}\geq 1 \right ] L_{loc}\left ( t,t^{\ast} \right )$$

$k^{\ast}$ 是真实类别，式中第一项是分类损失，第二项是定位损失，L 由 R 个输出取均值而来.

1. 对于分类 loss，是一个 N+1 路的 softmax 输出，其中的N是类别个数，1是背景。SVM → softmax

2. 对于回归 loss，是一个 4xN 路输出的 regressor，也就是说对于每个类别都会训练一个单独的 regressor，这里 regressor 的 loss 不是 L2 的，而是一个平滑的 L1，形式如下：

$$L_{loc}\left ( t,t^{\ast} \right ) = \sum_{i\in \{x,y,w,h\}} \text{smooth}_{L_{1}}\left ( t_{i},t_{i}^{\ast} \right )$$

in which

<div>
$$\text{smooth}_{L_{1}}\left (x \right )=
\begin{cases}
0.5x^{2} & \text{ if } \left | x \right | < 1 \\
\left | x \right | -0.5 & \text{ otherwise }
\end{cases}$$
</div>

#### 4.2.2 Mini-batch sampling

- each mini batch：sampling 64 Rols from eatch image
- images num：N = 2
- Rols num：R = 128
- data argumentation: flipped with probability 0.5

R个候选框的构成方式如下：

|类别|比例|方式|
|:---|:---|:---|
|前景|25%|与某个真值重叠在 [0.5,1] 的候选框|
|背景|75%|与真值重叠的最大值在 [0.1,0.5) 的候选框|

#### 4.2.3 全连接层提速

分类和位置调整都是通过全连接层(fc)实现的，设前一级数据为 x 后一级为 y，全连接层参数为 W，尺寸 $u \times v$。

一次前向传播(forward)即为：

$$y= Wx$$

计算复杂度为 $u \times v$ 。

将进行SVD分解，并用前t个特征值近似, 原来的前向传播分解成两步:

$$W = U \Sigma V^{T} \approx U\left ( :,1:t \right )\cdot \Sigma \left ( 1:t,1:t \right ) \cdot V \left ( :,1:t \right )^{T}$$

计算复杂度变为 $u \times t + v \times t$。

在实现时，相当于把一个全连接层拆分成两个，中间以一个低维数据相连。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Fast-RCNN/5.png">

</center>

## 5. 实验结论

- 多任务Loss学习方式可以提高算法准确率

- 多尺度图像训练Fast-R-CNN与单尺度图像训练相比只能提升微小的mAP,但是时间成本却增加了很多。因此，综合考虑训练时间和mAP，作者建议直接用一种尺度的图像训练Fast-R-CNN.

- 训练图像越多，模型准确率也会越高

- 网络直接输出各类概率(softmax)，比SVM分类器性能略好

- 不是说Proposal提取的越多效果会越好，提的太多反而会导致mAP下降



# Faster-RCNN

***Faster RCNN真正实现了完全end-to-end的CNN目标检测模型***

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/11.png">

</center>

## 1. 区域生成网络——RPN（Region Proposal Networks）

先通过对应关系把 feature map 的点映射回原图，在每一个对应的原图设计不同的固定尺度窗口（bbox），根据该窗口与ground truth的IOU给它正负标签，让它学习里面是否有object，这样就训练一个网络（Region Proposal Network）。

由于我们只需要找出大致的地方，无论是精确定位位置还是尺寸，后面的工作都可以完成，作者对bbox做了三个固定：固定尺度变化（三种尺度），固定scale ratio变化（三种ratio），固定采样方式（只在feature map的每个点在原图中的对应ROI上采样，反正后面的工作能进行调整） 。如此就可以降低任务复杂度。可以在特征图上提取proposal之后，网络前面就可以共享卷积计算结果（SPP减少计算量的思想）。

这个网络的结果就是卷积层的每个点都有有关于k个achor boxes的输出，包括是不是物体，调整box相应的位置。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/1.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/2.png">

</center>

**具体过程**

### 1.1 得到最终用来预测的feature map

图片在输入网络后，依次经过一系列conv+relu （套用ImageNet上常见的分类网络即可 本论文实验了5层的ZF,16层的VGG-16）得到的feature map，额外添加一个conv+relu层，输出51\*39\*256维特征（feature map）。准备后续用来选取proposal，并且此时坐标依然可以映射回原图。

### 1.2 计算Anchors

在feature map上的每个特征点预测多个region proposals。具体作法是：把每个特征点映射回原图的感受野的中心点当成一个基准点，然后围绕这个基准点选取k个不同scale、aspect ratio的anchor。论文中3个scale（三种面积 $\left\\{ 128^{2}, 256^{2}, 521^{2} \right\\}$），3 个aspect ratio( $\left\\{ 1:1, 1:2, 2:1 \right\\}$)。

<center>
<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/3.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/4.png">

</center>

### 1.3 关于正负样本的划分

考察训练集中的每张图像（含有人工标定的ground true box） 的所有anchor（N\*M\*k）

1. 对每个标定的ground true box区域，与其重叠比例最大的anchor记为 正样本 (保证每个ground true 至少对应一个正样本anchor)；

2. 对a)剩余的anchor，如果其与某个标定区域重叠比例大于0.7，记为正样本（每个ground true box可能会对应多个正样本anchor；但每个正样本anchor 只可能对应一个grand true box）；如果其与任意一个标定的重叠比例都小于0.3，记为负样本；

    <center>
    <img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/5.png">
    
    </center>
    
3. 对a\), b\)剩余的 anchor，弃去不用。

4. 跨越图像边界的 anchor 弃去不用

### 1.4 定义损失函数

对于每个anchor，首先在后面接上一个二分类softmax，有2个score 输出用以表示其是一个物体的概率与不是一个物体的概率 $p_i$, 然后再接上一个bounding box的regressor 输出代表这个anchor的4个坐标位置（$t_i$），因此RPN的总体Loss函数可以定义为：

$$
L\left ( \left \{ p_{i} \right \}\left \{ t_{i} \right \} \right ) = \frac{1}{N_{cls}} \sum_{i} L_{cls}\left ( p_{i},p_{i}^{\ast} \right ) + \lambda \frac{1}{N_{reg}} \sum_{i} p_{i}^{\ast} L_{reg} \left ( t_{i},t_{i}^{\ast} \right )
$$

i 表示第 i 个 anchor，当 anchor 是正样本时 $p_{i}^{\ast} = 1$，是负样本则=0；

$t_{i}^{\ast}$ 表示一个与正样本 anchor 相关的 ground true box 坐标；

每个正样本 anchor 只可能对应一个ground true box；

一个正样本 anchor 与某个 grand true box 对应，那么该 anchor 与 ground true box 的IOU要么是所有 anchor 中最大，要么大于0.7；

x, y, w, h分别表示 box 的中心坐标和宽高；

$x, x_{\alpha}, x^{\ast}$分别表示 predicted box, anchor box, and ground truth box (y,w,h同理)；

$t_{i}$ 表示 predict box 相对于 anchor box 的偏移；

$t_{i}^{\ast}$ 表示 ground true box 相对于 anchor box 的偏移，学习目标自然就是让前者接近后者的值；

<div>
$$
\begin{matrix}
t_{x} = \left ( x - x_{\alpha} \right ) / \omega_{\alpha},& t_{y} = \left ( y - y_{\alpha} \right ) / h_{\alpha},\\
t_{\omega} = \log \left ( \omega \right / \omega_{a}),& t_{h} = \log \left ( h \right / h_{\alpha}),\\
t_{x}^{\ast} = \left ( x^{\ast} - x_{\alpha} \right ) / \omega_{\alpha},& t_{y}^{\ast} = \left ( y^{\ast} - y_{\alpha} \right ) / h_{\alpha},\\
t_{\omega}^{\ast} = \log \left ( \omega^{\ast} \right / \omega_{\alpha}),& t_{h}^{\ast} = \log \left ( h^{\ast} \right / h_{\alpha}),
\end{matrix}
$$
</div>

<center>
<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/6.png">

</center>

其中 $L_{reg}$ 是：

<div>
$$
smooth_{L_{1}}\left ( x \right ) =
\begin{cases}
0.5x^{2} & \left | x \right | \leq 1 \\
\left | x \right | - 0.5 & \text{otherwise}
\end{cases}
$$
</div>

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/7.png">

</center>

$p_{i}^{\ast}$ 表示这些regressor的loss指针对正样本而言，因为负样本时 $p_{i}^{\ast} = 0$ 该项被消去；

$L_{cls}$ 是关于两种类别 (object vs. not object) 的 log loss；

### 1.5 训练RPN

文中提到如果每幅图的所有anchor都去参与优化loss function，那么最终会因为负样本过多导致最终得到的模型对正样本预测准确率很低。因此在每幅图像中随机采样256个anchors去参与计算一次mini-batch的损失。正负比例1:1(如果正样本少于128则补充采样负样本)

**注意点：**

在到达全连接层之前，卷积层和Pooling层对图片输入大小其实没有size的限制，因此RCNN系列的网络模型其实是不需要实现把图片resize到固定大小的；

n=3看起来很小，但是要考虑到这是非常高层的feature map，其size本身也没有多大，因此3×3 9个矩形中，每个矩形窗框都是可以感知到很大范围的。

## 2. Sharing Features for RPN and Fast R-CNN

前面已经讨论如何训练提取proposal的RPN，分类采用Fast R-CNN。如何把这两者放在同一个网络结构中训练出一个共享卷积的Multi-task网络模型。

我们知道，如果是分别训练两种不同任务的网络模型，即使它们的结构、参数完全一致，但各自的卷积层内的卷积核也会向着不同的方向改变，导致无法共享网络权重，论文作者提出了三种可能的方式：

### 2.1 Alternating training

此方法其实就是一个不断迭代的训练过程，既然分别训练RPN和Fast-RCNN可能让网络朝不同的方向收敛，

1. 那么我们可以先独立训练RPN，然后用这个RPN的网络权重对Fast-RCNN网络进行初始化并且用之前RPN输出proposal作为此时Fast-RCNN的输入训练Fast R-CNN；

2. 用Fast R-CNN的网络参数去初始化RPN。之后不断迭代这个过程，即循环训练RPN、Fast-RCNN；

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/8.png">

</center>

### 2.2 Approximate joint training

这里与前一种方法不同，不再是串行训练RPN和Fast-RCNN，而是尝试把二者融入到一个网络内，具体融合的网络结构如下图所示，可以看到，proposals是由中间的RPN层输出的，而不是从网络外部得到。需要注意的一点，名字中的"approximate"是因为反向传播阶段RPN产生的cls score能够获得梯度用以更新参数，但是proposal的坐标预测则直接把梯度舍弃了，这个设置可以使backward时该网络层能得到一个解析解（closed results），并且相对于Alternating traing减少了25-50%的训练时间。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/9.png">

</center>

### 2.3 Non-approximate training

上面的Approximate joint training把proposal的坐标预测梯度直接舍弃，所以被称作approximate，那么理论上如果不舍弃是不是能更好的提升RPN部分网络的性能呢？

作者把这种训练方式称为“ Non-approximate joint training”，但是此方法在paper中只是一笔带过。

### 2.4 4-Step Alternating Training（作者使用）

思路和迭代的Alternating training有点类似，但是细节有点差别：

1. 用ImageNet模型初始化，独立训练一个RPN网络；

2. 仍然用ImageNet模型初始化，但是使用上一步RPN网络产生的proposal作为输入，训练一个Fast-RCNN网络，至此，两个网络每一层的参数完全不共享；

3. 使用第二步的Fast-RCNN网络参数初始化一个新的RPN网络，但是把RPN、Fast-RCNN共享的那些卷积层的learning rate设置为0，也就是不更新，仅仅更新RPN特有的那些网络层，重新训练，此时，两个网络已经共享了所有公共的卷积层；

4. 仍然固定共享的那些网络层，把Fast-RCNN特有的网络层也加入进来，形成一个unified network，继续训练，fine tune Fast-RCNN特有的网络层，此时，该网络已经实现我们设想的目标，即网络内部预测proposal并实现检测的功能。

<center>
<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Faster-RCNN/10.png">

</center>



# RCNN系列总结



