---
title: AI编译优化
date: 2021-12-21 15:02:52
tags: compiler
---

### 参考
本文是对知乎大佬杨军的文章的一个总结学习笔记.
原文请参考：知乎 [AI编译优化--总纲](https://zhuanlan.zhihu.com/p/163717035)

### 常用的一些编译框架
- TensorFlow XLA
- TVM
- Tensor Comprehension
- GLOW
- MLIR

### 编译器概述
1. 深度学习编译器,其输入是比较灵活的,具备较高抽象度的计算图,输出包括CPU或者GPU等硬件平台上的底层机器码及执行引擎.
2. AI编译器的目标是针对AI计算任务,以通用编译器的方式完成性能优化. 让用户可以专注于上层模型开发，降低用户手工优化性能的人力开发成本，进一步压榨硬件性能空间.

- 计算密集算子(比如GEMM 和 Convolution)
- 访存密集算子(比如Elementwise Add 和 BN)

<!--more-->

## Google XLA
### 原理
XLA： Accelerated Linear Algebra 加速线性代数
主要用于Tensorflow的编译器, 当然前端也可以用于 JAX Julia PyTorch Nx.

XLA 使用JIT编译来分析用户在运行时创建的TF graph, 将TF OP 转换成HLO.并在HLO上面完成包括Op Fusion 在内的多种图优化. 
最后基于LLVM完成CPU/GPU等后端机器代码的生成.

对于自动CodeGen要求较高的计算密集型算子,如MatMul/Convolution等,和TensorFlow一样会直接调用cuBLAS/cuDNN等Vendor Library.
对除此之外的访存密集型算子, XLA会进行完全自动的Op Fusion和底层代码生成(CodeGen)

XLA还包含一套静态的执行引擎,这个静态性体现在静态的Fixed Shape编译(即,在运行时为每一套输入shape进行一次完整编译并保留编译结果)
静态的算子调度顺序,静态的显存/内存优化等方面.

### 性能收益
- 访存密集型算子的Op Fusion收益
- Fixed Shape架构下,计算图中的shape计算相关的子图会在编译时被分析为静态的编译时常量,节省执行时的节点数量
- HLO层在比TensorFlow Graph的颗粒度上可能存在更大的图优化空间
- 此外还包含 可以方便开发者扩展更多的图优化pass, 包括layout和并发调度优化等等.

### 劣势
- CodeGen原理简单，对计算密集型算子进一步提升性能的发挥空间不大
- 静态shape的架构，用户计算图shape变化范围大的时候，应用会存在一定的限制.

## TVM
### 原理
核心思想在于采用目标代码计算和调度分离的过程

比如给定一个简单的计算表示
```
C = tvm.compute((n,), lambda i: A[i] + B[i])
```
可以得到C语言代码
```
for (int i = 0; i < n; ++i)
{
    C[i] = A[i] + B[i];
}
```
对其额外的调度控制
```
s = tvm.create_schedule(C.op)
xo, xi = s[C].split(s[C].axis[0], factor=32)
```
则可以得到
```
for (int xo = 0; xo < ceil(n / 32); ++xo)
{
    for (int xi = 0; xi < 32; ++xi)
    {
        int i = xo * 32 + xi;
        if (i < n)
        {
            C[i] = A[i] + B[i];
        }
    }
}
```
这种对调度控制的分离和抽象，使得相同的计算逻辑在编译过程中可以很方便地针对不同硬件的计算特性进行调整，在不同的后端上生成对应的高效执行代码。经过优化的 TVM 算子能够达到甚至超越通用算子库和专家手动调优代码的性能。

TVM的 Auto Tuning 机制提供了自动化的性能探索能力，基于同一套调度模板，Auto TVM 能够通过在不同硬件上进行调优的方式寻找到最适合的模板参数，以达到最佳的性能

### 用户的使用方式
TVM 提供了两套不同层级的 API

Relay是更高层的图表示 API，包含了很多常见的计算图算子，如卷积、矩阵乘法、各类激活函数等。Relay 中的复杂算子由一系列预定义的 TOPI(TVM Operator Inventory)模板提供，TOPI 中包含了这些算子的计算定义以及调度定义，用户无需关注其中的细节即可很方便地实现自己需要的计算图

Relay 层 API 还提供了一系列图优化的能力，如 Op Fusion 等

Relay 计算图在执行时需要首先通过 graph runtime codegen 转换成底层 TVM IR 的表示方式，再进一步编译成对应硬件的可执行代码

在Relay API之外，用户也可能通过 TVM API 直接在 TVM IR 层进行计算表达，这一步则需要用户同时完成计算表示以及调度控制两部分内容，但这一更贴近硬件的表示层相对 Relay 层来说更为灵活。底层硬件的特殊计算指令可以通过 TVM Intrinsic 调用得以实现

### 优劣势
优势：
    - 非常适合计算密集型算子

劣势：
    - 调度控制需要由专家来撰写, 当前并未真正自动化.

## MLIR
### 原理
compiler infrasructure
- 表示数据流图的能力, 包含动态shape, 用户可拓展的op生态系统,以及TF变量.(针对XLA的当前局限性)
- 优化(optimizations)和转换(transform)在同一个图上(统一优化框架的野心)
- 适合优化的形式表示内核的算子 (与下一条期望将计算密集算子和访存密集算子集中在同一套框架里统一打击)
- 能够跨内核（融合、循环交换、平铺等）托管高性能计算风格的循环优化并转换数据的内存布局
- 代码生成的转换包括：DMA插入，显式缓存管理、内存平铺 以及 一维 和二维寄存器架构的向量化 (对硬件memory/cache/register的分层描述能力)
- 能够灵活的描述特定目标的操作 (灵活的对AI-domain ASIC的描述能力)
- 在深度学习图上完成量化和其他图转换 (模型层面的优化)

### 设计理念
- 模块化
- reuse 软件组件
- 逐步降低算子
- 通用的方法有效的针对硬件

通过引入dialect的设计理念，使得不同应用场景可以根据自己的需要对MLIR的表示能力(不同抽象层次的算子描述,数据类型以及在相应的dialect层级上施加不同的图变换能力) 进行非侵入式的扩充.

### compiler infrasructure
MLIR 可以很方便的将不同的已经存在的优化方法以Dialect的方式接入,比如(XLA,TVM,甚至TensorRT, nGraph)

### 优劣势
劣势：处于一个快速演化的阶段,很多东西仍然还在不断的发展.

