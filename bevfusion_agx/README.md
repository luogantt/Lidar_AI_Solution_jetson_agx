


[我的项目链接](https://github.com/luogantt/Lidar_AI_Solution/tree/master/CUDA-BEVFusion)

 环境是 11.4 ，我是用sdkmanager 刷的系统，各种组件 都刷上了，而且只能通用 ubuntu 本机刷系统 ，虚拟机不行 ，在刷的过程中 需要频繁 连接 。
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Sun_Oct_23_22:16:07_PDT_2022
Cuda compilation tools, release 11.4, V11.4.315
Build cuda_11.4.r11.4/compiler.31964100_0
```

######   cmake 也要更改过 ，在头部加几行代码

 [我的cmake](https://github.com/luogantt/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/CMakeLists.txt)
 

```

# 强制 Protobuf 使用动态库（添加在文件最顶部）
set(Protobuf_USE_STATIC_LIBS OFF)
set(Protobuf_USE_SHARED_LIBS ON)

# 显式指定 Protobuf 动态库路径（兜底）
find_library(PROTOBUF_SHARED_LIB protobuf PATHS /usr/local/lib NO_DEFAULT_PATH)
if(NOT PROTOBUF_SHARED_LIB)
    message(FATAL_ERROR "Protobuf dynamic library not found!")
endif()
```
 
<font color =darkred >还有一点要必须强调的是 Lidar_AI_Solution下面项目 编译的时候都依赖 /libraries/3DSparseConvolution/libspconv 下面的 动态库，比如我的 cuda=11.4 就需要 libspconv/lib/aarch64_cuda11.4/libspconv.so 这个动态库
 <font color =darkblue >/libspconv.so 这个动态库不开源！
  <font color =darkblue >/libspconv.so 这个动态库不开源！
    <font color =darkblue >/libspconv.so 这个动态库不开源！
      <font color =darkblue >重要事情说三遍！

##### 第二个问题就是 protobuf 需要安装好 基本上就是 这个两个问题要特别注意

###### 其他按照教程就可以了
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/67bfd492fd054acfa09c45a3813e0e12.jpeg#pic_center)

```
=============================================
==================BEVFusion===================
[⏰ [NoSt] CopyLidar]: 	0.52656 ms
[⏰ [NoSt] ImageNrom]: 	7.13533 ms
[⏰ Lidar Backbone]: 	18.64733 ms
[⏰ Camera Depth]: 	0.16378 ms
[⏰ Camera Backbone]: 	13.59942 ms
[⏰ Camera Bevpool]: 	2.53222 ms
[⏰ VTransform]: 	2.66922 ms
[⏰ Transfusion]: 	8.26883 ms
[⏰ Head BoundingBox]: 	16.14006 ms
Total: 62.021 ms
=============================================
Save to build/cuda-bevfusion.jpg
[Warning]: If you got an inaccurate boundingbox result please turn on the layernormplugin plan. (main.cpp:207)

```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2be2136bf87e4c41a83a14ea1335ddaf.jpeg#pic_center)

  
  

  



























<h1 style="text-align: center">Lidar AI Solution</h1>
This is a highly optimized solution for self-driving 3D-lidar repository.
It does a great job of speeding up sparse convolution/CenterPoint/BEVFusion/OSD/Conversion.

![title](assets/title.png)

## Pipeline overview
![pipeline](assets/pipeline.png)

## GetStart
```
$ git clone --recursive https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution
$ cd Lidar_AI_Solution
```
- For each specific task please refer to the readme in the sub-folder.

## 3D Sparse Convolution
A tiny inference engine for [3d sparse convolutional networks](https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/backbones/scn.py) using int8/fp16.
- **Tiny Engine:** Tiny Lidar-Backbone inference engine independent of TensorRT.
- **Flexible:** Build execution graph from ONNX.
- **Easy To Use:** Simple interface and onnx export solution.
- **High Fidelity:** Low accuracy drop on nuScenes validation.
- **Low Memory:** 422MB@SCN FP16, 426MB@SCN INT8.
- **Compact:** Based on the CUDA kernels and independent of cutlass.

## CUDA BEVFusion
CUDA & TensorRT solution for [BEVFusion](https://arxiv.org/abs/2205.13542) inference, including:
- **Camera Encoder**: ResNet50 and finetuned BEV pooling with TensorRT and onnx export solution.
- **Lidar Encoder**: Tiny Lidar-Backbone inference independent of TensorRT and onnx export solution.
- **Feature Fusion**: Camera & Lidar feature fuser with TensorRT and onnx export solution.
- **Pre/Postprocess**: Interval precomputing, lidar voxelization, feature decoder with CUDA kernels.
- **Easy To Use**: Preparation, inference, evaluation all in one to reproduce torch Impl accuracy.
- **PTQ**: Quantization solutions for [mmdet3d/spconv](https://github.com/mit-han-lab/bevfusion/tree/main/mmdet3d/ops/spconv), Easy to understand.


## cuOSD(CUDA On-Screen Display Library)
Draw all elements using a single CUDA kernel.
- **Line:** Plotting lines by interpolation(Nearest or Linear).
- **RotateBox:** Supports drawn with different border colors and fill colors.
- **Circle:** Supports drawn with different border colors and fill colors.
- **Rectangle:** Supports drawn with different border colors and fill colors.
- **Text:** Supports [stb_truetype](https://github.com/nothings/stb/blob/master/stb_truetype.h) and [pango-cairo](https://pango.gnome.org/) backends, allowing fonts to be read via TTF or using font-family.
- **Arrow:** Combination of arrows by 3 lines.
- **Point:** Plotting points by interpolation(Nearest or Linear).
- **Clock:** Time plotting based on text support

## cuPCL(CUDA Point Cloud Library)
Provide several GPU accelerated Point Cloud operations with high accuracy and high performance at the same time: cuICP, cuFilter, cuSegmentation, cuOctree, cuCluster, cuNDT, Voxelization(incoming).
- **cuICP:** CUDA accelerated iterative corresponding point vertex cloud(point-to-point) registration implementation.
- **cuFilter:** Support CUDA accelerated features: PassThrough and VoxelGrid.
- **cuSegmentation:** Support CUDA accelerated features: RandomSampleConsensus with a plane model.
- **cuOctree:** Support CUDA accelerated features: Approximate Nearest Search and Radius Search.
- **cuCluster:** Support CUDA accelerated features: Cluster based on the distance among points.
- **cuNDT:** CUDA accelerated 3D Normal Distribution Transform registration implementation for point cloud data.

## YUVToRGB(CUDA Conversion)
YUV to RGB conversion. Combine Resize/Padding/Conversion/Normalization into a single kernel function.
- **Most of the time, it can be bit-aligned with OpenCV.**
    - It will give an exact result when the scaling factor is a rational number.
    - Better performance is usually achieved when the stride can divide by 4.
- Supported Input Format:
    - **NV12BlockLinear**
    - **NV12PitchLinear**
    - **YUV422Packed_YUYV**
- Supported Interpolation methods:
    - **Nearest**
    - **Bilinear**
- Supported Output Data Type:
    - **Uint8**
    - **Float32**
    - **Float16**
- Supported Output Layout:
    - **CHW_RGB/BGR**
    - **HWC_RGB/BGR**
    - **CHW16/32/4/RGB/BGR for DLA input**
- Supported Features:
    - **Resize**
    - **Padding**
    - **Conversion**
    - **Normalization**

## ROI Conversion (ROIs To Continuous Tensor Conversion)
Combine Resize/Padding/Conversion/Normalization into a single kernel function.
- **Most of the time, it can be bit-aligned with OpenCV.**
    - It will give an exact result when the scaling factor is a rational number.
    - Better performance is usually achieved when the stride can divide by 4.
- Supported Input Format:
    - **NV12BlockLinear**
    - **NV12PitchLinear**
    - **YUV422Packed_YUYV**
- Supported Interpolation methods:
    - **Nearest**
    - **Bilinear**
- Supported Output Data Type:
    - **Uint8**
    - **Float32**
    - **Float16**
- Supported Output Layout:
    - **CHW_RGB/BGR**
    - **HWC_RGB/BGR**
    - **CHW16/32/4/RGB/BGR for DLA input**
    - **Gray**
- Supported Features:
    - **Resize**
    - **Padding**
    - **Conversion**
    - **Normalization**

## Thanks
This project makes use of a number of awesome open source libraries, including:

- [stb_image](https://github.com/nothings/stb) for PNG and JPEG support
- [pybind11](https://github.com/pybind/pybind11) for seamless C++ / Python interop
- and others! See the dependencies folder.

Many thanks to the authors of these brilliant projects!
