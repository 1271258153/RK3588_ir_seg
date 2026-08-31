# 部署至RK3588推理
## 1. 📁 仓库目录说明

## 2. 准备工作
### 1. 构建CMake工程
1. `把导出 best.rknn 拷贝到 `model/RK3588` 目录下
2. 确保交叉编译器和 `build-linux_RK3588.sh` 脚本中 GCC_COMPILER 的路径一致
3. 将第三方库 `include/` 和 `opencv/` 拷贝到当前工程中
4. 修改 `CMakeLists.txt` 修改如下参数
    |变量|含义|
    |-|-|
    |project|项目名称|
    |# rknn api|第三方库 `include/`|
    |# opencv 路径| opencv库 `opencv/`|
    |add_executable|工程名和 main 函数路径|
    |# install target and libraries|在 `install/` 生成 pidnet_Linux|
5. 执行脚本 `./build-linux_RK3588.sh` 会生成 `build/` 和 `install/` 目录 

### 2. 调用 RGA API 对图像进行预处理
**RGA** (Raster Graphic Acceleration Unit) 是一个独立的 2D 硬件加速器，可用于加速点/线绘制，执行图像缩放、旋转、 bitBlt、 alpha 混合等常见的 2D 图形操作。
- 查询模型并打印所有输入/输出张量属性，用于确认分割模型的 I/O 格式
把整个 install/pidnet_Linux/ 目录拷到 RK3588 板子的根目录，然后运行：
```bash
adb push install /          # 拷贝到开发板的根目录
# 打开 MobaXterm
cd /install/pidnet_Linux
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
./rknn_query ./model/RK3588/best.rknn
```
- 得到 
> 输入：[1,640,640,3] INT8 NHWC → 640×640 RGB 3通道（红外图复制成3通道）
> 输出：[1,10,80,80] FLOAT16 NCHW → 10类在80×80分辨率，需对通道维做 argmax

### 3. 查看分割的mask和叠加图
```bash
./build-linux_RK3588.sh             # 重新编译
adb push install /
adb push pic /install/pidnet_Linux  # 拷贝测试的图片
# 进入开发板
cd /install/pidnet_Linux
./seg_single ./model/RK3588/best.rknn pic/img/xxx.png
```

### 4. 评估rknn模型的miou
```bash
# 进入开发板
cd /install/pidnet_Linux
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
./eval_miou ./model/RK3588/best.rknn pic/test.txt
```

### 5. mpp 硬解码测试
```bash
adb push pic/test.h264 /install/pidnet_Linux/   # 拷贝测试的流文件
# 进入开发板
./dec_test test.h264 h264_test 20               # 存20帧图片到 h264_test/下
```

### 6. RTSP 拉流 -> MPP 硬解 -> RGA -> NPU 推理 (rtsp_seg)
完整视频流管线: FFmpeg 拉 RTSP -> 剥离协议容器得 H.264 包 -> MPP 硬解出 NV12 -> RGA 缩放转 RGB640x640 -> NPU 语义分割 -> 保存叠加图。
#### 6.1 准备 aarch64 FFmpeg 开发库 (从板子拷贝)
交叉编译需要 aarch64 的 FFmpeg 头文件 + .so。板子上已装 ffmpeg, 在板子执行:
```bash
# 在 RK3588 板子上: 打包头文件 + .so
cd /tmp
mkdir -p ffmpeg/include ffmpeg/lib
cp -a /usr/include/libavformat /usr/include/libavcodec /usr/include/libavutil /usr/include/libswscale ffmpeg/include/
cp -a /usr/lib/aarch64-linux-gnu/libavformat.so* ffmpeg/lib/
cp -a /usr/lib/aarch64-linux-gnu/libavcodec.so*  ffmpeg/lib/
cp -a /usr/lib/aarch64-linux-gnu/libavutil.so*   ffmpeg/lib/
cp -a /usr/lib/aarch64-linux-gnu/libswscale.so*  ffmpeg/lib/
tar czf ffmpeg.tar.gz ffmpeg
```
```bash
# 在交叉编译主机上: 拉回并解压到 3rdparty
scp root@<板子IP>:/tmp/ffmpeg.tar.gz /tmp/
cd /home/topeet/RKNPU_SDK/include/3rdparty
rm -rf ffmpeg && tar xzf /tmp/ffmpeg.tar.gz
```

#### 6.2 编译
```bash
./build-linux_RK3588.sh        # 生成 rtsp_seg
adb push install /
```

#### 6.3 运行 (PC 上先启动推流; 板子IP 192.168.137.10, 虚拟机IP 192.168.137.3)
```bash
# 板子上执行 (拉流地址指向 PC 的 MediaMTX)
cd /install/pidnet_Linux
./rtsp_seg rtsp://192.168.137.3:8554/cam model/RK3588/best.rknn h264_test 20 3  # 使用3线程(最多6线程)推理出20张图片在 h264_test/下
```
