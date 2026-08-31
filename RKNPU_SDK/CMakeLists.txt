cmake_minimum_required(VERSION 3.4.1)

# 项目名称
project(pidnet)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# CMAKE_INSTALL_PREFIX 与 CMAKE_INSTALL_RPATH 在文件末尾设置

# rknn api
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/include/librknnrt.so)
include_directories(${CMAKE_SOURCE_DIR}/include/3rdparty)

# 导出 compile_commands.json 供 clangd 索引 (含所有 include 路径)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# opencv 路径
set(OpenCV_PATH ${CMAKE_SOURCE_DIR}/opencv)
set(OpenCV_LIBS 
  ${OpenCV_PATH}/lib/libopencv_core.so
  ${OpenCV_PATH}/lib/libopencv_highgui.so
  ${OpenCV_PATH}/lib/libopencv_imgproc.so
  ${OpenCV_PATH}/lib/libopencv_videoio.so
  ${OpenCV_PATH}/lib/libopencv_imgcodecs.so
  ${OpenCV_PATH}/lib/libopencv_core.so.407
  ${OpenCV_PATH}/lib/libopencv_highgui.so.407
  ${OpenCV_PATH}/lib/libopencv_imgproc.so.407
  ${OpenCV_PATH}/lib/libopencv_videoio.so.407
  ${OpenCV_PATH}/lib/libopencv_imgcodecs.so.407
  ${OpenCV_PATH}/lib/libopencv_core.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_highgui.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_imgproc.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_videoio.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_imgcodecs.so.4.7.0
)
include_directories( ${OpenCV_PATH}/include/opencv4)

#rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(LIB_ARCH aarch64)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)
include_directories( ${RGA_PATH}/include)
set(CMAKE_INSTALL_RPATH "lib")

# mpp (RK3588 H.264/H.265 硬解)
set(MPP_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/mpp)
set(MPP_LIB ${MPP_PATH}/lib/Linux/${LIB_ARCH}/librockchip_mpp.so)
set(MPP_LIBS
  ${MPP_PATH}/lib/Linux/${LIB_ARCH}/librockchip_mpp.so
  ${MPP_PATH}/lib/Linux/${LIB_ARCH}/librockchip_mpp.so.1
  ${MPP_PATH}/lib/Linux/${LIB_ARCH}/librockchip_mpp.so.0)
include_directories(${MPP_PATH}/include)

# ffmpeg (RTSP 拉流: libavformat/libavcodec/libavutil)
# 需先从板子拷贝 aarch64 ffmpeg 头文件与 .so 到此目录 (见 README)
set(FFMPEG_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/ffmpeg)
set(FFMPEG_LIBS
  ${FFMPEG_PATH}/lib/libavformat.so
  ${FFMPEG_PATH}/lib/libavformat.so.58
  ${FFMPEG_PATH}/lib/libavformat.so.58.29.100
  ${FFMPEG_PATH}/lib/libavcodec.so
  ${FFMPEG_PATH}/lib/libavcodec.so.58
  ${FFMPEG_PATH}/lib/libavcodec.so.58.54.100
  ${FFMPEG_PATH}/lib/libavutil.so
  ${FFMPEG_PATH}/lib/libavutil.so.56
  ${FFMPEG_PATH}/lib/libavutil.so.56.31.100
  ${FFMPEG_PATH}/lib/libswscale.so
  ${FFMPEG_PATH}/lib/libswscale.so.5
  ${FFMPEG_PATH}/lib/libswscale.so.5.5.100)
include_directories(${FFMPEG_PATH}/include)

# 工程头文件目录
include_directories( ${CMAKE_SOURCE_DIR}/include)

# rknn_query: 模型 I/O 属性查询工具
add_executable(rknn_query src/rknn_query.cc)
target_link_libraries(rknn_query ${RKNN_RT_LIB})

# seg_single: 模块化单图分割推理 (rga_utils + rknn_infer + seg_main)
add_executable(seg_single
        src/rga_utils.cpp
        src/rknn_infer.cpp
        src/seg_main.cc)
target_link_libraries(seg_single
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS})

# eval_miou: 批量评估 mIoU/混淆矩阵 (复用 rga_utils + rknn_infer)
add_executable(eval_miou
        src/rga_utils.cpp
        src/rknn_infer.cpp
        src/eval_miou.cc)
target_link_libraries(eval_miou
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS})

# dec_test: H.264 本地硬解 (MPP -> NV12) + RGA 转 BGR 存盘验证
add_executable(dec_test
        src/mpp_decoder.cpp
        src/rga_utils.cpp
        src/dec_main.cc)
target_link_libraries(dec_test
  ${MPP_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS})

# rtsp_seg: RTSP 拉流 -> MPP 硬解 -> RGA 预处理 -> NPU 语义分割推理 -> 叠加图
add_executable(rtsp_seg
        src/rtsp_puller.cpp
        src/mpp_decoder.cpp
        src/rga_utils.cpp
        src/rknn_infer.cpp
        src/rtsp_seg_main.cc)
target_link_libraries(rtsp_seg
  ${FFMPEG_LIBS}
  ${MPP_LIB}
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS})

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/pidnet_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_query seg_single eval_miou dec_test rtsp_seg DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(PROGRAMS ${MPP_LIBS} DESTINATION lib)
install(PROGRAMS ${FFMPEG_LIBS} DESTINATION lib)
install(PROGRAMS ${OpenCV_LIBS} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
