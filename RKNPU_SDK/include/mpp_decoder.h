#ifndef _MPP_DECODER_H
#define _MPP_DECODER_H

#include "rk_mpi.h"
#include "mpp_buffer.h"
#include "mpp_frame.h"
#include "mpp_packet.h"

// RK3588 MPP 硬件 H.264 解码器: 读 .h264 文件 -> 切分 NALU -> 逐包喂 MPP -> 吐 NV12 帧
//
// 输出格式固定为 NV12 (MPP_FMT_YUV420SP, YYYY...UVUV...)
// 内存布局: Y 平面 = hor_stride * ver_stride, UV 平面紧随其后, 大小 = hor_stride * ver_stride / 2
// 注意 hor_stride >= width (16 对齐), ver_stride >= height (16 对齐), 取像素时需按 stride 跳行
class MppDecoder
{
public:
    MppDecoder();
    ~MppDecoder();

    // 初始化 MPP 解码上下文 (H.264 / AVC), 输出格式强制 NV12
    // 返回 0 成功, 负数失败
    int init();

    // 解码整个 .h264 文件, 每解出一帧 NV12 调用一次 on_frame
    //   nv12       : NV12 像素数据指针 (MPP 内部 buffer, 回调内有效, 用完即还)
    //   width      : 有效画面宽
    //   height     : 有效画面高
    //   hor_stride : Y 行跨距 (像素), >= width, 16 对齐
    //   ver_stride : Y 列跨距 (行),  >= height, 16 对齐
    //   pts        : 帧时间戳 (us)
    //   userdata   : 透传给回调的用户指针
    // 返回解出的帧数, 负数失败
    int decode_file(const char *path,
                     void (*on_frame)(const unsigned char *nv12,
                                      int width, int height,
                                      int hor_stride, int ver_stride,
                                      int64_t pts, void *userdata),
                     void *userdata);

    // 喂单个 H.264 包 (Annex B, 含起始码; 一个包可含多 NALU/一帧), 流式接口
    //   data/size : Annex B 码流 (FFmpeg AVPacket 经 h264_mp4toannexb 后的内存)
    //   pts       : 该包时间戳 (us), 透传给 MPP, 解出的帧继承此 pts
    //   eos       : 是否为最后一个包 (流结束), 触发排空
    //   on_frame  : 每解出一帧 NV12 调用一次 (同 decode_file)
    // 返回本次调用新吐出的帧数, 负数失败
    int decode_packet(const unsigned char *data, size_t size, int64_t pts, bool eos,
                      void (*on_frame)(const unsigned char *, int, int, int, int, int64_t, void *),
                      void *userdata);

    // 释放 MPP 上下文
    void deinit();

private:
    // 处理一帧 MppFrame: 处理 info_change / errinfo / 调回调
    // 返回 1 表示成功吐出一帧并调了回调, 0 表示该 frame 无有效输出
    int handle_frame(MppFrame frame,
                     void (*on_frame)(const unsigned char *, int, int, int, int, int64_t, void *),
                     void *userdata);

    MppCtx  ctx_;
    MppApi *mpi_;
    bool    initialized_;

    // 解码输出帧 buffer 池 (info_change 时创建, 半内部模式, 限制 24 帧上限)
    MppBufferGroup frm_grp_;

    // 可复用的输入包 (init 时创建, 跨多次 decode_packet 复用, deinit 释放)
    MppPacket packet_;
};

#endif // _MPP_DECODER_H
