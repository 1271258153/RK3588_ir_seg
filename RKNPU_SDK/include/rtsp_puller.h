#ifndef _RTSP_PULLER_H
#define _RTSP_PULLER_H

#include <stdint.h>

// FFmpeg RTSP 拉流器: 打开 RTSP 地址 -> 剥离网络协议/容器 -> 输出 Annex B H.264 包
// 输出的每个包 (含 SPS/PPS/IDR 或普通 slice) 直接喂给 MppDecoder::decode_packet
class RtspPuller
{
public:
    RtspPuller();
    ~RtspPuller();

    // 打开 RTSP 流 (默认 TCP 传输, 低延迟选项)
    //   url : rtsp://host[:port]/path
    // 返回 0 成功, 负数失败
    int open(const char *url);

    // 读取一个 H.264 Annex B 包 (经 h264_mp4toannexb 比特流过滤, 含起始码)
    //   out_data : 输出包数据指针 (内部缓冲, 在下一次 read_packet 前有效, 不要 free)
    //   out_size : 输出包字节数
    //   out_pts  : 输出包时间戳 (微秒)
    // 返回 1 成功取到包, 0 流结束 (EOS), 负数失败
    int read_packet(const unsigned char **out_data, int *out_size, int64_t *out_pts);

    // 关闭流, 释放 FFmpeg 资源
    void close();

    // 流的宽高 (open 成功后有效, 否则 0)
    int width() const;
    int height() const;

private:
    void *fmt_ctx_;     // AVFormatContext*
    void *bsf_ctx_;      // AVBSFContext* (h264_mp4toannexb)
    int  video_idx_;     // 视频流索引
    int  width_;
    int  height_;
    // 时间基 (us 换算): pts_us = pts * time_base_num * 1e6 / time_base_den
    int  tb_num_;
    int  tb_den_;

    // 上一包缓存 (BSF 输出, 下次 read_packet 前释放; void* 避免暴露 FFmpeg 头)
    void *cached_pkt_;
    bool  has_cached_;
};

#endif // _RTSP_PULLER_H
