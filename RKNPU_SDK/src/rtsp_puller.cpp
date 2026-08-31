// FFmpeg RTSP 拉流实现: libavformat 打开 RTSP -> av_read_frame 取包
// -> h264_mp4toannexb 比特流过滤 (保证 Annex B + SPS/PPS 内联) -> 输出给 MPP
#include "rtsp_puller.h"

#include <stdio.h>
#include <string.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/dict.h>
#include <libavutil/opt.h>
#include <libavutil/rational.h>
}

RtspPuller::RtspPuller()
    : fmt_ctx_(NULL), bsf_ctx_(NULL), video_idx_(-1),
      width_(0), height_(0), tb_num_(0), tb_den_(0),
      cached_pkt_(NULL), has_cached_(false) {}

RtspPuller::~RtspPuller() { close(); }

int RtspPuller::open(const char *url)
{
    if (fmt_ctx_) {
        fprintf(stderr, "[rtsp] 已打开, 请先 close\n");
        return -1;
    }

    // 1. 初始化网络 (RTSP/RTP 需要)
    avformat_network_init();

    // 2. RTSP 低延迟选项: TCP 传输 + 禁用缓冲 + 短超时
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);      // TCP 避免丢包乱序
    av_dict_set(&opts, "stimeout",   "5000000", 0);       // socket 超时 5s (us)
    av_dict_set(&opts, "fflags",     "nobuffer", 0);      // 禁用格式层缓冲
    av_dict_set(&opts, "flags",      "low_delay", 0);     // 低延迟
    av_dict_set(&opts, "max_delay",  "500000", 0);        // 最大延时 0.5s (us)
    av_dict_set(&opts, "reorder_queue_size", "0", 0);     // 不做重排缓冲

    AVFormatContext *ic = NULL;
    int ret = avformat_open_input(&ic, url, NULL, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        char err[128];
        av_strerror(ret, err, sizeof(err));
        fprintf(stderr, "[rtsp] avformat_open_input 失败: %s (%s)\n", url, err);
        return -2;
    }
    fmt_ctx_ = ic;

    // 3. 探测流信息 (拿到 codecpar / time_base)
    ret = avformat_find_stream_info(ic, NULL);
    if (ret < 0) {
        fprintf(stderr, "[rtsp] avformat_find_stream_info 失败 ret=%d\n", ret);
        close();
        return -3;
    }

    // 4. 找 H.264 视频流
    int vidx = av_find_best_stream(ic, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vidx < 0) {
        fprintf(stderr, "[rtsp] 未找到视频流\n");
        close();
        return -4;
    }
    AVStream *st = ic->streams[vidx];
    if (st->codecpar->codec_id != AV_CODEC_ID_H264) {
        fprintf(stderr, "[rtsp] 视频流非 H.264 (codec_id=%d)\n", st->codecpar->codec_id);
        close();
        return -5;
    }
    video_idx_ = vidx;
    width_  = st->codecpar->width;
    height_ = st->codecpar->height;
    tb_num_ = st->time_base.num;
    tb_den_ = st->time_base.den;
    printf("[rtsp] 已连接 %s  H.264 %dx%d  time_base=%d/%d\n",
           url, width_, height_, tb_num_, tb_den_);

    // 5. 创建 h264_mp4toannexb 比特流过滤器
    //    作用: 把 AVCC (length-prefixed) / RTP 装包转成 Annex B (00 00 00 01 起始码),
    //          并把 extradata 里的 SPS/PPS 内联到每个 IDR 前面 —— MPP 必须靠 SPS/PPS 初始化
    const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
    if (!bsf) {
        fprintf(stderr, "[rtsp] 未找到 h264_mp4toannexb 过滤器\n");
        close();
        return -6;
    }
    AVBSFContext *bsf_ctx = NULL;
    ret = av_bsf_alloc(bsf, &bsf_ctx);
    if (ret < 0) {
        fprintf(stderr, "[rtsp] av_bsf_alloc 失败 ret=%d\n", ret);
        close();
        return -7;
    }
    // 把流的 codecpar 拷到 BSF 输入端
    ret = avcodec_parameters_copy(bsf_ctx->par_in, st->codecpar);
    if (ret < 0) {
        fprintf(stderr, "[rtsp] avcodec_parameters_copy 失败 ret=%d\n", ret);
        av_bsf_free(&bsf_ctx);
        close();
        return -8;
    }
    ret = av_bsf_init(bsf_ctx);
    if (ret < 0) {
        fprintf(stderr, "[rtsp] av_bsf_init 失败 ret=%d\n", ret);
        av_bsf_free(&bsf_ctx);
        close();
        return -9;
    }
    bsf_ctx_ = bsf_ctx;
    return 0;
}

int RtspPuller::read_packet(const unsigned char **out_data, int *out_size, int64_t *out_pts)
{
    if (!fmt_ctx_ || !bsf_ctx_) return -1;

    AVFormatContext *ic = (AVFormatContext *)fmt_ctx_;
    AVBSFContext *bsf = (AVBSFContext *)bsf_ctx_;

    // 循环: 跳过非视频包, 直到拿到一个视频包并经 BSF 转换
    while (true) {
        AVPacket pkt;
        av_init_packet(&pkt);
        int ret = av_read_frame(ic, &pkt);
        if (ret < 0) {
            // 流结束或出错
            if (ret == AVERROR_EOF || ret == AVERROR(EIO))
                return 0;  // EOS
            char err[128];
            av_strerror(ret, err, sizeof(err));
            fprintf(stderr, "[rtsp] av_read_frame 失败: %s\n", err);
            return -2;
        }

        if (pkt.stream_index != video_idx_) {
            av_packet_unref(&pkt);
            continue;  // 非视频流 (音频等), 跳过
        }

        // 送进 BSF
        ret = av_bsf_send_packet(bsf, &pkt);
        av_packet_unref(&pkt);
        if (ret < 0) {
            fprintf(stderr, "[rtsp] av_bsf_send_packet 失败 ret=%d\n", ret);
            continue;
        }

        // 取出 Annex B 包 (可能一次 send 产生 0 或 1 个输出)
        AVPacket out;
        av_init_packet(&out);
        ret = av_bsf_receive_packet(bsf, &out);
        if (ret < 0) {
            // AVERROR(EAGAIN): 暂无输出, 继续读下一帧
            continue;
        }

        *out_data = out.data;
        *out_size = out.size;
        // pts 换算成微秒
        if (out.pts != AV_NOPTS_VALUE && tb_den_ > 0)
            *out_pts = (int64_t)out.pts * tb_num_ * 1000000 / tb_den_;
        else
            *out_pts = 0;

        // out 由 BSF 内部管理, 缓存到成员; 下次 read_packet 前释放上一包
        // (调用方拿到的指针在下次 read_packet 前一直有效)
        if (has_cached_) {
            AVPacket *c = (AVPacket *)cached_pkt_;
            av_packet_unref(c);
            av_packet_free(&c);
        }
        AVPacket *c = (AVPacket *)av_mallocz(sizeof(AVPacket));
        av_init_packet(c);
        av_packet_move_ref(c, &out);
        cached_pkt_ = c;
        has_cached_ = true;
        *out_data = c->data;
        *out_size = c->size;
        return 1;
    }
}

void RtspPuller::close()
{
    if (has_cached_ && cached_pkt_) {
        AVPacket *c = (AVPacket *)cached_pkt_;
        av_packet_unref(c);
        av_packet_free(&c);
        cached_pkt_ = NULL;
        has_cached_ = false;
    }
    if (bsf_ctx_) {
        AVBSFContext *bsf = (AVBSFContext *)bsf_ctx_;
        av_bsf_free(&bsf);
        bsf_ctx_ = NULL;
    }
    if (fmt_ctx_) {
        AVFormatContext *ic = (AVFormatContext *)fmt_ctx_;
        avformat_close_input(&ic);
        fmt_ctx_ = NULL;
    }
    video_idx_ = -1;
    width_ = height_ = 0;
    tb_num_ = tb_den_ = 0;
}

int RtspPuller::width() const  { return width_; }
int RtspPuller::height() const { return height_; }
