// RK3588 MPP H.264 硬解实现
// 支持两种输入: 整个 .h264 文件 (decode_file) / 单个 Annex B 包 (decode_packet, 供 RTSP 流式调用)
// 解码流程对齐 mpp/test/mpi_dec_test.c 的 dec_simple (async: put_packet + get_frame)
#include "mpp_decoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <utility>

MppDecoder::MppDecoder()
    : ctx_(NULL), mpi_(NULL), initialized_(false), frm_grp_(NULL), packet_(NULL) {}

MppDecoder::~MppDecoder() { deinit(); }

int MppDecoder::init()
{
    if (initialized_) return 0;

    // 1. 创建 MPP 上下文 + 函数表
    MPP_RET ret = mpp_create(&ctx_, &mpi_);
    if (ret != MPP_OK) {
        fprintf(stderr, "[mpp] mpp_create 失败 ret=%d\n", ret);
        return -1;
    }

    // 2. 解码器配置: 分块解析关闭, 快速解析开启 (降低首帧延迟)
    //    这些需在 mpp_init 之前设置才生效
    RK_U32 split_parse = 0;  // 关闭分块解析, 一次喂整帧
    mpi_->control(ctx_, MPP_DEC_SET_PARSER_SPLIT_MODE, &split_parse);
    RK_U32 fast_parse = 1;
    mpi_->control(ctx_, MPP_DEC_SET_PARSER_FAST_MODE, &fast_parse);

    // 3. 强制输出格式为 NV12 (MPP_FMT_YUV420SP), 与 RGA 后续处理对接
    MppFrameFormat out_fmt = MPP_FMT_YUV420SP;
    mpi_->control(ctx_, MPP_DEC_SET_OUTPUT_FORMAT, &out_fmt);

    // 4. 初始化为 H.264 解码器
    ret = mpp_init(ctx_, MPP_CTX_DEC, MPP_VIDEO_CodingAVC);
    if (ret != MPP_OK) {
        fprintf(stderr, "[mpp] mpp_init 失败 ret=%d\n", ret);
        mpp_destroy(ctx_);
        ctx_ = NULL;
        mpi_ = NULL;
        return -2;
    }

    // 5. 创建可复用的输入 MppPacket (跨多次 decode_packet 复用)
    ret = mpp_packet_init(&packet_, NULL, 0);
    if (ret != MPP_OK || !packet_) {
        fprintf(stderr, "[mpp] mpp_packet_init 失败\n");
        mpp_destroy(ctx_);
        ctx_ = NULL;
        mpi_ = NULL;
        return -3;
    }

    initialized_ = true;
    printf("[mpp] 初始化成功 (H.264 -> NV12)\n");
    return 0;
}

void MppDecoder::deinit()
{
    if (packet_) {
        mpp_packet_deinit(&packet_);
        packet_ = NULL;
    }
    if (frm_grp_) {
        mpp_buffer_group_put(frm_grp_);
        frm_grp_ = NULL;
    }
    if (initialized_) {
        mpp_destroy(ctx_);
        ctx_ = NULL;
        mpi_ = NULL;
        initialized_ = false;
    }
}

// 扫描 buf 中从 pos 开始的下一个 H.264 起始码 (00 00 01 或 00 00 00 01)
static size_t find_next_start_code(const unsigned char *buf, size_t size, size_t pos)
{
    while (pos + 2 < size) {
        if (buf[pos] == 0 && buf[pos + 1] == 0 && buf[pos + 2] == 1)
            return pos;
        pos++;
    }
    return size;
}

// 把整个 H.264 裸流切分成 NALU 列表 (含起始码), 每个 NALU = [offset, length)
static std::vector<std::pair<size_t, size_t>> split_nalus(const unsigned char *buf, size_t size)
{
    std::vector<std::pair<size_t, size_t>> nalus;
    size_t i = find_next_start_code(buf, size, 0);
    while (i < size) {
        size_t start = i;
        size_t p = find_next_start_code(buf, size, i + 3);
        size_t end = (p < size) ? p : size;
        nalus.push_back(std::make_pair(start, end - start));
        i = end;
    }
    return nalus;
}

int MppDecoder::handle_frame(MppFrame frame,
                              void (*on_frame)(const unsigned char *, int, int, int, int, int64_t, void *),
                              void *userdata)
{
    // info_change: 首帧时 MPP 上报分辨率/格式变化, 需配置 buffer 池后通知 ready
    if (mpp_frame_get_info_change(frame)) {
        RK_U32 width     = mpp_frame_get_width(frame);
        RK_U32 height    = mpp_frame_get_height(frame);
        RK_U32 hor_stride= mpp_frame_get_hor_stride(frame);
        RK_U32 ver_stride= mpp_frame_get_ver_stride(frame);
        size_t buf_size  = mpp_frame_get_buf_size(frame);
        printf("[mpp] info_change: %ux%u stride=%ux%u buf_size=%zu fmt=0x%x\n",
               width, height, hor_stride, ver_stride, buf_size,
               mpp_frame_get_fmt(frame));

        // 半内部模式: 创建/清空 buffer 池, 交给 MPP 管理, 限制 24 帧上限 (避免内存失控)
        if (!frm_grp_) {
            MPP_RET ret = mpp_buffer_group_get_internal(&frm_grp_, MPP_BUFFER_TYPE_ION);
            if (ret != MPP_OK) {
                fprintf(stderr, "[mpp] 创建 buffer group 失败 ret=%d\n", ret);
                return -1;
            }
            ret = mpi_->control(ctx_, MPP_DEC_SET_EXT_BUF_GROUP, frm_grp_);
            if (ret != MPP_OK) {
                fprintf(stderr, "[mpp] SET_EXT_BUF_GROUP 失败 ret=%d\n", ret);
                return -1;
            }
        } else {
            mpp_buffer_group_clear(frm_grp_);
        }
        mpp_buffer_group_limit_config(frm_grp_, buf_size, 24);

        MPP_RET ret = mpi_->control(ctx_, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
        if (ret != MPP_OK) {
            fprintf(stderr, "[mpp] SET_INFO_CHANGE_READY 失败 ret=%d\n", ret);
            return -1;
        }
        return 0;
    }
    // EOS 帧: 无像素, 仅标志结束
    if (mpp_frame_get_eos(frame)) {
        return 0;
    }
    // 错误帧 / 丢弃帧: 跳过
    if (mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame)) {
        return 0;
    }

    MppBuffer buf = mpp_frame_get_buffer(frame);
    if (!buf) return 0;

    const unsigned char *ptr = (const unsigned char *)mpp_buffer_get_ptr(buf);
    int w  = (int)mpp_frame_get_width(frame);
    int h  = (int)mpp_frame_get_height(frame);
    int hs = (int)mpp_frame_get_hor_stride(frame);
    int vs = (int)mpp_frame_get_ver_stride(frame);
    int64_t pts = mpp_frame_get_pts(frame);

    if (on_frame) on_frame(ptr, w, h, hs, vs, pts, userdata);
    return 1;
}

// 喂单个 Annex B 包, 流式接口 (RTSP 路径核心)
int MppDecoder::decode_packet(const unsigned char *data, size_t size, int64_t pts, bool eos,
                              void (*on_frame)(const unsigned char *, int, int, int, int, int64_t, void *),
                              void *userdata)
{
    if (!initialized_ || !packet_) {
        fprintf(stderr, "[mpp] 未初始化\n");
        return -1;
    }

    // 复用 packet: 设置数据指针/大小/pos/length + pts + eos
    mpp_packet_set_data(packet_, (void *)data);
    mpp_packet_set_size(packet_, size);
    mpp_packet_set_pos(packet_, (void *)data);
    mpp_packet_set_length(packet_, size);
    mpp_packet_set_pts(packet_, pts);
    if (eos) mpp_packet_set_eos(packet_);
    else     mpp_packet_clr_eos(packet_);

    int frame_cnt = 0;
    bool got_eos = false;
    int pkt_done = 0;

    do {
        // a. 送包 (队列满时重试)
        if (!pkt_done) {
            MPP_RET ret = mpi_->decode_put_packet(ctx_, packet_);
            if (ret == MPP_OK) {
                pkt_done = 1;
            } else {
                usleep(1000);
                continue;
            }
        }

        // b. 取所有可用帧
        bool frm_eos = false;
        do {
            MppFrame frame = NULL;
            MPP_RET ret = mpi_->decode_get_frame(ctx_, &frame);
            if (ret != MPP_OK && ret != MPP_ERR_TIMEOUT) {
                fprintf(stderr, "[mpp] decode_get_frame 失败 ret=%d\n", ret);
                break;
            }
            if (frame) {
                frm_eos = mpp_frame_get_eos(frame);
                int r = handle_frame(frame, on_frame, userdata);
                mpp_frame_deinit(&frame);
                if (r > 0) frame_cnt++;
                else if (r < 0) { got_eos = true; break; }
            }

            // 末包已送且还没收到 EOS 帧, 继续等
            if (eos && pkt_done && !frm_eos) {
                usleep(1000);
                continue;
            }
            if (frm_eos) {
                got_eos = true;
                break;
            }
            break;
        } while (1);

        if (got_eos) break;
        if (pkt_done) break;
        usleep(1000);
    } while (1);

    return frame_cnt;
}

int MppDecoder::decode_file(const char *path,
                              void (*on_frame)(const unsigned char *, int, int, int, int, int64_t, void *),
                              void *userdata)
{
    if (!initialized_) {
        fprintf(stderr, "[mpp] 未初始化\n");
        return -1;
    }

    // 1. 整文件读入内存
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[mpp] 打开文件失败: %s\n", path);
        return -2;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize <= 0) {
        fprintf(stderr, "[mpp] 文件为空: %s\n", path);
        fclose(fp);
        return -3;
    }
    unsigned char *buf = (unsigned char *)malloc(fsize);
    if (!buf) {
        fclose(fp);
        return -4;
    }
    size_t nread = fread(buf, 1, fsize, fp);
    fclose(fp);
    if ((long)nread != fsize) {
        fprintf(stderr, "[mpp] 读取不全: %zu/%ld\n", nread, fsize);
        free(buf);
        return -5;
    }
    printf("[mpp] 已读入 %s (%ld 字节)\n", path, fsize);

    // 2. 切分 NALU, 逐个喂 decode_packet
    std::vector<std::pair<size_t, size_t>> nalus = split_nalus(buf, (size_t)fsize);
    printf("[mpp] 切分出 %zu 个 NALU\n", nalus.size());

    int frame_cnt = 0;
    for (size_t i = 0; i < nalus.size(); i++) {
        size_t off = nalus[i].first;
        size_t len = nalus[i].second;
        bool eos = (i + 1 == nalus.size());
        frame_cnt += decode_packet(buf + off, len, 0, eos, on_frame, userdata);
    }

    free(buf);
    printf("[mpp] 解码完成, 共 %d 帧\n", frame_cnt);
    return frame_cnt;
}
