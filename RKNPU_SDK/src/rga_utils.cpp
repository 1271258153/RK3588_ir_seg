/* RGA 对图像进行预处理 */
#include "rga_utils.h"

#include <stdio.h>
#include <string.h>

#include <opencv2/imgproc.hpp>

#include "im2d.h"
#include "rga.h"

// RGA 成功状态码为正 (IM_STATUS_SUCCESS=1, IM_STATUS_NOERROR=2), 失败 <= 0
static inline bool rga_ok(IM_STATUS s) { return (int)s > 0; }

// RGA 硬件对 BGR888/RGB888 源要求宽高 16 对齐, 不对齐时用边缘复制填充到对齐尺寸
static cv::Mat align_to_16(const cv::Mat &img)
{
    int w = img.cols, h = img.rows;
    int aw = (w + 15) & ~15;   // 向上取整到 16 的倍数
    int ah = (h + 15) & ~15;
    if (aw == w && ah == h) return img;
    cv::Mat padded;
    // BORDER_REPLICATE: 用最近边缘像素填充, 避免引入黑边干扰模型
    cv::copyMakeBorder(img, padded, 0, ah - h, 0, aw - w, cv::BORDER_REPLICATE);
    return padded;
}

int rga_preprocess(const cv::Mat &bgr_img, int dst_w, int dst_h,
                   unsigned char *out_buf)
{
    if (bgr_img.empty()) {
        fprintf(stderr, "[rga] 输入图像为空\n");
        return -1;
    }
    if (!out_buf) {
        fprintf(stderr, "[rga] 输出 buffer 为空\n");
        return -2;
    }
    if (bgr_img.channels() != 3) {
        fprintf(stderr, "[rga] 仅支持 3 通道 BGR 输入, 实际通道数=%d\n", bgr_img.channels());
        return -3;
    }

    // RGA 要求 BGR888 源宽高 16 对齐, 不对齐则边缘复制填充
    cv::Mat aligned = align_to_16(bgr_img);
    int src_w = aligned.cols;
    int src_h = aligned.rows;

    // 用 wrapbuffer_virtualaddr 包装物理连续虚拟地址, src=BGR888, dst=RGB888
    // imresize 在硬件单次 pass 内同时完成 缩放 + BGR2RGB
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)aligned.data, src_w, src_h,
                                               RK_FORMAT_BGR_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)out_buf, dst_w, dst_h,
                                              RK_FORMAT_RGB_888);
    if (src.width == 0 || dst.width == 0) {
        fprintf(stderr, "[rga] wrapbuffer_virtualaddr 失败\n");
        return -4;
    }

    im_rect src_rect, dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    // imcheck 校验 src/dst 参数合法性
    IM_STATUS status = imcheck(src, dst, src_rect, dst_rect);
    if (!rga_ok(status)) {
        fprintf(stderr, "[rga] imcheck 失败: %s\n", imStrError(status));
        return -5;
    }

    // 单次 imresize 完成缩放 + 色彩转换 (RGA 硬件支持跨格式 1D/2D 引擎)
    status = imresize(src, dst);
    if (!rga_ok(status)) {
        fprintf(stderr, "[rga] imresize 失败: %s\n", imStrError(status));
        return -6;
    }

    return 0;
}

// NV12 -> BGR888: RGA 单次 imresize 跨格式转换 (YUV420SP -> BGR888)
// src 用 hor_stride/ver_stride 描述内存跨距, width/height 描述有效画面
int rga_nv12_to_bgr(const unsigned char *nv12, int width, int height,
                    int hor_stride, int ver_stride, unsigned char *bgr_out)
{
    if (!nv12 || !bgr_out) {
        fprintf(stderr, "[rga] nv12_to_bgr 输入/输出为空\n");
        return -1;
    }
    if (hor_stride < width || ver_stride < height) {
        fprintf(stderr, "[rga] stride(%dx%d) < 画面(%dx%d)\n", hor_stride, ver_stride, width, height);
        return -2;
    }

    // RGA 要求 BGR888 输出 width stride 16 对齐; width 非 16 对齐(如 1080)时,
    // 先 RGA 到对齐 stride 的临时 buffer, 再逐行紧凑拷贝到 bgr_out (width*height*3)
    const int dst_wstride = (width + 15) & ~15;
    const bool need_align = (dst_wstride != width);

    unsigned char *dst_buf = bgr_out;
    unsigned char *align_buf = NULL;
    if (need_align) {
        align_buf = (unsigned char *)malloc((size_t)dst_wstride * height * 3);
        if (!align_buf) {
            fprintf(stderr, "[rga] nv12_to_bgr 分配对齐 buffer 失败\n");
            return -3;
        }
        dst_buf = align_buf;
    }

    // src: NV12, 跨距 = hor_stride x ver_stride; dst: BGR888, 跨距 = dst_wstride x height
    // 注意 wrapbuffer_virtualaddr 宏签名是 (ptr, width, height, format, wstride, hstride)
    // format 是第 4 个位置参数, stride 放在可变参数里
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)nv12, width, height,
                                              RK_FORMAT_YCbCr_420_SP,
                                              hor_stride, ver_stride);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)dst_buf, width, height,
                                              RK_FORMAT_BGR_888,
                                              dst_wstride, height);
    if (src.width == 0 || dst.width == 0) {
        fprintf(stderr, "[rga] nv12_to_bgr wrapbuffer 失败\n");
        if (align_buf) free(align_buf);
        return -4;
    }

    im_rect src_rect, dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    IM_STATUS status = imcheck(src, dst, src_rect, dst_rect);
    if (!rga_ok(status)) {
        fprintf(stderr, "[rga] nv12_to_bgr imcheck 失败: %s\n", imStrError(status));
        if (align_buf) free(align_buf);
        return -5;
    }

    // imresize 在硬件单 pass 内完成 YUV->RGB 色彩转换 + (无缩放) 拷贝
    status = imresize(src, dst);
    if (!rga_ok(status)) {
        fprintf(stderr, "[rga] nv12_to_bgr imresize 失败: %s\n", imStrError(status));
        if (align_buf) free(align_buf);
        return -6;
    }

    if (need_align) {
        const size_t tight_row = (size_t)width * 3;
        const size_t align_row = (size_t)dst_wstride * 3;
        for (int h = 0; h < height; h++)
            memcpy(bgr_out + (size_t)h * tight_row,
                   align_buf + (size_t)h * align_row, tight_row);
        free(align_buf);
    }

    return 0;
}

// NV12 -> RGB888 缩放转换: RGA 单次 imresize 跨格式 + 缩放 (YUV420SP -> RGB888)
// src 用 hor_stride/ver_stride 描述内存跨距, dst 紧凑 dst_w x dst_h
int rga_nv12_to_rgb_resized(const unsigned char *nv12, int width, int height,
                            int hor_stride, int ver_stride,
                            unsigned char *rgb_out, int dst_w, int dst_h)
{
    if (!nv12 || !rgb_out) {
        fprintf(stderr, "[rga] nv12_to_rgb_resized 输入/输出为空\n");
        return -1;
    }
    if (hor_stride < width || ver_stride < height) {
        fprintf(stderr, "[rga] stride(%dx%d) < 画面(%dx%d)\n", hor_stride, ver_stride, width, height);
        return -2;
    }

    // 注意 wrapbuffer_virtualaddr 宏签名是 (ptr, width, height, format, wstride, hstride)
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)nv12, width, height,
                                               RK_FORMAT_YCbCr_420_SP,
                                               hor_stride, ver_stride);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)rgb_out, dst_w, dst_h,
                                              RK_FORMAT_RGB_888,
                                              dst_w, dst_h);
    if (src.width == 0 || dst.width == 0) {
        fprintf(stderr, "[rga] nv12_to_rgb_resized wrapbuffer 失败\n");
        return -3;
    }

    im_rect src_rect, dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    IM_STATUS status = imcheck(src, dst, src_rect, dst_rect);
    if (!rga_ok(status)) {
        fprintf(stderr, "[rga] nv12_to_rgb_resized imcheck 失败: %s\n", imStrError(status));
        return -4;
    }

    // 单 pass: 缩放 (width x height -> dst_w x dst_h) + 色彩转换 (NV12 -> RGB888)
    status = imresize(src, dst);
    if (!rga_ok(status)) {
        fprintf(stderr, "[rga] nv12_to_rgb_resized imresize 失败: %s\n", imStrError(status));
        return -5;
    }

    return 0;
}
