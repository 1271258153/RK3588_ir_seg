// H.264 本地硬解 + RGA 后处理 验证程序
// 流程: 读 .h264 -> MPP 切分 NALU 硬解出 NV12 -> RGA 转 BGR -> 存 PNG 验证
//
// 用法: ./dec_test [h264路径] [输出目录] [最大保存帧数]
//   默认: ./dec_test pic/test.h264 output 20

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "mpp_decoder.h"
#include "rga_utils.h"

// 回调上下文: 保存参数 + 统计
struct FrameCtx
{
    const char *out_dir;   // 输出目录
    int max_save;          // 最多保存多少帧 PNG
    int saved;             // 已保存帧数
    int total;             // 总帧数 (含未保存)
    int last_w, last_h, last_hs, last_vs;
};

// 确保 output 目录存在 (与 seg_main 一致, 兼容旧 glibc)
static int ensure_dir(const char *dir)
{
    if (mkdir(dir, 0755) == 0) return 0;
    if (errno == EEXIST) {
        DIR *d = opendir(dir);
        if (d) { closedir(d); return 0; }
        fprintf(stderr, "[output] %s 已存在但不是目录\n", dir);
        return -1;
    }
    perror("[output] mkdir 失败");
    return -2;
}

// 每帧回调: NV12 -> BGR -> 存 PNG
static void on_frame(const unsigned char *nv12, int width, int height,
                     int hor_stride, int ver_stride, int64_t pts, void *userdata)
{
    FrameCtx *ctx = (FrameCtx *)userdata;
    ctx->total++;
    ctx->last_w = width; ctx->last_h = height;
    ctx->last_hs = hor_stride; ctx->last_vs = ver_stride;

    printf("[frame#%d] %dx%d stride=%dx%d pts=%lld\n",
           ctx->total, width, height, hor_stride, ver_stride, (long long)pts);

    if (ctx->saved >= ctx->max_save) return;  // 超出保存上限只计数不存盘

    // RGA: NV12 -> BGR888 (硬件单 pass 色彩转换)
    unsigned char *bgr = (unsigned char *)malloc((size_t)width * height * 3);
    if (!bgr) {
        fprintf(stderr, "[frame#%d] 分配 BGR buffer 失败\n", ctx->total);
        return;
    }
    if (rga_nv12_to_bgr(nv12, width, height, hor_stride, ver_stride, bgr) != 0) {
        fprintf(stderr, "[frame#%d] RGA NV12->BGR 失败\n", ctx->total);
        free(bgr);
        return;
    }

    // 包装成 cv::Mat 存盘 (零拷贝, bgr 行步 = width*3 紧凑)
    cv::Mat img(height, width, CV_8UC3, bgr);
    char path[512];
    snprintf(path, sizeof(path), "%s/frame_%04d.png", ctx->out_dir, ctx->saved);
    if (cv::imwrite(path, img)) {
        printf("  -> 已保存 %s\n", path);
        ctx->saved++;
    } else {
        fprintf(stderr, "  -> 保存 %s 失败\n", path);
    }
    free(bgr);
}

int main(int argc, char **argv)
{
    const char *h264_path = (argc > 1) ? argv[1] : "pic/test.h264";
    const char *out_dir   = (argc > 2) ? argv[2] : "output";
    int max_save          = (argc > 3) ? atoi(argv[3]) : 20;

    printf("输入: %s\n输出目录: %s\n最大保存帧数: %d\n", h264_path, out_dir, max_save);

    if (ensure_dir(out_dir) != 0) {
        fprintf(stderr, "创建输出目录失败: %s\n", out_dir);
        return -1;
    }

    // 1. 初始化 MPP 解码器
    MppDecoder dec;
    if (dec.init() != 0) {
        fprintf(stderr, "MPP 初始化失败\n");
        return -2;
    }

    // 2. 解码 + 每帧回调 (NV12 经 RGA 转 BGR 存盘)
    FrameCtx ctx;
    ctx.out_dir = out_dir;
    ctx.max_save = max_save;
    ctx.saved = 0;
    ctx.total = 0;
    ctx.last_w = ctx.last_h = ctx.last_hs = ctx.last_vs = 0;

    int n = dec.decode_file(h264_path, on_frame, &ctx);
    if (n < 0) {
        fprintf(stderr, "解码失败: %d\n", n);
        dec.deinit();
        return -3;
    }

    // 3. 汇总
    printf("\n===== 解码汇总 =====\n");
    printf("总帧数: %d\n", ctx.total);
    printf("画面尺寸: %dx%d (stride %dx%d)\n",
           ctx.last_w, ctx.last_h, ctx.last_hs, ctx.last_vs);
    printf("已保存 PNG: %d/%d\n", ctx.saved, ctx.max_save);
    printf("输出格式: NV12 (MPP_FMT_YUV420SP) -> RGA -> BGR888 PNG\n");

    dec.deinit();
    printf("完成\n");
    return 0;
}
