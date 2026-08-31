// RTSP 拉流 -> MPP 硬解 -> [线程池] RGA 预处理 -> NPU 语义分割推理 (3核并行) -> 保存叠加图
// 多线程流水线: N 个 NPU 实例绑核 i%3, ThreadPool 异步推理, 解码与推理重叠
//
// 用法: ./rtsp_seg <rtsp_url> <rknn_model> <output_dir> [max_save] [thread_num]
//   ./rtsp_seg rtsp://127.0.0.1:8554/cam model/RK3588/best.rknn output 20 6

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>
#include <chrono>
#include <queue>
#include <future>
#include <vector>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rtsp_puller.h"
#include "mpp_decoder.h"
#include "rga_utils.h"
#include "rknn_infer.h"
#include "ThreadPool.hpp"

// 线程池大小 (并发推理线程数 = NPU 实例数)
//   RK3588 NPU 有 3 个核心, 各实例按 i%3 绑定到 CORE_0/1/2
//   推荐取 3 的倍数 (3 或 6), 6 时每核 2 实例轮询, 吞吐更高
#define DEFAULT_THREAD_NUM 6

// 10 类颜色 [R,G,B] -> OpenCV BGR (与训练 class_name.txt 一致)
static const cv::Vec3b kSegColors[10] = {
    cv::Vec3b(0, 0, 0),         // 0 _background_
    cv::Vec3b(60, 20, 220),     // 1 BL_Device
    cv::Vec3b(255, 144, 30),    // 2 CC_Server
    cv::Vec3b(50, 205, 50),     // 3 DP_Server
    cv::Vec3b(0, 165, 255),     // 4 KDVideo_Device
    cv::Vec3b(211, 0, 148),     // 5 KVM_Switcher
    cv::Vec3b(209, 206, 0),     // 6 SP_Cloud
    cv::Vec3b(0, 215, 255),     // 7 VPN_Gateway
    cv::Vec3b(180, 105, 255),   // 8 WEB_Firewall
    cv::Vec3b(128, 128, 128),   // 9 YP_Server
};

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

// 离散标签图逐像素查表上色
static cv::Mat colorize(const cv::Mat &label)
{
    cv::Mat out(label.size(), CV_8UC3);
    for (int h = 0; h < label.rows; h++) {
        const uchar *lr = label.ptr<uchar>(h);
        cv::Vec3b *cr = out.ptr<cv::Vec3b>(h);
        for (int w = 0; w < label.cols; w++)
            cr[w] = kSegColors[lr[w] % 10];
    }
    return out;
}

// 管线上下文: 跨回调共享
struct PipelineCtx
{
    std::vector<RKNNInfer *> engines;   // N 个 NPU 实例, 各绑不同核
    dpool::ThreadPool *pool;            // 线程池
    std::queue<std::future<int>> futs;  // 在途推理任务 (主线程访问, 无需锁)
    const char *out_dir;
    int max_save;
    int total;        // 已提交帧数 (解码+提交)
    int done;         // 已完成帧数 (推理完)
    int model_w, model_h;
    int thread_num;
    std::mutex rga_mtx;  // RGA 虚拟地址映射在多线程并发下会间歇性失败, 用锁串行化 RGA 调用

    // FPS 统计: 稳态时钟, 不受系统时间回拨影响
    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point t_last;
    int fps_cnt;
};

// 每完成一帧的统计回调 (主线程调用)
static void on_done(PipelineCtx *ctx)
{
    ctx->done++;
    const int fps_window = 30;
    ctx->fps_cnt++;
    if (ctx->fps_cnt >= fps_window) {
        auto now = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(now - ctx->t_last).count();
        printf("[fps] 最近 %d 帧 %.2f s, %.2f fps\n", ctx->fps_cnt, sec, ctx->fps_cnt / sec);
        ctx->t_last = now;
        ctx->fps_cnt = 0;
    }
}

// 线程池里跑的单帧推理任务: NV12 -> BGR -> RGB640x640 -> NPU -> 后处理 -> 存叠加图
//   nv12_buf : 调用方 malloc 并 memcpy 的独立副本 (MPP 帧缓冲会被复用), 任务内 free
//   eng      : 该 slot 对应的 NPU 实例 (同一时刻只被一个任务使用, 由 futures 队列保证)
static int infer_task(RKNNInfer *eng, unsigned char *nv12_buf, int width, int height,
                      int hor_stride, int ver_stride, int idx, PipelineCtx *ctx)
{
    // 1. NV12 -> BGR888 (原画面尺寸) —— RGA 调用加锁, 避免多线程并发映射虚拟地址失败
    unsigned char *bgr = (unsigned char *)malloc((size_t)width * height * 3);
    if (!bgr) { free(nv12_buf); return -1; }
    {
        std::lock_guard<std::mutex> lk(ctx->rga_mtx);
        if (rga_nv12_to_bgr(nv12_buf, width, height, hor_stride, ver_stride, bgr) != 0) {
            free(bgr); free(nv12_buf); return -2;
        }
    }
    cv::Mat img(height, width, CV_8UC3, bgr);

    // 2. BGR -> RGB888 (缩放到 model_w x model_h), 与单图路径一致 —— RGA 调用加锁
    unsigned char *rgb = (unsigned char *)malloc((size_t)ctx->model_w * ctx->model_h * 3);
    if (!rgb) { free(bgr); free(nv12_buf); return -3; }
    {
        std::lock_guard<std::mutex> lk(ctx->rga_mtx);
        if (rga_preprocess(img, ctx->model_w, ctx->model_h, rgb) != 0) {
            free(rgb); free(bgr); free(nv12_buf); return -4;
        }
    }

    // 3. NPU 推理
    if (eng->infer(rgb) != 0) { free(rgb); free(bgr); free(nv12_buf); return -5; }

    // 4. 后处理: argmax 出标签图 (放大回原画面尺寸, 边缘平滑)
    cv::Mat mask;
    if (eng->postprocess_smooth(mask, height, width) != 0) {
        free(rgb); free(bgr); free(nv12_buf); return -6;
    }

    // 5. 上色 + 叠加 + 存盘 (仅前 max_save 帧存盘)
    if (idx < ctx->max_save) {
        cv::Mat color_full = colorize(mask);
        cv::Mat overlay;
        cv::addWeighted(img, 0.5, color_full, 0.5, 0, overlay);
        char path[512];
        snprintf(path, sizeof(path), "%s/frame_%04d.png", ctx->out_dir, idx);
        if (cv::imwrite(path, overlay))
            printf("[frame#%d] %dx%d -> %s\n", idx, width, height, path);
    }

    free(rgb);
    free(bgr);
    free(nv12_buf);
    return 0;
}

// MPP 每解出一帧 NV12 的回调: 复制 NV12 -> 提交线程池异步推理 (流水线)
static void on_frame(const unsigned char *nv12, int width, int height,
                     int hor_stride, int ver_stride, int64_t pts, void *userdata)
{
    PipelineCtx *ctx = (PipelineCtx *)userdata;
    int N = ctx->thread_num;
    int slot = ctx->total % N;   // 轮询复用 N 个 NPU 实例
    int idx = ctx->total;

    // 流控: 在途任务达 N 个时, 等最老的一个完成
    //   (最老任务恰好用的就是当前要复用的 slot, 等它返回即保证 slot 空闲)
    if ((int)ctx->futs.size() >= N) {
        ctx->futs.front().get();
        ctx->futs.pop();
        on_done(ctx);
    }

    // 复制 NV12 (MPP 帧缓冲会被复用, 异步任务必须持有独立副本)
    size_t nv12_size = (size_t)hor_stride * ver_stride * 3 / 2;
    unsigned char *nv12_copy = (unsigned char *)malloc(nv12_size);
    if (!nv12_copy) { ctx->total++; return; }
    memcpy(nv12_copy, nv12, nv12_size);

    auto fut = ctx->pool->submit(infer_task, ctx->engines[slot], nv12_copy,
                                 width, height, hor_stride, ver_stride, idx, ctx);
    ctx->futs.push(std::move(fut));
    ctx->total++;
}

int main(int argc, char **argv)
{
    const char *rtsp_url = (argc > 1) ? argv[1] : "rtsp://127.0.0.1:8554/cam";
    const char *model    = (argc > 2) ? argv[2] : "model/RK3588/best.rknn";
    const char *out_dir  = (argc > 3) ? argv[3] : "output";
    int max_save         = (argc > 4) ? atoi(argv[4]) : 20;
    int thread_num       = (argc > 5) ? atoi(argv[5]) : DEFAULT_THREAD_NUM;
    if (thread_num < 1) thread_num = 1;

    printf("RTSP: %s\n模型: %s\n输出目录: %s\n最大保存: %d\n线程数: %d\n",
           rtsp_url, model, out_dir, max_save, thread_num);

    if (ensure_dir(out_dir) != 0) {
        fprintf(stderr, "创建输出目录失败\n");
        return -1;
    }

    // 1. 初始化 N 个 NPU 实例, 各按 i%3 绑定到 NPU 核 0/1/2 (3 核并行)
    std::vector<RKNNInfer *> engines;
    for (int i = 0; i < thread_num; i++) {
        RKNNInfer *e = new RKNNInfer();
        if (e->init(model, i % 3) != 0) {
            fprintf(stderr, "RKNN 实例[%d] 初始化失败\n", i);
            for (auto p : engines) { p->deinit(); delete p; }
            return -2;
        }
        engines.push_back(e);
    }
    printf("[npu] %d 个实例, 输入 %dx%dx%d, 输出 %dx%dx%d\n",
           thread_num,
           engines[0]->input_w(), engines[0]->input_h(), engines[0]->input_c(),
           engines[0]->out_c(), engines[0]->out_h(), engines[0]->out_w());

    // 2. 线程池 (大小 = 线程数)
    dpool::ThreadPool pool(thread_num);

    // 3. 初始化 MPP 硬解
    MppDecoder dec;
    if (dec.init() != 0) {
        fprintf(stderr, "MPP 初始化失败\n");
        for (auto p : engines) { p->deinit(); delete p; }
        return -3;
    }

    // 4. 打开 RTSP 流
    RtspPuller puller;
    if (puller.open(rtsp_url) != 0) {
        fprintf(stderr, "RTSP 打开失败\n");
        dec.deinit();
        for (auto p : engines) { p->deinit(); delete p; }
        return -4;
    }

    // 5. 主循环: 拉包 -> 喂 MPP -> on_frame 里复制 NV12 并提交线程池异步推理
    PipelineCtx ctx;
    ctx.engines = engines;
    ctx.pool = &pool;
    ctx.out_dir = out_dir;
    ctx.max_save = max_save;
    ctx.total = 0;
    ctx.done = 0;
    ctx.model_w = engines[0]->input_w();
    ctx.model_h = engines[0]->input_h();
    ctx.thread_num = thread_num;
    ctx.fps_cnt = 0;
    ctx.t_start = std::chrono::steady_clock::now();
    ctx.t_last = ctx.t_start;

    while (true) {
        const unsigned char *data = NULL;
        int size = 0;
        int64_t pts = 0;
        int r = puller.read_packet(&data, &size, &pts);
        if (r == 0) {
            printf("[rtsp] 流结束 (EOS)\n");
            // 送一个空 EOS 包排空 MPP 流水线
            dec.decode_packet(NULL, 0, 0, true, on_frame, &ctx);
            break;
        }
        if (r < 0) {
            fprintf(stderr, "[rtsp] 拉流失败, 退出\n");
            break;
        }
        // r == 1: 拿到一个 Annex B 包, 喂 MPP
        dec.decode_packet(data, size, pts, false, on_frame, &ctx);
    }

    // 6. 排空剩余在途任务 (EOS 后仍有 <= N 个任务未完成)
    while (!ctx.futs.empty()) {
        ctx.futs.front().get();
        ctx.futs.pop();
        on_done(&ctx);
    }

    printf("\n===== 管线汇总 =====\n");
    printf("提交帧数: %d, 完成帧数: %d\n", ctx.total, ctx.done);
    printf("已保存叠加图: %d/%d\n", ctx.total < max_save ? ctx.total : max_save, max_save);
    double total_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - ctx.t_start).count();
    if (total_sec > 0.0 && ctx.done > 0)
        printf("全程平均 FPS: %.2f (总耗时 %.2f s, %d 帧, %d 线程)\n",
               ctx.done / total_sec, total_sec, ctx.done, thread_num);
    printf("管线: RTSP -> FFmpeg -> MPP(NV12) -> [线程池] RGA(BGR->RGB640x640) -> NPU(3核并行) -> 叠加图\n");

    puller.close();
    dec.deinit();
    for (auto p : engines) { p->deinit(); delete p; }
    printf("完成\n");
    return 0;
}
