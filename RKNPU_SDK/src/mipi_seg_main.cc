// MIPI 摄像头 -> V4L2 MMAP -> BGR -> [线程池] RGA RGB缩放 -> NPU 分割 -> 预测叠加图
// 用法: ./mipi_seg [device] [model] [output_dir] [max_save] [thread_num] [width height] [--no-display]
// 示例: ./mipi_seg /dev/video11 model/RK3588/best.rknn mipi_output 20 3
// 默认 3 个线程/独立 NPU 实例，按 i%3 绑核，采集与推理重叠。
// 兼容旧的 max_save width height 参数形式 (不指定线程数时使用 3)。
// 默认沿用节点当前分辨率; 可选 width height 请求采集分辨率，以驱动返回值为准。
// 默认实时显示预测叠加图，Q/Esc/关闭窗口/Ctrl+C 退出。无图形会话时仅推理。
// max_save=0 不存图; 达到保存数量后继续预测和显示。不编码/推流。
// 需先配置好 sensor/CSI/ISP media 链路，节点必须输出已处理的图像，而非 Bayer RAW。

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <dirent.h>
#include <signal.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <exception>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "rga_utils.h"
#include "rknn_infer.h"
#include "ThreadPool.hpp"

static volatile sig_atomic_t g_stop = 0;
static void stop_capture(int) { g_stop = 1; }

static int xioctl(int fd, unsigned long request, void *arg)
{
    int ret;
    do { ret = ioctl(fd, request, arg); } while (ret < 0 && errno == EINTR && !g_stop);
    return ret;
}

static std::string fourcc(unsigned int fmt)
{
    char s[5] = {char(fmt), char(fmt >> 8), char(fmt >> 16), char(fmt >> 24), 0};
    return std::string(s);
}

static bool supported(unsigned int fmt)
{
    switch (fmt) {
    case V4L2_PIX_FMT_NV12: case V4L2_PIX_FMT_NV21:
    case V4L2_PIX_FMT_NV12M: case V4L2_PIX_FMT_NV21M:
    case V4L2_PIX_FMT_YUYV: case V4L2_PIX_FMT_UYVY:
    case V4L2_PIX_FMT_RGB24: case V4L2_PIX_FMT_BGR24:
    case V4L2_PIX_FMT_GREY: return true;
    default: return false;
    }
}

class V4L2Capture {
public:
    V4L2Capture() : fd_(-1), streaming_(false), multi_(false), width_(0), height_(0),
                    pixel_(0), planes_(0), type_(V4L2_BUF_TYPE_VIDEO_CAPTURE) {}
    ~V4L2Capture() { close_device(); }
    V4L2Capture(const V4L2Capture &) = delete;
    V4L2Capture &operator=(const V4L2Capture &) = delete;

    bool open_device(const char *device, int width, int height)
    {
        fd_ = open(device, O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd_ < 0) return fail("打开摄像头 (检查节点和 video 组权限)");
        v4l2_capability cap = {};
        if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) return fail("VIDIOC_QUERYCAP");
        unsigned int caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS)
                            ? cap.device_caps : cap.capabilities;
        multi_ = (caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) != 0;
        if (!(caps & V4L2_CAP_STREAMING) ||
            !(caps & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE))) {
            fprintf(stderr, "[camera] 节点不支持 V4L2 流式视频采集\n");
            return false;
        }
        type_ = multi_ ? V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE : V4L2_BUF_TYPE_VIDEO_CAPTURE;
        v4l2_format fmt = {};
        fmt.type = type_;
        if (xioctl(fd_, VIDIOC_G_FMT, &fmt) < 0) return fail("VIDIOC_G_FMT");
        unsigned int current = multi_ ? fmt.fmt.pix_mp.pixelformat : fmt.fmt.pix.pixelformat;
        if (!supported(current)) {
            // 只请求驱动明确枚举支持的可转换格式，不把 RAW 数据误当成 RGB。
            v4l2_fmtdesc desc = {};
            desc.type = type_;
            bool found = false;
            for (desc.index = 0; xioctl(fd_, VIDIOC_ENUM_FMT, &desc) == 0; ++desc.index) {
                if (supported(desc.pixelformat)) {
                    current = desc.pixelformat;
                    found = true;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "[camera] 当前格式 %s，节点无可用的 NV12/NV21/YUYV/UYVY/RGB/BGR/GREY 格式。\n"
                                "请检查 video11 是否为 ISP 输出节点及 media 链路配置。\n",
                        fourcc(current).c_str());
                return false;
            }
        }
        if (multi_) {
            fmt.fmt.pix_mp.pixelformat = current;
            if (width) { fmt.fmt.pix_mp.width = width; fmt.fmt.pix_mp.height = height; }
        } else {
            fmt.fmt.pix.pixelformat = current;
            if (width) { fmt.fmt.pix.width = width; fmt.fmt.pix.height = height; }
        }
        if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) return fail("VIDIOC_S_FMT");
        width_ = multi_ ? fmt.fmt.pix_mp.width : fmt.fmt.pix.width;
        height_ = multi_ ? fmt.fmt.pix_mp.height : fmt.fmt.pix.height;
        pixel_ = multi_ ? fmt.fmt.pix_mp.pixelformat : fmt.fmt.pix.pixelformat;
        planes_ = multi_ ? fmt.fmt.pix_mp.num_planes : 1;
        bool split = pixel_ == V4L2_PIX_FMT_NV12M || pixel_ == V4L2_PIX_FMT_NV21M;
        if (!supported(pixel_) || width_ <= 0 || height_ <= 0 ||
            planes_ != (split ? 2U : 1U) ||
            ((pixel_ == V4L2_PIX_FMT_NV12 || pixel_ == V4L2_PIX_FMT_NV21 || split) &&
             ((width_ % 2) || (height_ % 2))) ||
            ((pixel_ == V4L2_PIX_FMT_YUYV || pixel_ == V4L2_PIX_FMT_UYVY) && width_ % 2)) {
            fprintf(stderr, "[camera] 不支持驱动返回的格式/尺寸/平面数: %s %dx%d planes=%u\n",
                    fourcc(pixel_).c_str(), width_, height_, planes_);
            return false;
        }
        stride_.resize(planes_);
        unsigned int bytes = (pixel_ == V4L2_PIX_FMT_RGB24 || pixel_ == V4L2_PIX_FMT_BGR24)
                             ? 3 : ((pixel_ == V4L2_PIX_FMT_YUYV || pixel_ == V4L2_PIX_FMT_UYVY) ? 2 : 1);
        for (unsigned int p = 0; p < planes_; ++p) {
            stride_[p] = multi_ ? fmt.fmt.pix_mp.plane_fmt[p].bytesperline : fmt.fmt.pix.bytesperline;
            if (!stride_[p]) stride_[p] = width_ * bytes;
            if (stride_[p] < static_cast<size_t>(width_) * bytes) {
                fprintf(stderr, "[camera] bytesperline 小于有效行长度\n");
                return false;
            }
        }
        printf("[camera] %s (%s), %dx%d %s, %s, planes=%u, stride=%zu\n",
               device, cap.driver, width_, height_, fourcc(pixel_).c_str(),
               multi_ ? "MPLANE" : "单平面", planes_, stride_[0]);

        v4l2_requestbuffers req = {};
        req.count = 4;
        req.type = type_;
        req.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) return fail("VIDIOC_REQBUFS");
        if (req.count < 2) { fprintf(stderr, "[camera] 可用采集缓冲不足\n"); return false; }
        buffers_.resize(req.count);
        for (unsigned int i = 0; i < req.count; ++i) {
            v4l2_buffer buf = {};
            v4l2_plane plane[VIDEO_MAX_PLANES] = {};
            prepare(buf, plane, i);
            if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) return fail("VIDIOC_QUERYBUF");
            if (multi_ && buf.length != planes_) {
                fprintf(stderr, "[camera] QUERYBUF 平面数不匹配\n"); return false;
            }
            buffers_[i].resize(planes_);
            for (unsigned int p = 0; p < planes_; ++p) {
                Mapping &m = buffers_[i][p];
                m.length = multi_ ? plane[p].length : buf.length;
                off_t offset = multi_ ? plane[p].m.mem_offset : buf.m.offset;
                m.addr = mmap(NULL, m.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, offset);
                if (m.addr == MAP_FAILED) return fail("mmap");
            }
            if (!queue(i)) return false;
        }
        if (xioctl(fd_, VIDIOC_STREAMON, &type_) < 0)
            return fail("VIDIOC_STREAMON (检查 sensor/CSI/ISP 链路，或摄像头是否被占用)");
        streaming_ = true;
        last_frame_ = std::chrono::steady_clock::now();
        return true;
    }

    // 返回 1=独立 BGR 帧，0=重试/中断，-1=致命错误。
    int read_bgr(cv::Mat &bgr)
    {
        pollfd pfd = {fd_, POLLIN, 0};
        // 短轮询使主线程可以持续处理窗口事件，累计 2 秒无帧才报错。
        int ret = poll(&pfd, 1, 20);
        if (ret < 0) { if (errno == EINTR) return 0; fail("poll"); return -1; }
        if (!ret) {
            if (std::chrono::steady_clock::now() - last_frame_ < std::chrono::seconds(2)) return 0;
            fprintf(stderr, "[camera] 2 秒未收到帧，检查 MIPI 链路及摄像头启动状态\n");
            return -1;
        }
        if (g_stop) return 0;
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            fprintf(stderr, "[camera] 采集设备异常，poll revents=0x%x\n", pfd.revents);
            return -1;
        }
        v4l2_buffer buf = {};
        v4l2_plane plane[VIDEO_MAX_PLANES] = {};
        prepare(buf, plane, 0);
        if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN || (errno == EINTR && g_stop)) return 0;
            fail("VIDIOC_DQBUF"); return -1;
        }
        if (buf.index >= buffers_.size() || (multi_ && buf.length != planes_)) {
            fprintf(stderr, "[camera] 驱动返回无效缓冲索引或平面数\n"); return -1;
        }
        last_frame_ = std::chrono::steady_clock::now();
        if (buf.flags & V4L2_BUF_FLAG_ERROR) {
            fprintf(stderr, "[camera] 丢弃驱动标记为错误的帧\n");
            return queue(buf.index) ? 0 : -1;
        }
        const unsigned char *data[2] = {};
        size_t available[2] = {};
        for (unsigned int p = 0; p < planes_; ++p) {
            const Mapping &m = buffers_[buf.index][p];
            size_t offset = multi_ ? plane[p].data_offset : 0;
            size_t used = multi_ ? plane[p].bytesused : buf.bytesused;
            if (used > m.length || offset > used) {
                fprintf(stderr, "[camera] 无效 bytesused/data_offset\n"); return -1;
            }
            data[p] = static_cast<const unsigned char *>(m.addr) + offset;
            available[p] = used - offset;
        }
        bool ok = convert(data, available, bgr);
        // convert 必须先拷贝/转换完成：QBUF 后驱动会复用该缓冲。
        if (!queue(buf.index)) return -1;
        return ok ? 1 : -1;
    }

private:
    struct Mapping {
        void *addr;
        size_t length;
        Mapping() : addr(MAP_FAILED), length(0) {}
    };
    int fd_;
    bool streaming_, multi_;
    int width_, height_;
    unsigned int pixel_, planes_;
    v4l2_buf_type type_;
    std::vector<size_t> stride_;
    std::vector<std::vector<Mapping>> buffers_;
    std::chrono::steady_clock::time_point last_frame_;

    static bool fail(const char *what) { fprintf(stderr, "[camera] %s: %s\n", what, strerror(errno)); return false; }
    void prepare(v4l2_buffer &buf, v4l2_plane *plane, unsigned int index)
    {
        buf.type = type_; buf.memory = V4L2_MEMORY_MMAP; buf.index = index;
        if (multi_) { buf.m.planes = plane; buf.length = planes_; }
    }
    bool queue(unsigned int index)
    {
        v4l2_buffer buf = {};
        v4l2_plane plane[VIDEO_MAX_PLANES] = {};
        prepare(buf, plane, index);
        if (multi_) for (unsigned int p = 0; p < planes_; ++p) plane[p].length = buffers_[index][p].length;
        return xioctl(fd_, VIDIOC_QBUF, &buf) == 0 || fail("VIDIOC_QBUF");
    }
    bool convert(const unsigned char **data, const size_t *available, cv::Mat &bgr)
    {
        bool nv = pixel_ == V4L2_PIX_FMT_NV12 || pixel_ == V4L2_PIX_FMT_NV21 ||
                  pixel_ == V4L2_PIX_FMT_NV12M || pixel_ == V4L2_PIX_FMT_NV21M;
        if (nv) {
            size_t ysize = stride_[0] * height_;
            size_t uvstride = planes_ == 2 ? stride_[1] : stride_[0];
            size_t uvsize = uvstride * (height_ / 2);
            if (available[0] < ysize || (planes_ == 1 ? available[0] - ysize < uvsize : available[1] < uvsize))
                return short_frame();
            const unsigned char *uv = planes_ == 2 ? data[1] : data[0] + ysize;
            cv::Mat yuv(height_ + height_ / 2, width_, CV_8UC1);
            for (int y = 0; y < height_; ++y) memcpy(yuv.ptr(y), data[0] + y * stride_[0], width_);
            for (int y = 0; y < height_ / 2; ++y) memcpy(yuv.ptr(height_ + y), uv + y * uvstride, width_);
            cv::cvtColor(yuv, bgr, (pixel_ == V4L2_PIX_FMT_NV21 || pixel_ == V4L2_PIX_FMT_NV21M)
                                  ? cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12);
        } else {
            int channels = pixel_ == V4L2_PIX_FMT_GREY ? 1 :
                           ((pixel_ == V4L2_PIX_FMT_YUYV || pixel_ == V4L2_PIX_FMT_UYVY) ? 2 : 3);
            size_t required = stride_[0] * (height_ - 1) + static_cast<size_t>(width_) * channels;
            if (available[0] < required) return short_frame();
            cv::Mat src(height_, width_, CV_MAKETYPE(CV_8U, channels), const_cast<unsigned char *>(data[0]), stride_[0]);
            if (pixel_ == V4L2_PIX_FMT_BGR24) bgr = src.clone();
            else cv::cvtColor(src, bgr, pixel_ == V4L2_PIX_FMT_RGB24 ? cv::COLOR_RGB2BGR :
                             (pixel_ == V4L2_PIX_FMT_GREY ? cv::COLOR_GRAY2BGR :
                              (pixel_ == V4L2_PIX_FMT_YUYV ? cv::COLOR_YUV2BGR_YUY2 : cv::COLOR_YUV2BGR_UYVY)));
        }
        return true;
    }
    static bool short_frame() { fprintf(stderr, "[camera] 帧数据不足，检查格式/stride/bytesused\n"); return false; }
    void close_device()
    {
        if (streaming_) xioctl(fd_, VIDIOC_STREAMOFF, &type_);
        for (auto &buffer : buffers_) for (auto &m : buffer)
            if (m.addr != MAP_FAILED) munmap(m.addr, m.length);
        if (fd_ >= 0) ::close(fd_);
    }
};

static bool number(const char *s, int &value)
{
    char *end = NULL;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || end == s || *end || n < 0 || n > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(n);
    return true;
}

static bool ensure_dir(const char *path)
{
    if (mkdir(path, 0755) == 0) return true;
    if (errno == EEXIST) {
        DIR *d = opendir(path);
        if (d) { closedir(d); return true; }
    }
    fprintf(stderr, "[output] 创建目录 %s 失败: %s\n", path, strerror(errno));
    return false;
}

static cv::Mat colorize(const cv::Mat &mask)
{
    // 与 seg_single/rtsp_seg 的 10 类 BGR 颜色一致。
    static const cv::Vec3b colors[10] = {
        {0,0,0}, {60,20,220}, {255,144,30}, {50,205,50}, {0,165,255},
        {211,0,148}, {209,206,0}, {0,215,255}, {180,105,255}, {128,128,128}
    };
    cv::Mat color(mask.size(), CV_8UC3);
    for (int y = 0; y < mask.rows; ++y) {
        const unsigned char *src = mask.ptr<unsigned char>(y);
        cv::Vec3b *dst = color.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.cols; ++x) dst[x] = colors[src[x] % 10];
    }
    return color;
}

// 所有 HighGUI 调用只在主线程执行；窗口失败时回退为无窗口推理。
class PreviewWindow {
public:
    explicit PreviewWindow(bool requested)
        : active_(false), visibility_warned_(false), autosize_supported_(false) {
        if (!requested) return;
        const char *display = getenv("DISPLAY"), *wayland = getenv("WAYLAND_DISPLAY");
        if ((!display || !*display) && (!wayland || !*wayland)) {
            fprintf(stderr, "[preview] 未检测到图形会话，关闭预览。请在板子桌面的终端中运行以显示画面。\n");
            return;
        }
        try {
            cv::namedWindow(name(), cv::WINDOW_NORMAL);
            active_ = true;
            cv::resizeWindow(name(), 960, 540);
            // 立即显示占位画面，使加载/首帧等待期间窗口也能响应退出操作。
            cv::Mat waiting(540, 960, CV_8UC3, cv::Scalar::all(0));
            cv::putText(waiting, "Waiting for NPU prediction... (Q / Esc to quit)",
                        cv::Point(40, 270), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::imshow(name(), waiting);
            cv::waitKey(1);
            printf("[preview] 实时预测叠加预览已开启，按 Q/Esc 或关闭窗口退出\n");
        } catch (const cv::Exception &e) {
            disable(e.what());
        }
    }
    ~PreviewWindow() { close_window(); }
    PreviewWindow(const PreviewWindow &) = delete;
    PreviewWindow &operator=(const PreviewWindow &) = delete;
    bool enabled() const { return active_; }
    void show(const cv::Mat &overlay) {
        if (!active_ || overlay.empty()) return;
        try { cv::imshow(name(), overlay); }
        catch (const cv::Exception &e) { disable(e.what()); }
    }
    bool pump() {
        if (!active_) return true;
        try {
            int key = cv::waitKey(1);
            if (key >= 0) key &= 0xff;
            if (key == 'q' || key == 'Q' || key == 27) return false;
            double visible = cv::getWindowProperty(name(), cv::WND_PROP_VISIBLE);
            if (visible >= 0) return visible != 0;
            // OpenCV 4.7 GTK 后端不支持 VISIBLE，-1 不是“窗口已关闭”。
            // AUTOSIZE 对已存在的 WINDOW_NORMAL 窗口返回 0，关闭后返回 -1。
            if (!visibility_warned_) {
                fprintf(stderr, "[preview] 窗口后端不支持 VISIBLE 查询，使用 AUTOSIZE 存在性检查\n");
                visibility_warned_ = true;
            }
            double autosize = cv::getWindowProperty(name(), cv::WND_PROP_AUTOSIZE);
            if (autosize >= 0) { autosize_supported_ = true; return true; }
            // 两种查询都不支持时只通过 Q/Esc/Ctrl+C 退出，不把 -1 误判为关窗。
            return !autosize_supported_;
        } catch (const cv::Exception &e) {
            fprintf(stderr, "[preview] 窗口已关闭或发生异常，停止推理: %s\n", e.what());
            close_window();
            return false; // 关闭窗口后，部分后端查询窗口属性会抛异常，也应退出。
        }
    }
private:
    bool active_, visibility_warned_, autosize_supported_;
    static const char *name() { return "MIPI NPU Segmentation"; }
    void close_window() {
        if (!active_) return;
        active_ = false;
        try { cv::destroyWindow(name()); }
        catch (const cv::Exception &) {}
    }
    void disable(const char *reason) {
        fprintf(stderr, "[preview] 窗口不可用，继续无窗口推理: %s\n", reason);
        close_window();
    }
};

struct PipelineCtx {
    const char *output;
    int max_save;
    bool preview;
    std::mutex rga_mutex; // 与 RTSP 程序一样，避免多线程并发 RGA 虚拟地址映射失败。
};

struct FrameResult {
    int status;
    bool saved;
    cv::Mat overlay;
    FrameResult(int s, bool written, cv::Mat image = cv::Mat())
        : status(s), saved(written), overlay(std::move(image)) {}
};

// bgr 按值传递，持有采集时已复制/转换的独立图像，不引用 V4L2 的复用缓冲。
// 同一实例只处理一个在途任务，主线程在复用 slot 前等待其 future 完成。
static FrameResult infer_frame(RKNNInfer *engine, cv::Mat bgr,
                               unsigned long long index, PipelineCtx *ctx)
{
    std::vector<unsigned char> rgb(static_cast<size_t>(engine->input_w()) * engine->input_h() * 3);
    {
        std::lock_guard<std::mutex> lock(ctx->rga_mutex);
        if (rga_preprocess(bgr, engine->input_w(), engine->input_h(), rgb.data()) != 0) {
            fprintf(stderr, "[frame#%llu] RGA 预处理失败\n", index);
            return {-1, false};
        }
    }
    if (engine->infer(rgb.data()) != 0) {
        fprintf(stderr, "[frame#%llu] NPU 推理失败\n", index);
        return {-2, false};
    }
    cv::Mat mask;
    if (engine->postprocess_smooth(mask, bgr.rows, bgr.cols) != 0) {
        fprintf(stderr, "[frame#%llu] 分割后处理失败\n", index);
        return {-3, false};
    }
    bool save = index < static_cast<unsigned long long>(ctx->max_save);
    cv::Mat overlay;
    if (ctx->preview || save)
        cv::addWeighted(bgr, 0.5, colorize(mask), 0.5, 0, overlay);
    if (save) {
        char name[64];
        snprintf(name, sizeof(name), "/frame_%04llu.png", index);
        std::string path = std::string(ctx->output) + name;
        if (!cv::imwrite(path, overlay)) {
            fprintf(stderr, "[output] 保存 %s 失败\n", path.c_str());
            return {-4, false};
        }
        printf("[frame#%llu] %dx%d -> %s\n", index, bgr.cols, bgr.rows, path.c_str());
    }
    return {0, save, std::move(overlay)};
}

int main(int argc, char **argv)
{
    // 新增开关放在参数末尾，不改变原有的位置参数形式。
    bool display = true;
    if (argc > 1 && !strcmp(argv[argc - 1], "--no-display")) { display = false; --argc; }
    if (argc > 8 || (argc > 1 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")))) {
        printf("用法: %s [device] [model] [output_dir] [max_save] [thread_num] [width height] [--no-display]\n"
               "示例: %s /dev/video11 model/RK3588/best.rknn mipi_output 20 3\n"
               "thread_num=1..6 (默认3，推荐3或6); max_save=0 不存图。\n"
               "默认实时显示预测叠加图，Q/Esc/关闭窗口/Ctrl+C 退出。\n"
               "默认沿用采集分辨率，--no-display 关闭预览，无图形会话时自动关闭预览。\n"
               "兼容旧的 max_save width height 参数形式，使用默认3线程。\n", argv[0], argv[0]);
        return argc > 8 ? 1 : 0;
    }
    const char *device = argc > 1 ? argv[1] : "/dev/video11";
    const char *model = argc > 2 ? argv[2] : "model/RK3588/best.rknn";
    const char *output = argc > 3 ? argv[3] : "mipi_output";
    int max_save = 20, thread_num = 3, width = 0, height = 0;
    // argc==7 是旧的 width height 形式；argc==8 是 thread_num width height。
    int size_arg = argc == 7 ? 5 : (argc == 8 ? 6 : 0);
    if ((argc > 4 && !number(argv[4], max_save)) ||
        ((argc == 6 || argc == 8) && (!number(argv[5], thread_num) || thread_num < 1 || thread_num > 6)) ||
        (size_arg && (!number(argv[size_arg], width) || !number(argv[size_arg + 1], height) || !width || !height))) {
        fprintf(stderr, "max_save 必须是非负整数，thread_num 必须是1..6，width/height 必须是正整数\n"); return 1;
    }
    struct sigaction sa = {};
    sa.sa_handler = stop_capture;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    printf("摄像头: %s\n模型: %s\n输出目录: %s\n最大保存: %d (之后继续推理)\n线程数: %d\n",
           device, model, output, max_save, thread_num);
    try {
        if (max_save && !ensure_dir(output)) return 1;
        // 帧级并行由线程池负责，避免每个任务再启动一组 OpenCV CPU 线程。
        cv::setNumThreads(1);
        std::vector<std::unique_ptr<RKNNInfer>> engines;
        for (int i = 0; i < thread_num; ++i) {
            std::unique_ptr<RKNNInfer> engine(new RKNNInfer());
            if (engine->init(model, i % 3) != 0) {
                fprintf(stderr, "NPU 实例[%d] 初始化失败\n", i); return 1;
            }
            if (engine->input_c() != 3 || engine->out_c() != 10) {
                fprintf(stderr, "程序要求 RGB 3 通道输入、10 类分割输出\n"); return 1;
            }
            engines.push_back(std::move(engine));
        }
        printf("[npu] %d 个实例，按 i%%3 绑定核心 0/1/2，输入 %dx%dx%d\n", thread_num,
               engines[0]->input_w(), engines[0]->input_h(), engines[0]->input_c());
        V4L2Capture camera;
        if (!camera.open_device(device, width, height)) return 1;
        PreviewWindow preview(display);
        // 声明顺序保证异常退出时 pool 先排空/join，之后才销毁 ctx、摄像头和 NPU 实例。
        PipelineCtx ctx;
        ctx.output = output;
        ctx.max_save = max_save;
        ctx.preview = preview.enabled(); // 提交任务后保持只读，避免工作线程与窗口状态发生竞争。
        dpool::ThreadPool pool(thread_num);
        std::deque<std::future<FrameResult>> pending;
        auto start = std::chrono::steady_clock::now(), last = start;
        unsigned long long submitted = 0, completed = 0, succeeded = 0;
        int saved = 0, status = 0;
        // 主线程收集结果并统计，工作线程不修改共享计数。
        auto collect = [&]() {
            auto future = std::move(pending.front());
            pending.pop_front();
            // 不能直接阻塞 get()：等待 NPU 时也要处理窗口事件。
            while (future.wait_for(std::chrono::milliseconds(10)) != std::future_status::ready) {
                if (!g_stop && !preview.pump()) g_stop = 1;
            }
            try {
                FrameResult result = future.get();
                if (result.status != 0) status = 1;
                else ++succeeded;
                if (result.saved) ++saved;
                if (!g_stop && !status) preview.show(result.overlay);
            } catch (const std::exception &e) {
                fprintf(stderr, "[infer] 工作线程异常: %s\n", e.what());
                status = 1;
            } catch (...) {
                fprintf(stderr, "[infer] 工作线程发生未知异常\n");
                status = 1;
            }
            ++completed;
            if (completed % 30 == 0) {
                auto now = std::chrono::steady_clock::now();
                double sec = std::chrono::duration<double>(now - last).count();
                printf("[fps] 最近 30 个任务 %.2f fps，成功 %llu/%llu (%d 线程)\n",
                       sec > 0 ? 30.0 / sec : 0, succeeded, completed, thread_num);
                last = now;
            }
        };
        while (!g_stop) {
            if (!preview.pump()) { g_stop = 1; break; }
            // 最多 N 个在途任务：等待最旧任务后，submitted%N 对应的实例必然空闲。
            // 有界队列避免慢推理造成图像积压和内存持续增长。
            if (pending.size() >= static_cast<size_t>(thread_num)) {
                collect();
                if (status || g_stop) break;
            }
            // 采集等待期间也及时收集已结束任务，以便发现推理/保存错误。
            while (!pending.empty() &&
                   pending.front().wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                collect();
                if (status) break;
            }
            if (status || g_stop) break;
            cv::Mat bgr;
            int ret = camera.read_bgr(bgr);
            if (ret < 0) { status = 1; break; }
            if (!ret || g_stop) continue;
            size_t slot = submitted % static_cast<size_t>(thread_num);
            pending.push_back(pool.submit(infer_frame, engines[slot].get(), std::move(bgr), submitted, &ctx));
            ++submitted;
        }
        // Ctrl+C、采集错误或任务失败时，先等待在途任务完成，再释放其依赖的资源。
        while (!pending.empty()) collect();
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        printf("\n提交: %llu 帧，完成: %llu，预测成功: %llu，保存叠加图: %d，平均 %.2f fps (%d 线程)\n",
               submitted, completed, succeeded, saved, sec > 0 ? succeeded / sec : 0, thread_num);
        return status;
    } catch (const std::exception &e) {
        fprintf(stderr, "[error] %s\n", e.what());
        return 1;
    }
}
