// 红外语义分割 RKNN 推理业务总线 (单图)
// 流程: 读图 -> RGA硬件前处理 -> NPU推理 -> argmax后处理 -> 保存mask与叠加图

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <errno.h>
#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "rga_utils.h"
#include "rknn_infer.h"

/*-------------------------------------------
                Config
-------------------------------------------*/
// THREAD_NUM: 单图推理模式固定为 1 (单张图片单次推理, 无需线程池)
//   如需批量多图并发, 可改为 >1 并引入线程池, 可调范围 [1, 6]
//   RK3588 NPU 有 3 个核心, 多线程时推荐取 3 的倍数 (3 或 6)
#define THREAD_NUM 1

// 输出目录 (相对当前工作目录)
#define OUTPUT_DIR "output"

// 10 类颜色 [R,G,B] (与训练 class_name.txt 一致), 这里转 OpenCV BGR
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

// 从路径提取不带扩展名的文件名, 如 "pic/1_26.png" -> "1_26"
// 同一张图重跑生成同名文件 (覆盖), 不同图 basename 不同自然不冲突
static std::string get_basename(const char *path)
{
    std::string p(path);
    size_t slash = p.find_last_of("/\\");
    if (slash != std::string::npos) p = p.substr(slash + 1);
    size_t dot = p.find_last_of('.');
    if (dot != std::string::npos) p = p.substr(0, dot);
    return p;
}

// 确保 output 目录存在 (不存在则创建, 权限 0755)
// 注意: 不使用 stat() —— 新版 glibc 把 stat 重定向到需要 GLIBC_2.33 的 64 位变体,
//       板子旧 glibc 无法解析。改用 mkdir + opendir (均为老符号), 兼容性好
static int ensure_output_dir(const char *dir)
{
    if (mkdir(dir, 0755) == 0) return 0;  // 新建成功
    if (errno == EEXIST)
    {
        // 已存在, 用 opendir 确认是目录而非文件
        DIR *d = opendir(dir);
        if (d) { closedir(d); return 0; }
        fprintf(stderr, "[output] %s 已存在但不是目录\n", dir);
        return -1;
    }
    perror("[output] mkdir 失败");
    return -2;
}

// 离散标签图逐像素查表上色, 保证只输出 kSegColors 里的 10 种纯色
static cv::Mat colorize(const cv::Mat &label_full)
{
    cv::Mat out(label_full.size(), CV_8UC3);
    for (int h = 0; h < label_full.rows; h++)
    {
        const uchar *lr = label_full.ptr<uchar>(h);
        cv::Vec3b *cr = out.ptr<cv::Vec3b>(h);
        for (int w = 0; w < label_full.cols; w++)
            cr[w] = kSegColors[lr[w] % 10];
    }
    return out;
}

int main(int argc, char **argv)
{
    const char *model_path = (argc > 1) ? argv[1] : "./model/RK3588/best.rknn";
    const char *image_path = (argc > 2) ? argv[2] : "test.jpg";
    printf("模型: %s\n图像: %s\n", model_path, image_path);

    // 0. 准备输出目录
    if (ensure_output_dir(OUTPUT_DIR) != 0) {
        fprintf(stderr, "创建输出目录失败: %s\n", OUTPUT_DIR);
        return -10;
    }
    std::string base = get_basename(image_path);

    // 1. 初始化 RKNN 引擎
    RKNNInfer engine;
    if (engine.init(model_path) != 0) {
        fprintf(stderr, "RKNN 初始化失败\n");
        return -1;
    }

    // 2. 读取本地图片 (BGR)
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "读取图片失败: %s\n", image_path);
        engine.deinit();
        return -2;
    }
    printf("原图尺寸: %dx%d\n", img.cols, img.rows);

    // 3. RGA 硬件前处理: 缩放到 640x640 + BGR2RGB
    int W = engine.input_w(), H = engine.input_h(), C = engine.input_c();
    size_t buf_size = (size_t)W * H * C;
    unsigned char *rgb_buf = (unsigned char *)malloc(buf_size);
    if (!rgb_buf) {
        fprintf(stderr, "分配输入 buffer 失败\n");
        engine.deinit();
        return -3;
    }
    if (rga_preprocess(img, W, H, rgb_buf) != 0) {
        fprintf(stderr, "RGA 前处理失败\n");
        free(rgb_buf);
        engine.deinit();
        return -4;
    }

    // 4. NPU 推理
    if (engine.infer(rgb_buf) != 0) {
        fprintf(stderr, "NPU 推理失败\n");
        free(rgb_buf);
        engine.deinit();
        return -5;
    }

    // 5+6. 后处理(平滑): 直接对模型原始软输出 CUBIC 放大到原图尺寸再 argmax
    //    比先 argmax 再 one-hot 上采样质量更高 (保留模型亚像素置信度, 边缘更平滑)
    cv::Mat mask_full;
    if (engine.postprocess_smooth(mask_full, img.rows, img.cols) != 0) {
        fprintf(stderr, "后处理失败\n");
        free(rgb_buf);
        engine.deinit();
        return -6;
    }
    // 7. 生成伪彩色 mask (逐像素查表上色, 10 种纯色, 与训练 class_name.txt 一致)
    //    mask_full 是 CV_8UC1 标签图 (背景=0, 直接保存会黑乎乎), 上色后才直观可读
    cv::Mat color_full = colorize(mask_full);
    std::string mask_path = std::string(OUTPUT_DIR) + "/" + base + "_mask.png";
    if (!cv::imwrite(mask_path, color_full)) {
        fprintf(stderr, "保存 %s 失败\n", mask_path.c_str());
    } else {
        printf("已保存 %s (%dx%d)\n", mask_path.c_str(), color_full.cols, color_full.rows);
    }

    // 8. 生成原图 + 伪彩色 mask 叠加图 (10 种纯色, 无混色)
    cv::Mat overlay;
    cv::addWeighted(img, 0.5, color_full, 0.5, 0, overlay);
    std::string overlay_path = std::string(OUTPUT_DIR) + "/" + base + "_overlay.png";
    if (!cv::imwrite(overlay_path, overlay)) {
        fprintf(stderr, "保存 %s 失败\n", overlay_path.c_str());
    } else {
        printf("已保存 %s (%dx%d)\n", overlay_path.c_str(), overlay.cols, overlay.rows);
    }

    // 9. 释放资源
    free(rgb_buf);
    engine.deinit();
    printf("完成\n");
    return 0;
}
