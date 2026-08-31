// 批量评估分割模型: 遍历验证集 -> 推理 -> 与 GT 算 mIoU/每类IoU/混淆矩阵
// 用法: ./eval_miou <model.rknn> <list.txt>
//   list.txt 每行: <image_path> <gt_path>  (空白分隔, GT 为单通道索引 PNG, 像素值 0-9, 255=ignore)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "rga_utils.h"
#include "rknn_infer.h"

#define NUM_CLASSES 10
#define IGNORE_LABEL 255

static const char *kClassNames[NUM_CLASSES] = {
    "_background_", "BL_Device", "CC_Server", "DP_Server", "KDVideo_Device",
    "KVM_Switcher", "SP_Cloud", "VPN_Gateway", "WEB_Firewall", "YP_Server"};

struct EvalStats
{
    long confusion[NUM_CLASSES][NUM_CLASSES]; // confusion[gt][pred]
    EvalStats() { memset(confusion, 0, sizeof(confusion)); }
};

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <model.rknn> <list.txt>\n", argv[0]);
        printf("  list.txt 每行: <image_path> <gt_path>\n");
        return -1;
    }
    const char *model_path = argv[1];
    const char *list_path = argv[2];

    // 1. 初始化 RKNN
    RKNNInfer engine;
    if (engine.init(model_path) != 0)
    {
        fprintf(stderr, "RKNN 初始化失败\n");
        return -1;
    }
    int W = engine.input_w(), H = engine.input_h(), C = engine.input_c();
    unsigned char *rgb_buf = (unsigned char *)malloc((size_t)W * H * C);
    if (!rgb_buf) { fprintf(stderr, "malloc 失败\n"); engine.deinit(); return -2; }

    // 2. 遍历 list
    std::ifstream fin(list_path);
    if (!fin.is_open()) { fprintf(stderr, "打开列表失败: %s\n", list_path); free(rgb_buf); engine.deinit(); return -3; }

    EvalStats stats;
    int n_ok = 0, n_fail = 0;
    std::string line;
    while (std::getline(fin, line))
    {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string img_path, gt_path;
        if (!(iss >> img_path >> gt_path)) { fprintf(stderr, "[skip] 解析行失败: %s\n", line.c_str()); n_fail++; continue; }

        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty()) { fprintf(stderr, "[skip] 读图失败: %s\n", img_path.c_str()); n_fail++; continue; }

        if (rga_preprocess(img, W, H, rgb_buf) != 0) { fprintf(stderr, "[skip] RGA 失败: %s\n", img_path.c_str()); n_fail++; continue; }
        if (engine.infer(rgb_buf) != 0) { fprintf(stderr, "[skip] 推理失败: %s\n", img_path.c_str()); n_fail++; continue; }

        // 对齐 PyTorch testval 评估: 在模型输入分辨率 (W×H=640×640) 比对
        // - 预测: postprocess_smooth 把 soft logits CUBIC 上采样到 W×H 再 argmax (对齐 PyTorch bilinear)
        // - GT: NEAREST 下采样到 W×H (对齐 PyTorch cv2.resize label INTER_NEAREST)
        cv::Mat pred_full;
        if (engine.postprocess_smooth(pred_full, H, W) != 0) {
            fprintf(stderr, "[skip] 后处理失败: %s\n", img_path.c_str()); n_fail++; continue;
        }

        cv::Mat gt = cv::imread(gt_path, cv::IMREAD_GRAYSCALE);
        if (gt.empty()) { fprintf(stderr, "[skip] 读GT失败: %s\n", gt_path.c_str()); n_fail++; continue; }

        // GT 缩到模型输入尺寸 (NEAREST 保持标签索引, 与 PyTorch 一致)
        cv::Mat gt_aligned;
        if (gt.cols != W || gt.rows != H)
            cv::resize(gt, gt_aligned, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
        else
            gt_aligned = gt;

        // 累加混淆矩阵, 跳过 ignore 标签
        for (int h = 0; h < gt_aligned.rows; h++)
        {
            const uchar *g = gt_aligned.ptr<uchar>(h);
            const uchar *p = pred_full.ptr<uchar>(h);
            for (int w = 0; w < gt_aligned.cols; w++)
            {
                int gi = g[w];
                if (gi == IGNORE_LABEL) continue;
                int pi = p[w];
                if (gi >= 0 && gi < NUM_CLASSES && pi >= 0 && pi < NUM_CLASSES)
                    stats.confusion[gi][pi]++;
            }
        }
        n_ok++;
        if (n_ok % 10 == 0) printf("已处理 %d 张\n", n_ok);
    }

    free(rgb_buf);
    engine.deinit();
    fin.close();

    // 3. 计算每类 IoU
    long row_sum[NUM_CLASSES] = {0}, col_sum[NUM_CLASSES] = {0};
    for (int i = 0; i < NUM_CLASSES; i++)
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            row_sum[i] += stats.confusion[i][j];
            col_sum[j] += stats.confusion[i][j];
        }

    printf("\n========== 评估结果 ==========\n");
    printf("成功: %d 张, 失败: %d 张\n\n", n_ok, n_fail);

    // 混淆矩阵
    printf("混淆矩阵 [gt][pred] (行=GT, 列=预测):\n     ");
    for (int j = 0; j < NUM_CLASSES; j++) printf("%8d", j);
    printf("\n");
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        printf("%4d ", i);
        for (int j = 0; j < NUM_CLASSES; j++) printf("%8ld", stats.confusion[i][j]);
        printf("\n");
    }

    // 每类 IoU
    printf("\n各类 IoU:\n");
    double miou_all = 0, miou_nobg = 0;
    int cnt_all = 0, cnt_nobg = 0;
    long total_px = 0, correct_px = 0;
    for (int c = 0; c < NUM_CLASSES; c++)
    {
        long tp = stats.confusion[c][c];
        long denom = row_sum[c] + col_sum[c] - tp;
        total_px += row_sum[c];
        correct_px += tp;
        if (denom == 0)
        {
            printf("  [%d] %-16s: 无样本\n", c, kClassNames[c]);
            continue;
        }
        double iou = (double)tp / denom;
        printf("  [%d] %-16s: IoU=%.4f  (TP=%ld FP=%ld FN=%ld)\n",
               c, kClassNames[c], iou, tp, col_sum[c] - tp, row_sum[c] - tp);
        miou_all += iou; cnt_all++;
        if (c != 0) { miou_nobg += iou; cnt_nobg++; }
    }

    printf("\n---------- 汇总 ----------\n");
    if (cnt_all > 0) printf("mIoU (含背景, %d 类) = %.4f\n", cnt_all, miou_all / cnt_all);
    if (cnt_nobg > 0) printf("mIoU (不含背景, %d 类) = %.4f\n", cnt_nobg, miou_nobg / cnt_nobg);
    if (total_px > 0) printf("Pixel Accuracy = %.4f\n", (double)correct_px / total_px);
    printf("总有效像素 = %ld\n", total_px);

    return 0;
}
