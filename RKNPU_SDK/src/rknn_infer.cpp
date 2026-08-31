/* RKNN 推理 + argmax 后处理 */
#include "rknn_infer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <opencv2/imgproc.hpp>

// __fp16 是 ARM aarch64 的半精度浮点类型, 用于直接解析 FLOAT16 输出 buffer
typedef __fp16 float16_t;

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "[rknn] 打开模型失败: %s\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char *data = (unsigned char *)malloc(size);
    if (!data) {
        fclose(fp);
        return NULL;
    }
    if ((int)fread(data, 1, size, fp) != size) {
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *model_size = size;
    return data;
}

RKNNInfer::RKNNInfer()
    : ctx_(0), initialized_(false),
      in_w_(0), in_h_(0), in_c_(0),
      out_c_(0), out_h_(0), out_w_(0),
      input_attrs_(NULL), output_attrs_(NULL),
      n_input_(0), n_output_(0)
{
    memset(inputs_, 0, sizeof(inputs_));
}

RKNNInfer::~RKNNInfer() { deinit(); }

int RKNNInfer::init(const char *model_path, int core_id)
{
    if (initialized_) {
        fprintf(stderr, "[rknn] 已初始化, 请先 deinit\n");
        return -1;
    }

    int model_size = 0;
    unsigned char *model_data = load_model(model_path, &model_size);
    if (!model_data) return -2;

    int ret = rknn_init(&ctx_, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        fprintf(stderr, "[rknn] rknn_init 失败 ret=%d\n", ret);
        return -3;
    }

    // 绑定 NPU 核心: 多线程时各实例绑不同核 (i%3) 以实现 3 核并行
    if (core_id >= 0) {
        rknn_core_mask mask = RKNN_NPU_CORE_AUTO;
        if (core_id == 0)      mask = RKNN_NPU_CORE_0;
        else if (core_id == 1) mask = RKNN_NPU_CORE_1;
        else if (core_id == 2) mask = RKNN_NPU_CORE_2;
        else                   mask = RKNN_NPU_CORE_0_1_2;
        int cm = rknn_set_core_mask(ctx_, mask);
        if (cm < 0)
            fprintf(stderr, "[rknn] rknn_set_core_mask 警告 ret=%d (忽略, 回退 AUTO)\n", cm);
    }

    // 查询 SDK 版本 (诊断用)
    rknn_sdk_version ver;
    memset(&ver, 0, sizeof(ver));
    rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    printf("[rknn] SDK version: %s\n", ver.api_version);

    // 查询输入/输出数量
    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        fprintf(stderr, "[rknn] 查询 IO 数量失败 ret=%d\n", ret);
        deinit();
        return -4;
    }
    n_input_ = io_num.n_input;
    n_output_ = io_num.n_output;
    if (n_input_ != 1 || n_output_ != 1) {
        fprintf(stderr, "[rknn] 仅支持单输入单输出, 实际 n_input=%d n_output=%d\n",
                n_input_, n_output_);
        deinit();
        return -5;
    }

    // 查询输入属性
    input_attrs_ = new rknn_tensor_attr[n_input_];
    memset(input_attrs_, 0, sizeof(rknn_tensor_attr) * n_input_);
    for (int i = 0; i < n_input_; i++) {
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            fprintf(stderr, "[rknn] 查询输入属性[%d]失败 ret=%d\n", i, ret);
            deinit();
            return -6;
        }
    }
    // 输入为 NHWC [1,640,640,3]
    if (input_attrs_[0].fmt == RKNN_TENSOR_NCHW) {
        in_c_ = input_attrs_[0].dims[1];
        in_h_ = input_attrs_[0].dims[2];
        in_w_ = input_attrs_[0].dims[3];
    } else { // NHWC
        in_h_ = input_attrs_[0].dims[1];
        in_w_ = input_attrs_[0].dims[2];
        in_c_ = input_attrs_[0].dims[3];
    }
    printf("[rknn] 输入: [%d,%d,%d,%d] fmt=%s\n",
            input_attrs_[0].dims[0], input_attrs_[0].dims[1],
            input_attrs_[0].dims[2], input_attrs_[0].dims[3],
            input_attrs_[0].fmt == RKNN_TENSOR_NCHW ? "NCHW" : "NHWC");

    // 查询输出属性
    output_attrs_ = new rknn_tensor_attr[n_output_];
    memset(output_attrs_, 0, sizeof(rknn_tensor_attr) * n_output_);
    for (int i = 0; i < n_output_; i++) {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            fprintf(stderr, "[rknn] 查询输出属性[%d]失败 ret=%d\n", i, ret);
            deinit();
            return -7;
        }
    }
    // 输出为 NCHW [1,10,80,80]
    if (output_attrs_[0].fmt == RKNN_TENSOR_NCHW) {
        out_c_ = output_attrs_[0].dims[1];
        out_h_ = output_attrs_[0].dims[2];
        out_w_ = output_attrs_[0].dims[3];
    } else {
        out_h_ = output_attrs_[0].dims[1];
        out_w_ = output_attrs_[0].dims[2];
        out_c_ = output_attrs_[0].dims[3];
    }
    printf("[rknn] 输出: [%d,%d,%d,%d] type=%d fmt=%s\n",
            output_attrs_[0].dims[0], output_attrs_[0].dims[1],
            output_attrs_[0].dims[2], output_attrs_[0].dims[3],
            output_attrs_[0].type,
            output_attrs_[0].fmt == RKNN_TENSOR_NCHW ? "NCHW" : "NHWC");

    // 配置输入: 喂 UINT8 RGB888, 由 runtime 内部量化到 INT8
    memset(inputs_, 0, sizeof(inputs_));
    inputs_[0].index = 0;
    inputs_[0].type = RKNN_TENSOR_UINT8;
    inputs_[0].size = in_w_ * in_h_ * in_c_;
    inputs_[0].fmt = RKNN_TENSOR_NHWC;
    inputs_[0].pass_through = 0;

    initialized_ = true;
    return 0;
}

int RKNNInfer::infer(const unsigned char *input_buf)
{
    if (!initialized_) {
        fprintf(stderr, "[rknn] 未初始化\n");
        return -1;
    }
    if (!input_buf) {
        fprintf(stderr, "[rknn] 输入 buffer 为空\n");
        return -2;
    }

    inputs_[0].buf = (void *)input_buf;

    int ret = rknn_inputs_set(ctx_, n_input_, inputs_);
    if (ret < 0) {
        fprintf(stderr, "[rknn] rknn_inputs_set 失败 ret=%d\n", ret);
        return -3;
    }

    ret = rknn_run(ctx_, NULL);
    if (ret < 0) {
        fprintf(stderr, "[rknn] rknn_run 失败 ret=%d\n", ret);
        return -4;
    }

    return 0;
}

int RKNNInfer::postprocess(cv::Mat &label_mask)
{
    if (!initialized_) {
        fprintf(stderr, "[rknn] 未初始化\n");
        return -1;
    }

    // want_float=1 让 runtime 自动把输出转成 float32, 无论模型输出是 INT8/FLOAT16/FLOAT32 都正确
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;

    int ret = rknn_outputs_get(ctx_, n_output_, outputs, NULL);
    if (ret < 0) {
        fprintf(stderr, "[rknn] rknn_outputs_get 失败 ret=%d\n", ret);
        return -2;
    }
    if (!outputs[0].buf) {
        fprintf(stderr, "[rknn] 输出 buffer 为空\n");
        rknn_outputs_release(ctx_, n_output_, outputs);
        return -3;
    }

    // 输出 NCHW [1, out_c_, out_h_, out_w_], runtime 已转成 float32
    const float *out = (const float *)outputs[0].buf;
    const int plane = out_h_ * out_w_;

    // argmax over channel -> 80x80 CV_8UC1 标签图 (0 ~ out_c_-1)
    label_mask.create(out_h_, out_w_, CV_8UC1);
    for (int h = 0; h < out_h_; h++) {
        uchar *row = label_mask.ptr<uchar>(h);
        for (int w = 0; w < out_w_; w++) {
            int best_cls = 0;
            float best_val = out[h * out_w_ + w]; // c=0
            for (int c = 1; c < out_c_; c++) {
                float v = out[c * plane + h * out_w_ + w];
                if (v > best_val) {
                    best_val = v;
                    best_cls = c;
                }
            }
            row[w] = (uchar)best_cls;
        }
    }

    rknn_outputs_release(ctx_, n_output_, outputs);
    return 0;
}

int RKNNInfer::postprocess_smooth(cv::Mat &label_mask, int dst_h, int dst_w)
{
    if (!initialized_) {
        fprintf(stderr, "[rknn] 未初始化\n");
        return -1;
    }

    // want_float=1 让 runtime 自动把输出转成 float32, 保留模型软概率 (比 argmax 后的硬标签信息更丰富)
    // 无论模型输出是 INT8/FLOAT16/FLOAT32 都正确, 不依赖模型输出类型
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;

    int ret = rknn_outputs_get(ctx_, n_output_, outputs, NULL);
    if (ret < 0) {
        fprintf(stderr, "[rknn] rknn_outputs_get 失败 ret=%d\n", ret);
        return -2;
    }
    if (!outputs[0].buf) {
        fprintf(stderr, "[rknn] 输出 buffer 为空\n");
        rknn_outputs_release(ctx_, n_output_, outputs);
        return -3;
    }

    const float *out = (const float *)outputs[0].buf;
    const int plane = out_h_ * out_w_;

    // 1. 把 [10, out_h, out_w] float32 拆成 10 个 float32 通道 (保留软概率)
    std::vector<cv::Mat> ch(out_c_);
    for (int c = 0; c < out_c_; c++) {
        cv::Mat m(out_h_, out_w_, CV_32F);
        for (int h = 0; h < out_h_; h++) {
            float *r = m.ptr<float>(h);
            const float *src = out + c * plane + h * out_w_;
            for (int w = 0; w < out_w_; w++) r[w] = src[w];
        }
        ch[c] = m;
    }

    // 2. 每通道 CUBIC 平滑放大到目标尺寸 (软概率连续, 边界亚像素过渡)
    std::vector<cv::Mat> up(out_c_);
    for (int c = 0; c < out_c_; c++)
        cv::resize(ch[c], up[c], cv::Size(dst_w, dst_h), 0, 0, cv::INTER_CUBIC);

    // 3. 全分辨率 argmax -> 离散标签图 (边缘平滑, 每像素仍只属一类)
    label_mask.create(dst_h, dst_w, CV_8UC1);
    for (int h = 0; h < dst_h; h++) {
        uchar *lr = label_mask.ptr<uchar>(h);
        std::vector<const float *> pr(out_c_);
        for (int c = 0; c < out_c_; c++) pr[c] = up[c].ptr<float>(h);
        for (int w = 0; w < dst_w; w++) {
            int best = 0;
            float bestv = pr[0][w];
            for (int c = 1; c < out_c_; c++) {
                float v = pr[c][w];
                if (v > bestv) { bestv = v; best = c; }
            }
            lr[w] = (uchar)best;
        }
    }

    rknn_outputs_release(ctx_, n_output_, outputs);
    return 0;
}

void RKNNInfer::deinit()
{
    if (input_attrs_) {
        delete[] input_attrs_;
        input_attrs_ = NULL;
    }
    if (output_attrs_) {
        delete[] output_attrs_;
        output_attrs_ = NULL;
    }
    if (initialized_ && ctx_) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
    initialized_ = false;
}
