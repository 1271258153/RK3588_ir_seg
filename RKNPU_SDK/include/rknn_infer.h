#ifndef _RKNN_INFER_H
#define _RKNN_INFER_H

#include <opencv2/core/core.hpp>
#include "rknn_api.h"

// 模块化 RKNN 推理封装: 单输入(640x640x3 RGB UINT8) -> 单输出([1,10,80,80] FLOAT16)
class RKNNInfer
{
public:
    RKNNInfer();
    ~RKNNInfer();

    // 加载 rknn 模型, 查询输入/输出张量属性
    // core_id: NPU 绑核 (-1=AUTO, 0/1/2=绑定到指定核, 多线程时各实例绑不同核以并行)
    int init(const char *model_path, int core_id = -1);

    // 推理: input_buf 为 RGB888 NHWC, 大小 = input_w()*input_h()*input_c()
    int infer(const unsigned char *input_buf);

    // 后处理: 对 [10,80,80] 在通道维做 argmax, 输出 80x80 CV_8UC1 标签图(0-9)
    int postprocess(cv::Mat &label_mask);

    // 后处理(平滑版): 对原始 [10,80,80] 软输出先 CUBIC 放大到 dst_h x dst_w, 再 argmax
    // 输出 dst_h x dst_w CV_8UC1 标签图, 边缘亚像素平滑且保持离散类别
    // 比 postprocess+one-hot 上采样质量更高 (保留模型软概率, 边界更贴合)
    int postprocess_smooth(cv::Mat &label_mask, int dst_h, int dst_w);

    // 释放资源
    void deinit();

    int input_w() const { return in_w_; }
    int input_h() const { return in_h_; }
    int input_c() const { return in_c_; }
    int out_c() const { return out_c_; }
    int out_h() const { return out_h_; }
    int out_w() const { return out_w_; }

private:
    rknn_context ctx_;
    bool initialized_;

    int in_w_, in_h_, in_c_;    // 模型输入尺寸
    int out_c_, out_h_, out_w_;  // 模型输出尺寸

    rknn_tensor_attr *input_attrs_;
    rknn_tensor_attr *output_attrs_;
    int n_input_, n_output_;

    rknn_input inputs_[1];
};

#endif // _RKNN_INFER_H
