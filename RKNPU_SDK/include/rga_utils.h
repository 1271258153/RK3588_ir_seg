#ifndef _RGA_UTILS_H
#define _RGA_UTILS_H

#include <opencv2/core/core.hpp>

// 硬件加速前处理: 将任意尺寸 BGR cv::Mat 缩放到 dst_w x dst_h 并转为 RGB
// 绝对禁止使用 OpenCV 的 cv::resize / cv::cvtColor, 全部由 RK3588 librga (im2d) 完成
// out_buf 由调用方分配, 大小 = dst_w * dst_h * 3 字节, 数据为 RGB888 排列
// 返回 0 成功, 负数失败
int rga_preprocess(const cv::Mat &bgr_img, int dst_w, int dst_h,
                   unsigned char *out_buf);

// 硬件加速 NV12 -> BGR888 转换 (单次 imresize 同时完成色彩转换 + 拷贝)
//   nv12      : MPP 解码输出指针 (Y 平面 + 紧随 UV 平面)
//   width     : 有效画面宽
//   height    : 有效画面高
//   hor_stride: Y 行跨距 (像素), 16 对齐, MPP 输出
//   ver_stride: Y 列跨距 (行),  16 对齐, MPP 输出
//   bgr_out   : 调用方分配, 大小 = width * height * 3, BGR888 排列
// 返回 0 成功, 负数失败
int rga_nv12_to_bgr(const unsigned char *nv12, int width, int height,
                    int hor_stride, int ver_stride, unsigned char *bgr_out);

// 硬件加速 NV12 -> RGB888 缩放转换 (单次 imresize 同时完成 缩放 + YUV2RGB 色彩转换)
// 用于把 MPP 解码出的 NV12 帧直接转成 NPU 模型输入 (RGB888, dst_w x dst_h)
//   nv12       : MPP 解码输出指针
//   width/height : 源有效画面宽高
//   hor_stride/ver_stride : 源跨距 (16 对齐)
//   rgb_out    : 调用方分配, 大小 = dst_w * dst_h * 3, RGB888 排列
//   dst_w/dst_h : 目标尺寸 (如 NPU 输入 640x640)
// 返回 0 成功, 负数失败
int rga_nv12_to_rgb_resized(const unsigned char *nv12, int width, int height,
                            int hor_stride, int ver_stride,
                            unsigned char *rgb_out, int dst_w, int dst_h);

#endif // _RGA_UTILS_H
