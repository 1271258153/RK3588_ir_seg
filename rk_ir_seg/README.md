# 基于RK3588的红外发热检测系统

## 1. 📁 仓库目录说明
```text
rk_ir_seg/
├── data/                          # 输入模型与校准数据列表
│   ├── best.onnx
│   └── dataset.txt
├── pic/                           # 校准 / 测试用红外图片
├── tools/                         # PC 端导出与量化脚本
│   ├── accuracy_analysis.py
│   ├── step1.py
│   ├── step2.py
│   ├── sync_cfg.py                # 同步 custom_quantize_layers 到各算法 cfg
│   ├── eval_perf.py
│   ├── eval_mem.py
│   ├── eval_summary.py
│   └── pareto_plot.py             # 混合精度 Pareto 敏感度散点图
├── output/                        # 导出、分析与量化输出
└── README.md
```

## 2. 简要的操作步骤
```bash
# 先在开发板中开启rknn_serve服务
rknn_server

adb shell "mkdir -p /userdata/dumps && chmod 777 /userdata/dumps" # 创建/userdata/dumps
# 修改脚本中模型对应的 mean_values 和 std_values 参数
python tools/accuracy_analysis.py      # 量化精度分析，需要连接开发板
python tools/step1.py                  # 量化step1,生成cfg文件
python tools/step2.py                  # 量化step2,导出best.rknn，需要连接开发板
python tools/eval_perf.py              # 评估性能
python tools/eval_mem.py               # 评估内存
python tools/eval_summary.py           # 推算大致的fps
```

## 3. 开发环境
1. 虚拟机（PC 端）
   - 系统：Ubuntu 20.04
   - 交叉编译器：`gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu`
   - 工具：Miniconda、Python 3.8、RKNN-Toolkit2-1.6

2. 开发板
   - 硬件：RK3588 8G+64G（Kernel 5.10）
   - 工具：Python 3.9、RKNN-Toolkit-Lite2-1.6、RKNPU2

## 4. 部署前准备
### 1. 模型及数据集准备
1. 将图片放入 `pic/` 下
2. 将 `best.onnx`、`dataset.txt`（图片的路径）放入 `data/` 下 

### 2. 量化精度分析
- 在量化之前，需要对模型网络的每一层进行精度分析，对某些层进行IN8量化可能导致数值爆炸，需要连接开发板
- 先在开发板中开启rknn_server服务 `rknn_server`
```bash
python tools/accuracy_analysis.py
```
> 会在 `output/snapshot` 下生成 error_analysis.txt 文件，查看该文件发现
> pag3/Div、pag3/edge_attn、pag4/Div、pag4/edge_attn、ema/pool_5、ema/pool_7 精度很差
> 在后面混合量化中要取消对这些层的INT8量化

### 3. 量化并导出rknn
- 先执行 step1.py ,生成cfg文件
```bash
python tools/step1.py
```
|参数|含义|
|-|-|
|proposal|True表示自动产生混合量化的配置建议值|
|proposal_dataset_size|使用的图片数量(值越大程序运行的越慢)|
> 会在 `output/hybrid_quantization/` 下生成 best.data、best.model、best.quantization.cfg
> 在 `output/hybrid_quantization/best.quantization.cfg` 中修改要对哪些层非量化

- 执行 step2.py, 导入 step1 生成的文件进行量化，最终导出 best.rknn，需要连接开发板
```bash
python tools/step2.py
```
> 会在 `output/q_snapshot` 生成量化后的精度分析文件 `error_analysis.txt`，可与 `output/snapshot/error_analysis.txt` 进行对比，看 best.quantization.cfg 中的配置是否合理
> `step2.py` 同样支持 `--quantized-algorithm` / `QUANTIZED_ALGORITHM` 切换，与 step1.py 目录约定一致：
> 不同算法的 q_snapshot 与 best.rknn 会分别写入 `output/q_snapshot`、`output_kl_divergence/q_snapshot`、`output_mmse/q_snapshot`，互不覆盖

### 4. 性能和内存评估
- 对量化后的rknn模型进行性能和内存评估
```bash
python tools/eval_perf.py           # 评估性能
python tools/eval_mem.py            # 评估内存
python tools/eval_summary.py        # 推算大致的fps
```
> 在 `output/` 下生成 eval_perf.csv 和 eval_mem.txt

## 5. h264 / RTSP 流测试（伪装海康摄像头）
> 板端 MPP 最终仍吃 H.264 NALU；PC 侧转 MP4 仅便于 `-stream_loop` 无缝循环推流。
> `pic/` 含 4 种分辨率，**必须分开编码**（单路 H.264 不能混分辨率）。

```bash
cd /home/topeet/pidnet
mkdir -p h264_out/by_res/{1280x720,640x480,1080x1440,480x640}

# 0) 按分辨率分类（只需一次；需 identify / imagemagick）
for f in pic/*; do
  s=$(identify -format '%wx%h' "$f")
  case "$s" in
    1280x720|640x480|1080x1440|480x640)
      ln -sfn "$(realpath "$f")" "h264_out/by_res/$s/$(basename "$f")" ;;
  esac
done
for d in h264_out/by_res/*; do echo "$(basename "$d"): $(ls "$d" | wc -l)"; done
# 期望：1280x720=94, 640x480=21, 1080x1440=4, 480x640=1

# 1) 四种分辨率各自 → H.264 裸流
ffmpeg -y -framerate 25 -pattern_type glob -i 'h264_out/by_res/1280x720/*.png' \
  -c:v libx264 -profile:v high -level 4.0 \
  -pix_fmt yuv420p -bf 0 -g 25 -x264-params "annexb=1" \
  -an -f h264 h264_out/1280x720.h264

ffmpeg -y -framerate 25 -pattern_type glob -i 'h264_out/by_res/640x480/*.png' \
  -c:v libx264 -profile:v high -level 4.0 \
  -pix_fmt yuv420p -bf 0 -g 25 -x264-params "annexb=1" \
  -an -f h264 h264_out/640x480.h264

ffmpeg -y -framerate 25 -pattern_type glob -i 'h264_out/by_res/1080x1440/*.png' \
  -c:v libx264 -profile:v high -level 4.0 \
  -pix_fmt yuv420p -bf 0 -g 25 -x264-params "annexb=1" \
  -an -f h264 h264_out/1080x1440.h264

ffmpeg -y -framerate 25 -pattern_type glob -i 'h264_out/by_res/480x640/*.png' \
  -c:v libx264 -profile:v high -level 4.0 \
  -pix_fmt yuv420p -bf 0 -g 25 -x264-params "annexb=1" \
  -an -f h264 h264_out/480x640.h264

# 每秒只出一帧
ffmpeg -y -framerate 1 -pattern_type glob -i 'h264_out/by_res/1280x720/*.png' \
  -c:v libx264 -profile:v high -level 4.0 \
  -pix_fmt yuv420p -bf 0 -g 1 -x264-params "annexb=1" \
  -an -f h264 h264_out/1280x720_1fps.h264

# 2) 裸流 → MP4（推流用；-c copy 不重编码）
ffmpeg -y -framerate 25 -i h264_out/1280x720.h264 -c copy h264_out/1280x720.mp4
ffmpeg -y -framerate 25 -i h264_out/640x480.h264  -c copy h264_out/640x480.mp4
ffmpeg -y -framerate 25 -i h264_out/1080x1440.h264 -c copy h264_out/1080x1440.mp4
ffmpeg -y -framerate 25 -i h264_out/480x640.h264  -c copy h264_out/480x640.mp4

# 3) 启动 MediaMTX（另开终端）
cd /home/topeet/pidnet/tools/mediamtx
./mediamtx ./mediamtx.yml

# 4) 任选一路循环推流
cd /home/topeet/pidnet
ffmpeg -re -stream_loop -1 -i h264_out/1280x720.mp4 \
  -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam
# 其他：cam480←640x480.mp4  cam1440←1080x1440.mp4  cam640v←480x640.mp4

# 5) 板端拉流验证
ffmpeg -rtsp_transport tcp -i rtsp://192.168.137.3:8554/cam -f null -
```

## 6. 量化校准算法对比（论文实验）
`accuracy_analysis.py`、`step1.py`、`step2.py` 均支持通过 `--quantized-algorithm`（短选项 `--qa`）
或环境变量 `QUANTIZED_ALGORITHM` 切换校准算法，便于做 normal / kl_divergence / mmse 三组对比
（红外长尾分布下 Min-Max 易被极端值拉宽 scale）。

三组对比的完整流程（每组三步，需连板）：
```bash
# 1) normal（默认 Min-Max） ！！normal组上面默认已经跑了，重新跑会覆盖掉原来设置好的cfg文件
#python tools/accuracy_analysis.py --quantized-algorithm normal
#python tools/step1.py                  --quantized-algorithm normal

# 2) kl_divergence（KL 散度截断）
python tools/accuracy_analysis.py --quantized-algorithm kl_divergence
python tools/step1.py                  --quantized-algorithm kl_divergence

# 3) mmse（最小均方误差）
python tools/accuracy_analysis.py --quantized-algorithm mmse
python tools/step1.py                  --quantized-algorithm mmse

python tools/sync_cfg.py               # 同步混合精度配置到各算法 cfg

# 跑三次混合量化，生成对应的output_xxx/q_snapshot 和 rknn 文件
python tools/step2.py                  --quantized-algorithm normal
python tools/step2.py                  --quantized-algorithm kl_divergence
python tools/step2.py                  --quantized-algorithm mmse
```

目录约定（三个脚本一致）：
| 算法 | 产物根目录 | snapshot | q_snapshot | best.rknn |
|-|-|-|-|-|
| normal | `output/` | `output/snapshot` | `output/q_snapshot` | `output/best.rknn` |
| kl_divergence | `output_kl_divergence/` | `.../snapshot` | `.../q_snapshot` | `.../best.rknn` |
| mmse | `output_mmse/` | `.../snapshot` | `.../q_snapshot` | `.../best.rknn` |

> 对比三组 `error_analysis.txt` 的逐层 cos/euc 与最终 Mask 输出，即可在论文中给出
> 「Min-Max / KL / MMSE 校准算法选型对比」实验。


## 6. 混合精度 Pareto 敏感度评估
基于 `output/q_snapshot/error_analysis.txt`（逐层 cos/euc）与 `output/eval_perf.csv`（逐层耗时），
生成「FP16 化延迟代价 vs INT8 量化误差」二维散点图，量化每一层提 FP16 的性价比，
指导 `best.quantization.cfg` 中 `custom_quantize_layers` 的选择：
```bash
python tools/pareto_plot.py   # 混合量化后再看还有哪些层可以提f16

# 或者量化前（全 INT8）看哪些层最敏感（用来决定 cfg 该加哪些层）
python tools/pareto_plot.py --error output/snapshot/error_analysis.txt --out output/pareto_snapshot.png

# 或指定输入输出
python tools/pareto_plot.py \
    --error output/q_snapshot/error_analysis.txt \
    --perf  output/eval_perf.csv \
    --out   output/pareto.png \
    --csv   output/pareto_sensitivity.csv
```
|参数|含义|
|-|-|
|--error|error_analysis.txt 路径（默认 output/q_snapshot/error_analysis.txt）|
|--perf|eval_perf.csv 路径（默认 output/eval_perf.csv）|
|--out|输出 PNG 路径（默认 output/pareto.png）|
|--csv|输出敏感度表 CSV 路径（默认 output/pareto_sensitivity.csv）|
|--no-plot|只导出 CSV，不画图（matplotlib 缺失时自动启用）|

> - 纵坐标 `1 - cos`（simulator single）= 该层保持 INT8 的量化误差
> - 横坐标 `time(us)` = 该层 INT8 耗时，近似提 FP16 的延迟代价
> - 红点为 Pareto 前沿（敏感度高且延迟低，最该提 FP16）
> - 当前 `best.quantization.cfg` 已据此将 pag3、pag4 的 Div/edge_attn/edge_expand 及 ema/pool_5/pool_7 提为 float16
> - 画图需 `pip install matplotlib`，未安装时脚本自动降级为只导出 CSV


- Normal:
各类 IoU:
  [0] _background_    : IoU=0.8536  (TP=19449149 FP=1285237 FN=2050529)
  [1] BL_Device       : IoU=0.8098  (TP=727993 FP=42767 FN=128247)
  [2] CC_Server       : IoU=0.8779  (TP=6567696 FP=382254 FN=530882)
  [3] DP_Server       : IoU=0.9014  (TP=3336832 FP=158194 FN=206990)
  [4] KDVideo_Device  : IoU=0.8366  (TP=5862166 FP=962368 FN=182688)
  [5] KVM_Switcher    : IoU=0.9269  (TP=484999 FP=21025 FN=17229)
  [6] SP_Cloud        : IoU=0.8658  (TP=1230382 FP=33266 FN=157477)
  [7] VPN_Gateway     : IoU=0.8596  (TP=2682533 FP=225091 FN=213121)
  [8] WEB_Firewall    : IoU=0.8660  (TP=2068434 FP=146745 FN=173423)
  [9] YP_Server       : IoU=0.8196  (TP=2957552 FP=527317 FN=123678)

---------- 汇总 ----------
mIoU (含背景, 10 类) = 0.8617
mIoU (不含背景, 9 类) = 0.8626
Pixel Accuracy = 0.9230
总有效像素 = 49152000


- kl:
各类 IoU:
  [0] _background_    : IoU=0.8560  (TP=19463656 FP=1237323 FN=2036022)
  [1] BL_Device       : IoU=0.8183  (TP=736660 FP=43960 FN=119580)
  [2] CC_Server       : IoU=0.8798  (TP=6592454 FP=394937 FN=506124)
  [3] DP_Server       : IoU=0.9063  (TP=3344215 FP=146116 FN=199607)
  [4] KDVideo_Device  : IoU=0.8421  (TP=5870799 FP=926670 FN=174055)
  [5] KVM_Switcher    : IoU=0.9237  (TP=485560 FP=23432 FN=16668)
  [6] SP_Cloud        : IoU=0.8689  (TP=1232412 FP=30574 FN=155447)
  [7] VPN_Gateway     : IoU=0.8560  (TP=2683875 FP=239781 FN=211779)
  [8] WEB_Firewall    : IoU=0.8693  (TP=2081804 FP=153004 FN=160053)
  [9] YP_Server       : IoU=0.8245  (TP=2958117 FP=506651 FN=123113)

---------- 汇总 ----------
mIoU (含背景, 10 类) = 0.8645
mIoU (不含背景, 9 类) = 0.8654
Pixel Accuracy = 0.9247
总有效像素 = 49152000

- mmse:
各类 IoU:
  [0] _background_    : IoU=0.8565  (TP=19467819 FP=1229469 FN=2031859)
  [1] BL_Device       : IoU=0.8220  (TP=740235 FP=44240 FN=116005)
  [2] CC_Server       : IoU=0.8780  (TP=6591802 FP=409488 FN=506776)
  [3] DP_Server       : IoU=0.9052  (TP=3343380 FP=149711 FN=200442)
  [4] KDVideo_Device  : IoU=0.8419  (TP=5859803 FP=915706 FN=185051)
  [5] KVM_Switcher    : IoU=0.9231  (TP=485074 FP=23278 FN=17154)
  [6] SP_Cloud        : IoU=0.8681  (TP=1231170 FP=30400 FN=156689)
  [7] VPN_Gateway     : IoU=0.8584  (TP=2690477 FP=238497 FN=205177)
  [8] WEB_Firewall    : IoU=0.8697  (TP=2081545 FP=151684 FN=160312)
  [9] YP_Server       : IoU=0.8244  (TP=2959574 FP=508648 FN=121656)

---------- 汇总 ----------
mIoU (含背景, 10 类) = 0.8647
mIoU (不含背景, 9 类) = 0.8656
Pixel Accuracy = 0.9247
总有效像素 = 49152000
