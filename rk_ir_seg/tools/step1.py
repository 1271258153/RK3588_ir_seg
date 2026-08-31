"""混合量化 step1：生成 .model / .data / .quantization.cfg。

支持通过命令行参数或环境变量切换量化校准算法，便于做
normal / kl_divergence / mmse 三组对比实验（论文用）。

用法:
    python tools/step1.py                                 # 默认 normal
    python tools/step1.py --quantized-algorithm kl_divergence
    python tools/step1.py --quantized-algorithm mmse --output-suffix mmse
    QUANTIZED_ALGORITHM=mmse python tools/step1.py
"""
from rknn.api import RKNN
import os
import glob
import shutil
import argparse

from mem_watchdog import start_mem_watchdog, stop_mem_watchdog

OUTPUT_DIR = './output'

# 支持的量化校准算法（与 rknn.config 的 quantized_algorithm 参数对齐）
SUPPORTED_ALGORITHMS = ('normal', 'kl_divergence', 'mmse')


def get_quantized_algorithm():
    """从命令行参数或环境变量读取量化算法，默认 normal。"""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--quantized-algorithm', '--qa',
                    default=os.environ.get('QUANTIZED_ALGORITHM', 'normal'),
                    choices=SUPPORTED_ALGORITHMS,
                    help='量化校准算法 (default: normal, env: QUANTIZED_ALGORITHM)')
    ap.add_argument('--output-suffix', '--os',
                    default=None,
                    help='产物目录后缀；不填则用算法名，例如 mmse -> output_mmse/')
    args, _ = ap.parse_known_args()
    return args.quantized_algorithm, args.output_suffix


if __name__ == "__main__":
    quant_algo, output_suffix = get_quantized_algorithm()
    suffix = output_suffix if output_suffix else ("" if quant_algo == "normal" else f"_{quant_algo}")
    output_dir = OUTPUT_DIR + suffix
    hybrid_dir = os.path.join(output_dir, 'hybrid_quantization')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hybrid_dir, exist_ok=True)

    print(f'--> Quantized algorithm : {quant_algo}')
    print(f'--> Output dir          : {output_dir}')

    # verbose = True：打印详细日志
    rknn = RKNN(verbose=True)

    # 调用 config 接口配置要生成的RKNN模型
    # mean_values：预处理要减去的均值化参数
    # std_values：预处理要除以的标准化参数
    # quantized_algorithm：量化校准算法
    #   - normal        : Min-Max（默认，对极端值敏感）
    #   - kl_divergence : 基于信息熵的截断，对长尾分布更鲁棒
    #   - mmse          : 最小均方误差，直接最小化量化前后 MSE
    rknn.config(
        mean_values=[[163.8008, 28.7509, 111.2492]],
        std_values=[[59.8040, 38.5198, 50.8686]],
        target_platform='rk3588',                   # 目标平台：rk3588
        quantized_algorithm=quant_algo,
    )

    # 调用 load_onnx 接口加载 ONNX 模型
    rknn.load_onnx(model='data/best.onnx')

    stop_event, mem_thread = start_mem_watchdog()

    # 混合量化 step1：内部会完成量化分析，不要先调用 build
    # 成功后会生成 .model / .data / .quantization.cfg 等文件
    try:
        ret = rknn.hybrid_quantization_step1(
            dataset='data/dataset.txt',                 # 量化数据集路径
            rknn_batch_size=1,                          # 推理时每个批次的数据量
            proposal=False,                              # 自动产生混合量化的配置建议值
            proposal_dataset_size=10,                   # proposal 使用的图片数量
        )
    finally:
        stop_mem_watchdog(stop_event, mem_thread)

    if ret != 0:
        print('Hybrid quantization step1 failed!')
        rknn.release()
        exit(ret)

    # check*.onnx 放到 output/；混合量化产物放到 output/hybrid_quantization/
    for path in glob.glob('check*.onnx'):
        shutil.move(path, os.path.join(output_dir, os.path.basename(path)))
    for pattern in ('*.model', '*.data', '*.quantization.cfg'):
        for path in glob.glob(pattern):
            dst = os.path.join(hybrid_dir, os.path.basename(path))
            if os.path.abspath(path) != os.path.abspath(dst):
                shutil.move(path, dst)

    # 调用 release 接口释放RKNN模型
    rknn.release()
