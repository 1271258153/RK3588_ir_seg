"""量化精度分析：构建 INT8 模型并对每层做 FP32 vs 量化 cos/euc 对比。

支持通过命令行参数或环境变量切换量化校准算法，便于做
normal / kl_divergence / mmse 三组对比实验（论文用）。

用法:
    python tools/accuracy_analysis.py                                 # 默认 normal
    python tools/accuracy_analysis.py --quantized-algorithm kl_divergence
    python tools/accuracy_analysis.py --quantized-algorithm mmse --output-suffix mmse
    QUANTIZED_ALGORITHM=mmse python tools/accuracy_analysis.py
"""
import os
import glob
import shutil
import argparse
from rknn.api import RKNN

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
                    help='产物目录后缀；不填则用算法名，例如 mmse -> output_mmse/snapshot_mmse')
    args, _ = ap.parse_known_args()
    return args.quantized_algorithm, args.output_suffix


if __name__ == "__main__":
    quant_algo, output_suffix = get_quantized_algorithm()
    suffix = output_suffix if output_suffix else ("" if quant_algo == "normal" else f"_{quant_algo}")
    output_dir = OUTPUT_DIR + suffix
    snapshot_dir = os.path.join(output_dir, 'snapshot')
    os.makedirs(snapshot_dir, exist_ok=True)

    print(f'--> Quantized algorithm : {quant_algo}')
    print(f'--> Output dir          : {output_dir}')

    # verbose = True：打印详细日志
    rknn = RKNN(verbose=True)

    # 调用 config 接口配置要生成的RKNN模型
    # mean_values：预处理要减去的均值化参数
    # std_values：预处理要除以的标准化参数
    # quantized_algorithm：量化校准算法（normal/kl_divergence/mmse）
    rknn.config(
        mean_values=[[163.8008, 28.7509, 111.2492]],
        std_values=[[59.8040, 38.5198, 50.8686]],
        quant_img_RGB2BGR=False,
        target_platform='rk3588',                   # 目标平台：rk3588
        optimization_level=3,
        quantized_algorithm=quant_algo,
    )

    # 调用 load_onnx 接口加载 ONNX 模型
    rknn.load_onnx(model='data/best.onnx')

    # 调用 build 接口生成RKNN模型（精度分析前必须先量化构建）
    print('--> Building model with INT8 Quantization')
    rknn.build(
        do_quantization=True,                       # 是否进行量化
        dataset='data/dataset.txt',                 # 量化数据集路径
    )

    # verbose 模式下中间过程 onnx 默认写在当前目录，移到 output/
    for path in glob.glob('check*.onnx'):
        shutil.move(path, os.path.join(output_dir, os.path.basename(path)))

    # 调用 accuracy_analysis 接口进行模型量化精度分析
    # target=None：仅在 PC 模拟器对比 FP32 vs 量化；连板分析时再改为 'rk3588'
    print('--> Accuracy analysis')
    ret = rknn.accuracy_analysis(
        inputs=['pic/1_5.png'],                     # 进行推理的图像路径列表
        output_dir=snapshot_dir,
        target='rk3588',
        device_id=None,
    )
    if ret != 0:
        print('Accuracy analysis failed!')
        exit(ret)

    # 调用 release 接口释放RKNN模型
    rknn.release()
