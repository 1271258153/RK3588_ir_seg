"""混合量化 step2：导入 step1 产物完成量化、精度分析并导出 rknn。

支持通过命令行参数或环境变量切换量化校准算法，与 step1.py 的产物目录约定一致，
便于做 normal / kl_divergence / mmse 三组对比实验（论文用）。

用法:
    python tools/step2.py                                 # 默认 normal，读 output/，写 output/q_snapshot
    python tools/step2.py --quantized-algorithm kl_divergence
    python tools/step2.py --quantized-algorithm mmse --output-suffix mmse
    QUANTIZED_ALGORITHM=mmse python tools/step2.py

目录约定（与 step1.py 一致）:
    normal        -> output/                    （output/hybrid_quantization -> output/q_snapshot, output/best.rknn）
    kl_divergence -> output_kl_divergence/       （.../hybrid_quantization -> .../q_snapshot, .../best.rknn）
    mmse          -> output_mmse/                （.../hybrid_quantization -> .../q_snapshot, .../best.rknn）
"""
from rknn.api import RKNN
import os
import glob
import shutil
import argparse

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
                    help='量化校准算法 (default: normal, env: QUANTIZED_ALGORITHM)\n'
                         '需与 step1.py 保持一致，否则会找不到 step1 产物')
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
    snapshot_dir = os.path.join(output_dir, 'q_snapshot')
    os.makedirs(snapshot_dir, exist_ok=True)

    print(f'--> Quantized algorithm : {quant_algo}')
    print(f'--> Output dir          : {output_dir}')
    print(f'--> Read  hybrid from   : {hybrid_dir}')
    print(f'--> Write  snapshot to   : {snapshot_dir}')

    # 校验 step1 产物是否存在，避免空跑
    for fname, label in (('best.model', 'model'),
                         ('best.data', 'data'),
                         ('best.quantization.cfg', 'cfg')):
        p = os.path.join(hybrid_dir, fname)
        if not os.path.exists(p):
            print(f'[ERROR] 找不到 step1 产物 {label}: {p}\n'
                  f'请先运行: python tools/step1.py --quantized-algorithm {quant_algo}')
            exit(1)

    rknn = RKNN(verbose=True)

    # 调用 hybrid_quantization_step2 接口进行量化的第二步
    rknn.hybrid_quantization_step2(
        model_input=os.path.join(hybrid_dir, "best.model"),            # step1生成的模型文件
        data_input=os.path.join(hybrid_dir, "best.data"),              # step1生成的配置文件
        model_quantization_cfg=os.path.join(hybrid_dir, "best.quantization.cfg"),  # step1生成的量化cfg文件
    )

    # 调用accuracy_analysis 对量化后的模型进行精度分析
    rknn.accuracy_analysis(
        inputs=["pic/1_5.png"],        # 进行推理的图像数据集路径
        output_dir=snapshot_dir,        # 输出目录
        target="rk3588",                # 目标平台：rk3588
        device_id=None,                 # 电脑连接了多个开发板，设备的选择
    )

    # verbose 模式下中间过程 onnx 默认写在当前目录，移到 output{suffix}/
    for path in glob.glob('check*.onnx'):
        shutil.move(path, os.path.join(output_dir, os.path.basename(path)))

    # 调用 RKNN 导出接口导出 RKNN 模型
    # 按算法命名：normal->best.rknn, kl_divergence->best_kl.rknn, mmse->best_mmse.rknn
    rknn_name = {
        'normal': 'best.rknn',
        'kl_divergence': 'best_kl.rknn',
        'mmse': 'best_mmse.rknn',
    }.get(quant_algo, f'best_{quant_algo}.rknn')
    print('--> Export rknn model')
    rknn.export_rknn(
        export_path=os.path.join(output_dir, rknn_name),
    )

    rknn.release()
