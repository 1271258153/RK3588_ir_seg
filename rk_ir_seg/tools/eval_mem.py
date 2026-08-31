from rknn.api import RKNN
import os

OUTPUT_DIR = './output'
RKNN_PATH = os.path.join(OUTPUT_DIR, 'best.rknn')


def _to_mib(nbytes):
    return nbytes / (1024 * 1024)


def save_mem_profile(mem_result, model_path, out_path):
    """按 RKNN eval_memory 的打印格式写入文件。"""
    weight = mem_result.get('weight_memory', 0)
    internal = mem_result.get('internal_memory', 0)
    other = mem_result.get('other_memory', 0)
    total = mem_result.get('total_memory', weight + internal + other)
    model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0

    lines = [
        '======================================================',
        '            Memory Profile Info Dump                  ',
        '======================================================',
        'NPU model memory detail(bytes):',
        f'    Weight Memory: {_to_mib(weight):.2f} MiB',
        f'    Internal Tensor Memory: {_to_mib(internal):.2f} MiB',
        f'    Other Memory: {_to_mib(other):.2f} MiB',
        f'    Total Memory: {_to_mib(total):.2f} MiB',
        '',
        'INFO: When evaluating memory usage, we need consider  ',
        f'the size of model, current model size is: {_to_mib(model_size):.2f} MiB       ',
        '======================================================',
        '',
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建RKNN对象
    rknn = RKNN(verbose=True)

    # 使用load_rknn接口加载RKNN模型
    rknn.load_rknn(
        path = os.path.join(OUTPUT_DIR, 'best.rknn'),
    )

    # 使用init_runtime接口初始化RKNN运行时环境
    # 评估内存时：perf_debug=False，eval_mem=True
    rknn.init_runtime(
        target='rk3588',
        perf_debug=False,          # 关闭性能评估
        eval_mem=True,             # 开启内存评估模式
    )

    # 使用eval_memory接口进行内存评估
    mem_result = rknn.eval_memory(is_print=True)

    # 按终端打印格式保存到 output/eval_mem.txt
    out_path = os.path.join(OUTPUT_DIR, 'eval_mem.txt')
    if isinstance(mem_result, dict):
        save_mem_profile(mem_result, RKNN_PATH, out_path)
        print(f'--> Memory result saved to {out_path}')
    else:
        print('eval_memory returned unexpected result, skip save.')

    rknn.release()
