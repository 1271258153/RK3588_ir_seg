from rknn.api import RKNN
import os
import shutil

OUTPUT_DIR = './output'

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建RKNN对象
    rknn = RKNN(verbose=True)

    # 使用load_rknn接口加载RKNN模型
    rknn.load_rknn(
        path = os.path.join(OUTPUT_DIR, 'best.rknn'),
    )

    # 使用init_runtime接口初始化RKNN运行时环境
    # 评估性能时：perf_debug=True，eval_mem=False
    rknn.init_runtime(
        target='rk3588',
        perf_debug=True,           # 开启性能评估的debug模式
        eval_mem=False,            # 关闭内存评估模式
    )

    # 使用eval_perf接口进行模型性能评估
    rknn.eval_perf(is_print=True)

    # 将 toolkit 生成的 eval_perf.csv 移到 output/
    src = 'eval_perf.csv'
    dst = os.path.join(OUTPUT_DIR, 'eval_perf.csv')
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f'--> Perf result saved to {dst}')

    rknn.release()
