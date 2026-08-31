"""读取 RKNN 输出的 eval_perf.csv，累加算子耗时并估算 FPS。

用法:
    python3 tools/eval_summary.py
    python3 tools/eval_summary.py --perf output/eval_perf.csv
"""
import argparse
import csv
import os

DEFAULT_PERF = os.path.join('output', 'eval_perf.csv')


def parse_perf(perf_path):
    """从 eval_perf.csv 解析每个算子的耗时(us)。"""
    total_us = 0
    n_ops = 0
    with open(perf_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row or len(row) < 15:
                continue
            rid = row[0].strip()
            if not rid.isdigit():
                continue
            try:
                t_us = int(row[9])
            except ValueError:
                continue
            total_us += t_us
            n_ops += 1
    return n_ops, total_us


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--perf', default=DEFAULT_PERF, help='eval_perf.csv 路径')
    args = ap.parse_args()

    if not os.path.exists(args.perf):
        print(f'找不到文件: {args.perf}')
        return

    n_ops, total_us = parse_perf(args.perf)
    fps = 1e6 / total_us if total_us else 0

    print(f'算子总数 : {n_ops}')
    print(f'总耗时   : {total_us} us = {total_us/1000:.2f} ms')
    print(f'估算 FPS : {fps:.2f}')


if __name__ == '__main__':
    main()
