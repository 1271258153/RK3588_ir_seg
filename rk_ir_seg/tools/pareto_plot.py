"""混合精度 Pareto 散点图：解析 RKNN 的逐层精度分析 + 性能分析，
生成「FP16 化延迟代价 vs INT8 量化误差」二维散点图，供论文使用。

横坐标: 该层转化为 FP16 带来的推理延迟代价（用该层 INT8 耗时 us 近似）
纵坐标: 该层保持 INT8 导致的量化误差（1 - cos，cos 取 simulator single）
高亮:   落在 Pareto 前沿的层（敏感度高且延迟代价低，最该提 FP16）

用法:
    python tools/pareto_plot.py
    python tools/pareto_plot.py \
        --error output/q_snapshot/error_analysis.txt \
        --perf  output/eval_perf.csv \
        --out   output/pareto.png
    python tools/pareto_plot.py --no-plot          # 只导出 CSV，不画图
"""
import argparse
import csv
import os
import re

DEFAULT_ERROR = os.path.join('output', 'q_snapshot', 'error_analysis.txt')
DEFAULT_PERF = os.path.join('output', 'eval_perf.csv')
DEFAULT_OUT = os.path.join('output', 'pareto.png')
DEFAULT_CSV = os.path.join('output', 'pareto_sensitivity.csv')

# 量化误差取哪一列：simulator single cos（最能反映单层量化精度，不受误差累积影响）
# error_analysis.txt 列布局: [OpType] name  sim_cos_entire sim_euc_entire sim_cos_single sim_euc_single  rt_...
# 我们用 simulator single cos。
SIM_SINGLE_COS_COL = 4  # 0-indexed: 0=op, 1=name, 2=sim_cos_entire, 3=sim_euc_entire, 4=sim_cos_single

# 去掉 error_analysis 层名里的这些后缀，得到与 eval_perf.csv FullName 去掉 "OpType:" 后一致的基础名
NAME_SUFFIXES = (
    '_cvt_int8_float16', '_cvt_float16_int8',
    '__float16', '__int8',
    '_output_0',
)


def strip_layer_suffix(name):
    """把 /pag3/Add_output_0__float16 -> /pag3/Add"""
    for suf in NAME_SUFFIXES:
        if name.endswith(suf):
            name = name[:-len(suf)]
    return name


def parse_error_analysis(path):
    """解析 error_analysis.txt，返回 [(base_name, cos_single, euc_single), ...]。

    q_inf / q_nan 等异常值记为 cos=0（即误差 1.0，最大敏感度）。
    """
    rows = []
    line_re = re.compile(r'^\[(\w+)\]\s+(.*)$')
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            m = line_re.match(line)
            if not m:
                continue
            op_type = m.group(1)
            rest = m.group(2)
            # rest 形如: /pag3/Div_output_0   q_inf | q_inf   1.00000 | 0.0515  ...
            # 用空白切分，第一个 token 是层名
            tokens = rest.split()
            if len(tokens) < 5:
                continue
            layer_name = tokens[0]
            # 后面是 4 个 cos|euc 对（sim entire/single, rt entire/single），可能缺失
            # 简单取 tokens[2] (sim_cos_entire) tokens[4] (sim_cos_single)
            # 但因为 "| " 分隔，token 化后 cos 和 euc 是分开的 token
            # tokens: [name, cos_ent, '|', euc_ent, cos_sing, '|', euc_sing, rt_cos_ent, '|', rt_euc_ent, rt_cos_sing, '|', rt_euc_sing]
            # 注意 q_inf 这种会变成 'q_inf' 占一个 token
            def safe_float(tok):
                try:
                    return float(tok)
                except ValueError:
                    return None  # q_inf / q_nan / rt_inf / s_nan

            # 找到 sim single cos：跳过 name 后，按 '|' 分隔的 4 组里第 2 组的 cos
            # tokens[1] = sim_cos_entire, tokens[2]='|', tokens[3]=sim_euc_entire,
            # tokens[4] = sim_cos_single, tokens[5]='|', tokens[6]=sim_euc_single
            cos_single = safe_float(tokens[4]) if len(tokens) > 4 else None
            euc_single = safe_float(tokens[6]) if len(tokens) > 6 else None

            base = strip_layer_suffix(layer_name)
            rows.append({
                'op_type': op_type,
                'layer': layer_name,
                'base': base,
                'cos_single': cos_single,  # None 表示 q_inf/q_nan
                'euc_single': euc_single,
            })
    return rows


def parse_perf_csv(path):
    """解析 eval_perf.csv，返回 {base_name: time_us}。base_name = FullName 去掉 'OpType:' 前缀。"""
    perf = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
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
            full_name = row[14].strip()  # FullName
            if ':' in full_name:
                base = full_name.split(':', 1)[1]
            else:
                base = full_name
            # 同一 base 可能出现多次（如 Conv + 后续 exDataConvert），取最大耗时
            if base not in perf or t_us > perf[base]:
                perf[base] = t_us
    return perf


def compute_sensitivity(error_rows, perf):
    """合并精度与性能，计算每层敏感度 S = 1 - cos_single，延迟代价 = time_us。"""
    points = []
    for r in error_rows:
        base = r['base']
        cos = r['cos_single']
        # q_inf/q_nan 视为最大敏感度
        sens = 1.0 if cos is None else (1.0 - cos)
        t_us = perf.get(base, None)
        points.append({
            'op_type': r['op_type'],
            'layer': r['layer'],
            'base': base,
            'cos_single': cos,
            'sensitivity': sens,
            'time_us': t_us,  # None 表示在 perf 里没匹配到
        })
    return points


def pareto_front(points):
    """求 Pareto 前沿：敏感度越高、延迟代价越低的层越该提 FP16。
    最小化 time_us，最大化 sensitivity -> 等价于最小化 (-sensitivity, time_us)。
    """
    valid = [p for p in points if p['time_us'] is not None and p['time_us'] > 0]
    # 按 time_us 升序，扫描时维护已见最大 sensitivity
    valid.sort(key=lambda p: p['time_us'])
    front = []
    max_sens = -1
    for p in valid:
        if p['sensitivity'] > max_sens:
            front.append(p)
            max_sens = p['sensitivity']
    return front


def export_csv(points, out_csv):
    """导出敏感度表 CSV（论文附录用）。"""
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['op_type', 'layer', 'cos_single', 'sensitivity(1-cos)',
                    'time_us', 'on_pareto_front'])
        front_set = {id(p) for p in pareto_front(points)}
        # 按敏感度降序排，方便看哪些层最该提 FP16
        for p in sorted(points, key=lambda x: -x['sensitivity']):
            cos_str = '' if p['cos_single'] is None else f'{p["cos_single"]:.5f}'
            t_str = '' if p['time_us'] is None else str(p['time_us'])
            w.writerow([p['op_type'], p['layer'], cos_str,
                        f'{p["sensitivity"]:.5f}', t_str,
                        'Y' if id(p) in front_set else 'N'])
    print(f'--> Sensitivity CSV saved to {out_csv}')


def get_module(layer_path):
    """根据层路径前缀归类到模块，返回 (module_name, color)。"""
    # 去掉 __float16 / __int8 后缀的 exDataConvert 重复层，避免图面混乱
    if '__float16' in layer_path or '__int8' in layer_path:
        return None, None
    p = layer_path.lstrip('/')
    if p.startswith('spp'):
        return 'SPP', 'tab:red'
    if p.startswith('pag3'):
        return 'PAG3', 'tab:orange'
    if p.startswith('pag4'):
        return 'PAG4', 'tab:brown'
    if p.startswith('ema'):
        return 'EMA', 'tab:green'
    if p.startswith('dfm'):
        return 'DFM', 'tab:purple'
    if p.startswith('final_layer'):
        return 'FinalLayer', 'tab:pink'
    if p.startswith('coord_att'):
        return 'CoordAtt', 'tab:olive'
    if p.startswith('compression'):
        return 'Compression', 'tab:cyan'
    if p.startswith('conv1'):
        return 'Conv1', 'tab:gray'
    if p.startswith('layer') or p.startswith('relu'):
        return 'Backbone', 'steelblue'
    return 'Other', 'lightgray'


def plot_pareto(points, out_png):
    """画 Pareto 散点图。matplotlib 缺失时跳过。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print('[WARN] matplotlib 未安装，跳过画图，仅导出 CSV。'
              ' 可用 `pip install matplotlib` 后重跑。')
        return

    # 过滤：只画有延迟数据且非 exDataConvert 重复的层
    valid = []
    for p in points:
        if p['time_us'] is None or p['time_us'] <= 0:
            continue
        mod, _ = get_module(p['layer'])
        if mod is None:
            continue
        p['module'] = mod
        valid.append(p)

    front = pareto_front(points)
    front_layers = {p['layer'] for p in front}

    # 按模块分组画散点
    fig, ax = plt.subplots(figsize=(14, 9))
    module_colors = {}
    for p in valid:
        mod = p['module']
        if mod not in module_colors:
            _, color = get_module(p['layer'])
            module_colors[mod] = color

    plotted_modules = set()
    for p in valid:
        mod = p['module']
        color = module_colors[mod]
        is_front = p['layer'] in front_layers
        label = mod if mod not in plotted_modules else None
        ax.scatter(p['time_us'], p['sensitivity'],
                    c=color, s=100 if is_front else 30,
                    alpha=0.85, edgecolors='black' if is_front else 'none',
                    linewidths=0.8, label=label, zorder=3 if is_front else 2)
        plotted_modules.add(mod)

    # 只标注 Top-10 最敏感层。文字贴在点旁边并配短箭头。
    top_n = sorted(valid, key=lambda x: -x['sensitivity'])[:10]
    max_x = max(p['time_us'] for p in valid)

    def short_label(p, ox, oy):
        label = p['layer'].replace('_output_0', '').lstrip('/')
        ax.annotate(label,
                    (p['time_us'], p['sensitivity']),
                    xytext=(ox, oy), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    ha='left' if ox >= 0 else 'right', va='center',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.7,
                                    shrinkA=3, shrinkB=4),
                    bbox=dict(boxstyle='round,pad=0.25', fc='lightyellow',
                              ec='gray', alpha=0.9), zorder=5)

    # 底部密集簇（低 y 且低 x）：单独拎到簇右侧的小竖列，按 y 拉开，短箭头向左
    dense = [p for p in top_n if p['sensitivity'] < 0.13 and p['time_us'] < 320]
    sparse = [p for p in top_n if p not in dense]

    if dense:
        g = sorted(dense, key=lambda x: x['sensitivity'])  # y 升序
        n = len(g)
        y_top, y_bot = 0.115, -0.015
        step = (y_top - y_bot) / (n - 1) if n > 1 else 0.0
        col_x = 340  # 簇右侧，箭头短
        for i, p in enumerate(g):
            ly = y_top - i * step
            label = p['layer'].replace('_output_0', '').lstrip('/')
            ax.annotate(label,
                        (p['time_us'], p['sensitivity']),
                        xytext=(col_x, ly), textcoords='data',
                        fontsize=9, fontweight='bold',
                        ha='left', va='center',
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.7,
                                        shrinkA=3, shrinkB=4),
                        bbox=dict(boxstyle='round,pad=0.25', fc='lightyellow',
                                  ec='gray', alpha=0.9), zorder=5)

    # 其余点：贴旁小偏移 + 短箭头
    offsets = [(7, 8), (7, -12), (-7, 8), (-7, -12),
               (10, 2), (-10, 2), (7, 16), (7, -18)]
    for i, p in enumerate(sparse):
        ox, oy = offsets[i % len(offsets)]
        if p['time_us'] > max_x * 0.85:
            ox = -abs(ox) - 18
        short_label(p, ox, oy)

    # 敏感度阈值参考线
    ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(max_x * 0.98, 0.052, 'sensitivity=0.05',
            fontsize=8, color='gray', ha='right', va='bottom')

    ax.set_xlabel('Layer INT8 latency (us)  —  proxy for FP16 conversion cost',
                  fontsize=13)
    ax.set_ylabel('Quantization error  (1 - cos, simulator single)',
                  fontsize=13)
    ax.set_title('Mixed-Precision Pareto Front: Layer Sensitivity vs Latency Cost',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, title='Module', title_fontsize=11,
              framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    # 收紧 x 轴范围，减少右侧留白
    ax.set_xlim(left=-20, right=max_x * 1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f'--> Pareto plot saved to {out_png}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--error', default=DEFAULT_ERROR,
                    help='error_analysis.txt 路径')
    ap.add_argument('--perf', default=DEFAULT_PERF,
                    help='eval_perf.csv 路径')
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help='输出的 PNG 路径')
    ap.add_argument('--csv', default=DEFAULT_CSV,
                    help='输出的敏感度表 CSV 路径')
    ap.add_argument('--no-plot', action='store_true',
                    help='只导出 CSV，不画图')
    args = ap.parse_args()

    if not os.path.exists(args.error):
        print(f'找不到文件: {args.error}')
        return
    if not os.path.exists(args.perf):
        print(f'找不到文件: {args.perf}')
        return

    error_rows = parse_error_analysis(args.error)
    perf = parse_perf_csv(args.perf)
    points = compute_sensitivity(error_rows, perf)

    matched = sum(1 for p in points if p['time_us'] is not None)
    print(f'--> 解析: {len(error_rows)} 层精度数据, {len(perf)} 层性能数据, '
          f'{matched} 层成功匹配')

    export_csv(points, args.csv)
    if not args.no_plot:
        plot_pareto(points, args.out)

    # 打印 Top-10 最敏感层（论文里写"剥离了这些层"的依据）
    print('\n--> Top-10 最敏感层（建议提 FP16）:')
    for p in sorted(points, key=lambda x: -x['sensitivity'])[:10]:
        cos_str = 'q_inf/q_nan' if p['cos_single'] is None else f'{p["cos_single"]:.5f}'
        t_str = f'{p["time_us"]}us' if p['time_us'] else 'N/A'
        print(f'  {p["layer"]:<55} cos={cos_str:<12} 1-cos={p["sensitivity"]:.5f}  {t_str}')


if __name__ == '__main__':
    main()
