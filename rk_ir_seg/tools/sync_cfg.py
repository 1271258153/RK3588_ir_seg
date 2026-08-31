"""把主 cfg 的 custom_quantize_layers 段同步到其他算法的 cfg。

step1.py 在不同算法目录下会生成全新的 best.quantization.cfg（只有 quantize_parameters，
没有 custom_quantize_layers）。为了让 normal / kl_divergence / mmse 三组混合精度配置一致、
对比公平，需要把主 cfg（output/）里手动编辑的 custom_quantize_layers 段复制到其他算法的 cfg。

本脚本自动完成这件事：从源 cfg 提取 custom_quantize_layers 段，替换/插入到目标 cfg。

用法:
    python tools/sync_cfg.py                          # 默认: output/ -> output_kl_divergence/ + output_mmse/
    python tools/sync_cfg.py --src output/.../best.quantization.cfg \
                             --dst output_kl_divergence/.../best.quantization.cfg \
                                  output_mmse/.../best.quantization.cfg
    python tools/sync_cfg.py --algorithms kl_divergence mmse   # 指定目标算法
"""
import argparse
import os
import re

OUTPUT_DIR = './output'
CFG_NAME = 'best.quantization.cfg'
SUPPORTED_ALGORITHMS = ('normal', 'kl_divergence', 'mmse')

# 顶层 key（行首顶格的 word:），允许行尾空白或 # 注释（RKNN 导出的 cos 分数）
TOP_KEY_RE = re.compile(r'^(\w+):\s*(?:#.*)?$')


def split_sections(text):
    """把 cfg 文本按顶层 key 切成 {key: [lines]}，保留顺序。"""
    sections = []
    current_key = None
    current_lines = []
    for line in text.splitlines():
        m = TOP_KEY_RE.match(line)
        if m:
            if current_key is not None:
                sections.append((current_key, current_lines))
            current_key = m.group(1)
            current_lines = [line]
        else:
            if current_key is None:
                # 文件开头的空行/注释
                sections.append((None, [line])) if line.strip() else None
            else:
                current_lines.append(line)
    if current_key is not None:
        sections.append((current_key, current_lines))
    return sections


def get_section(sections, key):
    """取指定 key 的段（含 key 行本身）。"""
    for k, lines in sections:
        if k == key:
            return lines
    return None


def extract_custom_block(src_path):
    """从源 cfg 提取 custom_quantize_layers 段的完整文本（含 key 行）。"""
    with open(src_path, encoding='utf-8') as f:
        text = f.read()
    sections = split_sections(text)
    block = get_section(sections, 'custom_quantize_layers')
    if block is None:
        raise RuntimeError(f'源 cfg 中找不到 custom_quantize_layers 段: {src_path}')
    return '\n'.join(block)


def merge_into_dst(dst_path, custom_block):
    """把 custom_quantize_layers 段合并进目标 cfg：
    - 若目标已有该段，整段替换；
    - 若没有，插到文件开头（custom_quantize_layers 应在 quantize_parameters 之前）。
    保留目标 cfg 自身的 quantize_parameters 段不动。
    """
    with open(dst_path, encoding='utf-8') as f:
        text = f.read()
    sections = split_sections(text)

    # 重建：把 custom_quantize_layers 段放在最前，其余段保持原顺序跳过旧的 custom 段
    out = [custom_block]
    for k, lines in sections:
        if k == 'custom_quantize_layers':
            continue  # 跳过旧的，用源 cfg 的替换
        if k is None:
            # 顶部空行/注释：保留非空注释，跳过纯空行（避免开头多空行）
            joined = '\n'.join(lines).strip()
            if joined:
                out.append(joined)
            continue
        out.append('\n'.join(lines))

    merged = '\n'.join(out) + '\n'
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(merged)


def cfg_path_for(algo):
    """根据算法名拼出 cfg 路径，与 step1/step2 目录约定一致。"""
    suffix = "" if algo == 'normal' else f"_{algo}"
    return os.path.join(OUTPUT_DIR + suffix, 'hybrid_quantization', CFG_NAME)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--src', default=os.path.join(OUTPUT_DIR, 'hybrid_quantization', CFG_NAME),
                    help='源 cfg 路径（默认 output/hybrid_quantization/best.quantization.cfg）')
    ap.add_argument('--dst', nargs='+', default=None,
                    help='目标 cfg 路径列表（默认由 --algorithms 推导）')
    ap.add_argument('--algorithms', nargs='+',
                    default=['kl_divergence', 'mmse'],
                    choices=SUPPORTED_ALGORITHMS,
                    help='目标算法名（默认 kl_divergence mmse），自动推导 cfg 路径')
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f'[ERROR] 源 cfg 不存在: {args.src}')
        return

    custom_block = extract_custom_block(args.src)
    n_layers = sum(1 for ln in custom_block.splitlines()
                   if ln.strip() and not ln.strip().startswith('#')
                   and not ln.strip().endswith(':'))
    print(f'--> 从源 cfg 提取 custom_quantize_layers: {n_layers} 层 float16')
    print('    ' + '\n    '.join(l.strip() for l in custom_block.splitlines()
          if l.strip() and not l.strip().startswith('#')))

    if args.dst:
        targets = args.dst
    else:
        targets = [cfg_path_for(a) for a in args.algorithms]

    for dst in targets:
        if not os.path.exists(dst):
            print(f'[WARN] 目标 cfg 不存在，跳过（请先跑对应 step1.py）: {dst}')
            continue
        merge_into_dst(dst, custom_block)
        print(f'--> 已同步 -> {dst}')

    print('\n完成。现在三组算法的 custom_quantize_layers 配置一致，可分别跑 step2.py 对比。')


if __name__ == '__main__':
    main()
