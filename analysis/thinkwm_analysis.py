"""
thinkWM 可复用数据分析模块

使用方式:
    from thinkwm_analysis import load_subject, summarize
    df = load_subject('006')
    stats = summarize(df)

    from thinkwm_analysis import diff_analysis
    result = diff_analysis(df, ['006', '007', '008'])
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = Path('/home/Tobf/Project/attentionWM/data')
SUBJECTS = ['001', '006', '007', '008', '009', '010', '011', '012']

# ============================================================
# 加载
# ============================================================

def load_subject(sid, data_root=DATA_ROOT):
    """加载单个被试的全部 session，返回合并 DataFrame。"""
    subj_dir = data_root / sid
    if not subj_dir.is_dir():
        raise FileNotFoundError(f"Subject directory not found: {subj_dir}")

    csv_files = sorted(glob(str(subj_dir / '*_explog.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for subject {sid}")

    sessions = []
    for fpath in csv_files:
        # 跳过空文件/极小文件
        if Path(fpath).stat().st_size < 100:
            continue
        df = pd.read_csv(fpath, dtype=str, low_memory=False)
        # 清理可能的 BOM
        df.columns = [c.replace('﻿', '') for c in df.columns]
        # 跳过旧格式文件 (无 Trial_Type 列)
        if 'Trial_Type' not in df.columns or len(df) == 0:
            continue
        # 剔除行内嵌入的 header 行
        df = df[df['Trial_Type'] != 'Trial_Type'].copy()
        # 剔除 QUIT 行
        df = df[df['Block'] != 'QUIT'].copy()
        sessions.append(df)

    df = pd.concat(sessions, ignore_index=True)
    return _coerce_types(df)


def _coerce_types(df):
    """转换列类型。"""
    numeric_cols = [
        'Math_Acc', 'Math_RT', 'WM_Score', 'Trial', 'Block',
        'Z_RT',
        'RT_Mean5', 'RT_SD5', 'RT_CV5', 'ACC_Mean10',
        'RT_Micro', 'RT_Meso', 'RT_Macro',
        'Operand_A', 'Operand_B',
    ]
    # 新增序列列 (如果有)
    for col in ['Prev_Math_Acc', 'Prev_RT', 'Trials_Since_Probe']:
        if col in df.columns:
            numeric_cols.append(col)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Is_Probe -> bool
    if 'Is_Probe' in df.columns:
        col = df['Is_Probe']
        if col.dtype == 'object':
            df['Is_Probe'] = col.str.lower().isin(['true', '1', 't'])
        elif col.dtype == 'float64':
            df['Is_Probe'] = col == 1.0

    return df


# ============================================================
# 难度分层
# ============================================================

def _extract_operands(df):
    """从 Operand 列或 Equation 字符串提取 a, b。

    支持格式:
      a×b     (v0.3+, 新格式)
      a×b+c   (v0.2, pilot)
      a+b     (旧格式, 006-009)

    返回 a, b, c 三个 Series (c 向后兼容用途，新格式均为 0)。
    """
    if 'Operand_A' in df.columns and df['Operand_A'].notna().any():
        a = pd.to_numeric(df['Operand_A'], errors='coerce')
        b = pd.to_numeric(df['Operand_B'], errors='coerce') if 'Operand_B' in df.columns else pd.Series(np.nan, index=df.index)
        # 旧 CSV 可能有 Operand_C，新格式没有
        if 'Operand_C' in df.columns:
            c = pd.to_numeric(df['Operand_C'], errors='coerce')
        else:
            c = pd.Series(0, index=df.index)
        if b.notna().any():
            return a, b, c

    # 回退: 从 Equation 解析
    import re
    a_vals, b_vals, c_vals = [], [], []

    for _, row in df.iterrows():
        eq = str(row.get('Equation', ''))
        if not eq or eq == 'nan':
            a_vals.append(np.nan); b_vals.append(np.nan); c_vals.append(np.nan)
            continue

        has_mul = '×' in eq
        has_add = '＋' in eq or '+' in eq
        nums = re.findall(r'\d+', eq)

        if has_mul and has_add and len(nums) >= 4:
            a_vals.append(int(nums[0])); b_vals.append(int(nums[1]))
            c_vals.append(int(nums[2]))
        elif has_mul and len(nums) >= 3:
            a_vals.append(int(nums[0])); b_vals.append(int(nums[1]))
            c_vals.append(0)
        elif len(nums) >= 3:
            a_vals.append(int(nums[0])); b_vals.append(int(nums[1]))
            c_vals.append(np.nan)
        else:
            a_vals.append(np.nan); b_vals.append(np.nan); c_vals.append(np.nan)

    return (pd.Series(a_vals, index=df.index),
            pd.Series(b_vals, index=df.index),
            pd.Series(c_vals, index=df.index))


def add_difficulty(df):
    """基于文献的公式化难度评分，三级离散化。

    a×b 格式 (v0.3+, c=0):
      difficulty_score = (a * b) / 81
      问题规模效应 (Ashcraft 1992) 为唯一来源。
      去除了 c 项以消除 LTM检索/程序性计算的混合。

    a×b+c 格式 (010, v0.2):
      difficulty_score = 0.7 × (a×b) / 81 + 0.3 × (c / 18)

    a+b 格式 (006-009, 旧格式):
      difficulty_score = (a+b) / 99

    离散化: 按被试内 difficulty_score 的 33%/67% 分位点切 easy/medium/hard。
    """
    a, b, c = _extract_operands(df)
    has_c = c.notna().any()
    all_c_zero = has_c and (c == 0).all()

    if all_c_zero:
        # a×b only 格式: 纯问题规模效应
        difficulty_score = (a * b) / 81.0
    elif has_c:
        difficulty_score = 0.7 * (a * b) / 81.0 + 0.3 * c / 18.0
    else:
        difficulty_score = (a + b) / 99.0

    df['Difficulty_Score'] = difficulty_score

    valid_idx = difficulty_score.dropna().index
    difficulty = pd.Series('unknown', index=df.index)

    if len(valid_idx) == 0:
        df['Difficulty'] = difficulty
        return df

    lo = difficulty_score.quantile(0.33)
    hi = difficulty_score.quantile(0.67)

    for idx in valid_idx:
        s = difficulty_score[idx]
        if s <= lo:
            difficulty[idx] = 'easy'
        elif s <= hi:
            difficulty[idx] = 'medium'
        else:
            difficulty[idx] = 'hard'

    df['Difficulty'] = difficulty
    return df


# ============================================================
# 核心统计
# ============================================================

def summarize(df):
    """返回一句话级别的全局摘要 dict。"""
    math = df[df['Trial_Type'] == 'math']
    probes = df[df['Is_Probe'] == True]

    return {
        'n_trials': len(df),
        'n_math': len(math),
        'n_probes': len(probes),
        'math_acc': math['Math_Acc'].mean(),
        'math_rt_mean': math['Math_RT'].mean(),
        'math_rt_sd': math['Math_RT'].std(),
        'wm_mean': probes['WM_Score'].mean(),
        'wm_sd': probes['WM_Score'].std(),
    }


def wm_by_state(df):
    """按 State_Label 分组统计 WM（仅探测行）。"""
    probes = df[df['Is_Probe'] == True].copy()
    state_stats = probes.groupby('State_Label')['WM_Score'].agg(
        n='count', mean='mean', sd='std'
    ).sort_values('mean', ascending=False)

    # 投入态 vs 脱离态汇总
    engaged_labels = {'optimal', 'neutral'}
    disengaged_labels = {'cautious', 'lapse', 'acc_decline', 'impulsive'}

    engaged = probes[probes['State_Label'].isin(engaged_labels)]['WM_Score']
    disengaged = probes[probes['State_Label'].isin(disengaged_labels)]['WM_Score']

    pattern = None
    if len(engaged) > 0 and len(disengaged) > 0:
        diff = disengaged.mean() - engaged.mean()
        pattern = 'reverse' if diff > 0 else 'standard'

    return {
        'by_state': state_stats,
        'engaged_mean': engaged.mean() if len(engaged) else float('nan'),
        'disengaged_mean': disengaged.mean() if len(disengaged) else float('nan'),
        'delta': (disengaged.mean() - engaged.mean()) if (len(engaged) > 0 and len(disengaged) > 0) else float('nan'),
        'pattern': pattern,
    }


def rt_wm_correlation(df):
    """|Z_RT| 与 WM 的 Pearson 相关（仅探测行）。"""
    probes = df[df['Is_Probe'] == True].copy()
    probes = probes.dropna(subset=['Z_RT', 'WM_Score'])
    if len(probes) < 3:
        return {'r': float('nan'), 'p': float('nan'), 'n': len(probes)}
    r, p = sp_stats.pearsonr(probes['Z_RT'].abs(), probes['WM_Score'])
    return {'r': r, 'p': p, 'n': len(probes)}


def math_by_state(df):
    """按 State_Label 统计数学表现（仅数学行）。"""
    math = df[df['Trial_Type'] == 'math']
    return math.groupby('State_Label').agg(
        n=('Math_Acc', 'count'),
        math_acc=('Math_Acc', 'mean'),
        math_rt_mean=('Math_RT', 'mean'),
        math_rt_sd=('Math_RT', 'std'),
        abs_z_rt=('Z_RT', lambda x: x.abs().mean()),
    ).sort_values('math_rt_mean', ascending=False)


# ============================================================
# 难度分层分析
# ============================================================

def diff_analysis(df, subject_ids=None):
    """按难度分层的主分析：WM × State × Difficulty。

    返回：
    - table: 难度 × 状态 的 WM 交叉表
    - math_stats: 各难度的数学表现
    - interaction: 难度与投入/脱离的交互检验
    """
    df = _propagate_difficulty(df)
    probes = df[df['Is_Probe'] == True].dropna(subset=['Difficulty', 'WM_Score'])
    probes = probes[probes['Difficulty'] != 'unknown']

    # 难度 × 状态 WM 交叉表
    engaged_labels = {'optimal', 'neutral'}
    disengaged_labels = {'cautious', 'lapse', 'acc_decline', 'impulsive'}

    def _group_label(s):
        if s in engaged_labels:
            return 'engaged'
        elif s in disengaged_labels:
            return 'disengaged'
        return 'other'

    probes = probes.copy()
    probes['Group'] = probes['State_Label'].apply(_group_label)

    # WM by Difficulty × Group
    table = probes.groupby(['Difficulty', 'Group'])['WM_Score'].agg(
        n='count', mean='mean', sd='std'
    ).round(2)

    # 各难度下的投入/脱离 Δ
    deltas = {}
    for diff in ['easy', 'medium', 'hard']:
        diff_data = probes[probes['Difficulty'] == diff]
        eng = diff_data[diff_data['Group'] == 'engaged']['WM_Score']
        dis = diff_data[diff_data['Group'] == 'disengaged']['WM_Score']
        if len(eng) > 0 and len(dis) > 0:
            deltas[diff] = {
                'engaged_n': len(eng), 'engaged_wm': eng.mean(),
                'disengaged_n': len(dis), 'disengaged_wm': dis.mean(),
                'delta': dis.mean() - eng.mean(),
                'p': sp_stats.mannwhitneyu(eng, dis, alternative='two-sided')[1]
                if len(eng) >= 2 and len(dis) >= 2 else float('nan'),
            }

    # 数学表现按难度
    math = df[df['Trial_Type'] == 'math'].dropna(subset=['Difficulty'])
    math = math[math['Difficulty'] != 'unknown']
    math_stats = math.groupby('Difficulty').agg(
        n=('Math_Acc', 'count'),
        math_acc=('Math_Acc', 'mean'),
        math_rt_mean=('Math_RT', 'mean'),
        math_rt_sd=('Math_RT', 'std'),
    ).round(3)

    return {
        'table': table,
        'deltas': deltas,
        'math_stats': math_stats,
    }


# ============================================================
# 跨被试汇总
# ============================================================

def cross_subject_summary(subject_ids=SUBJECTS):
    """汇总所有被试的 WM-by-State 模式。"""
    rows = []
    for sid in subject_ids:
        try:
            df = load_subject(sid)
            result = wm_by_state(df)
            rows.append({
                'subject': sid,
                'n_probes': result['by_state']['n'].sum(),
                'engaged_wm': result['engaged_mean'],
                'disengaged_wm': result['disengaged_mean'],
                'delta': result['delta'],
                'pattern': result['pattern'],
            })
        except (FileNotFoundError, KeyError) as e:
            rows.append({'subject': sid, 'n_probes': 0, 'engaged_wm': None,
                         'disengaged_wm': None, 'delta': None, 'pattern': 'error'})
    return pd.DataFrame(rows)


def cross_subject_difficulty(subject_ids=SUBJECTS):
    """所有被试的难度分层分析汇总。

    探测试次从紧前数学试次继承难度分类。
    """
    all_probes = []
    for sid in subject_ids:
        try:
            df = load_subject(sid)
            df = _propagate_difficulty(df)
            probes = df[df['Is_Probe'] == True].dropna(subset=['Difficulty', 'WM_Score'])
            probes = probes[probes['Difficulty'] != 'unknown'].copy()
            probes['subject'] = sid
            all_probes.append(probes)
        except FileNotFoundError:
            continue

    if not all_probes:
        return None

    combined = pd.concat(all_probes, ignore_index=True)

    engaged_labels = {'optimal', 'neutral'}
    disengaged_labels = {'cautious', 'lapse', 'acc_decline', 'impulsive'}

    def _group_label(s):
        if s in engaged_labels:
            return 'engaged'
        elif s in disengaged_labels:
            return 'disengaged'
        return 'other'

    combined['Group'] = combined['State_Label'].apply(_group_label)

    table = combined.groupby(['Difficulty', 'Group'])['WM_Score'].agg(
        n='count', mean='mean', sd='std'
    ).round(2)

    # 统计检验: 每难度下投入 vs 脱离
    tests = {}
    for diff in ['easy', 'medium', 'hard']:
        diff_data = combined[combined['Difficulty'] == diff]
        eng = diff_data[diff_data['Group'] == 'engaged']['WM_Score']
        dis = diff_data[diff_data['Group'] == 'disengaged']['WM_Score']
        if len(eng) >= 2 and len(dis) >= 2:
            u, p = sp_stats.mannwhitneyu(dis, eng, alternative='greater')
            tests[diff] = {'U': u, 'p': p, 'dis_gt_eng': dis.mean() > eng.mean()}

    return {'table': table, 'tests': tests, 'n_total': len(combined)}


def _propagate_difficulty(df):
    """将难度从数学试次传播到紧接的探测试次。

    策略: 先对所有行尝试提取操作数，然后对探测行用前一行的难度填充。
    """
    df = add_difficulty(df)

    # 对于探测行 (Difficulty == 'unknown' 且 Trial_Type != 'math')，
    # 继承前一行的 Difficulty
    for i in range(1, len(df)):
        if df.loc[df.index[i], 'Difficulty'] == 'unknown':
            df.loc[df.index[i], 'Difficulty'] = df.loc[df.index[i-1], 'Difficulty']

    return df


# ============================================================
# Post-hoc 序列效应
# ============================================================

def post_error_analysis(df):
    """分析 Post-error 对 WM 的影响。

    要求 CSV 包含 Prev_Math_Acc 列（v2.1+）。
    如果列不存在，回退到手动计算。
    """
    if 'Prev_Math_Acc' not in df.columns or df['Prev_Math_Acc'].isna().all():
        # 回退：手动从上一行提取
        return _post_error_fallback(df)

    probes = df[df['Is_Probe'] == True].dropna(subset=['Prev_Math_Acc', 'WM_Score']).copy()
    return _post_error_compute(probes)


def _post_error_compute(probes):
    """计算 post-error vs post-correct 的 WM 对比。"""
    post_err = probes[probes['Prev_Math_Acc'] == 0]['WM_Score']
    post_corr = probes[probes['Prev_Math_Acc'] == 1]['WM_Score']

    if len(post_err) >= 2 and len(post_corr) >= 2:
        u, p = sp_stats.mannwhitneyu(post_err, post_corr, alternative='two-sided')
    else:
        u, p = float('nan'), float('nan')

    return {
        'post_error_n': len(post_err),
        'post_error_wm': post_err.mean() if len(post_err) > 0 else float('nan'),
        'post_correct_n': len(post_corr),
        'post_correct_wm': post_corr.mean() if len(post_corr) > 0 else float('nan'),
        'delta': (post_err.mean() - post_corr.mean()) if (len(post_err) and len(post_corr)) else float('nan'),
        'mw_u': u, 'mw_p': p,
    }


def _post_error_fallback(df):
    """无 Prev_Math_Acc 列时的手动回退计算。"""
    probes = []
    math_rows = df[df['Trial_Type'] == 'math'].copy()
    for idx in math_rows.index:
        if idx + 1 in df.index:
            next_row = df.loc[idx + 1]
            if next_row.get('Is_Probe') == True:
                probes.append({
                    'prev_acc': math_rows.loc[idx, 'Math_Acc'],
                    'wm': next_row['WM_Score'],
                })
    if not probes:
        return {'post_error_n': 0, 'error': 'no probe trials found'}
    return _post_error_compute(pd.DataFrame(probes).rename(columns={'prev_acc': 'Prev_Math_Acc', 'wm': 'WM_Score'}))


# ============================================================
# 主入口
# ============================================================

def _report_header(title):
    print(); print('=' * 72); print(f'  {title}'); print('=' * 72)

def _print_subject_report(sid):
    """输出单个被试的完整分析报告。"""
    try:
        df = load_subject(sid)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"  [{sid}] 无有效数据"); return

    s = summarize(df)
    math = df[df['Trial_Type'] == 'math']
    probes = df[df['Is_Probe'] == True]

    print(f"\n  被试 {sid}")
    print(f"  {'─' * 50}")
    print(f"  总试次: {s['n_trials']} (数学: {s['n_math']}, 探测: {s['n_probes']})")
    print(f"  数学正确率: {s['math_acc']:.1%}")
    print(f"  数学 RT: {s['math_rt_mean']:.3f}s (SD={s['math_rt_sd']:.3f}s)")
    print(f"  WM: {s['wm_mean']:.2f}/6 (SD={s['wm_sd']:.2f})")

    # 各 block 探测数
    if 'Block' in probes.columns:
        pb = probes.groupby('Block').size()
        print(f"  各 block 探测数: {dict(pb)}")

    # WM by state
    ws = wm_by_state(df)
    print(f"\n  WM 按状态:")
    for label, row in ws['by_state'].iterrows():
        print(f"    {label:>12s}: n={int(row['n'])}, WM={row['mean']:.2f}+/-{row['sd']:.2f}")
    print(f"  投入态 (optimal): WM={ws['engaged_mean']:.2f}")
    print(f"  脱离态 (cautious+lapse+acc_decline): WM={ws['disengaged_mean']:.2f}")
    print(f"  Δ (脱离-投入): {ws['delta']:+.2f}  模式: {ws['pattern']}")

    # Post-error
    pe = post_error_analysis(df)
    if 'error' not in pe:
        print(f"\n  Post-error WM: {pe['post_error_wm']:.2f} (n={pe['post_error_n']})")
        print(f"  Post-correct WM: {pe['post_correct_wm']:.2f} (n={pe['post_correct_n']})")
        if not np.isnan(pe['mw_p']):
            print(f"  Δ={pe['delta']:+.3f}, p={pe['mw_p']:.3f}")

    # 难度分析
    df_diff = _propagate_difficulty(df.copy())
    probes_diff = df_diff[df_diff['Is_Probe'] == True].dropna(subset=['Difficulty', 'WM_Score'])
    probes_diff = probes_diff[probes_diff['Difficulty'] != 'unknown']
    if len(probes_diff) > 0:
        print(f"\n  WM × 难度 (a×b/81 三分位):")
        for d in ['easy', 'medium', 'hard']:
            dd = probes_diff[probes_diff['Difficulty'] == d]
            if len(dd) > 0:
                eng = dd[dd['State_Label'].isin({'optimal'})]['WM_Score']
                dis = dd[dd['State_Label'].isin({'cautious', 'lapse', 'acc_decline'})]['WM_Score']
                eng_wm = f"{eng.mean():.2f}" if len(eng) > 0 else "--"
                dis_wm = f"{dis.mean():.2f}" if len(dis) > 0 else "--"
                delta = f"{dis.mean() - eng.mean():+.2f}" if len(eng) > 0 and len(dis) > 0 else "--"
                n_str = f"n={len(dd)}"
                print(f"    {d:>8s}: {n_str}  投入={eng_wm}  脱离={dis_wm}  Δ={delta}")


if __name__ == '__main__':
    import sys

    sids = sys.argv[1:] if len(sys.argv) > 1 else SUBJECTS
    valid_sids = [s for s in sids]
    print(f"\n  thinkWM 数据分析报告")
    print(f"  {'═' * 50}")

    # Per-subject report
    _report_header("逐被试分析")
    for sid in valid_sids:
        try:
            _print_subject_report(sid)
        except Exception as e:
            print(f"  [{sid}] 加载失败: {e}")

    # Cross-subject summary (only for valid a×b format subjects)
    _report_header("跨被试汇总 (投入 vs 脱离)")
    cs = cross_subject_summary(valid_sids)
    if cs is not None and len(cs) > 0:
        print(f"\n{cs.to_string(index=False)}")
        avg_delta = cs['delta'].mean()
        print(f"\n  平均 Δ = {avg_delta:+.2f}")
        rev_count = (cs['pattern'] == 'reverse').sum()
        all_count = cs['pattern'].notna().sum()
        if all_count > 0:
            print(f"  反向模式: {rev_count}/{all_count} 被试")

    # Difficulty × State
    _report_header("难度 × 状态分析 (合并)")
    result = cross_subject_difficulty(valid_sids)
    if result:
        print(f"\n{result['table'].to_string()}")
        print()
        for diff, t in result['tests'].items():
            p_str = f"p={t['p']:.4f}"
            print(f"  {diff}: U={t['U']:.1f}, {p_str}, 脱离>投入={t['dis_gt_eng']}")

    # Post-error pooled
    _report_header("Post-error 汇总")
    all_pe = {'post_error_n': 0, 'post_correct_n': 0, 'post_error_wm': 0, 'post_correct_wm': 0}
    valid_pe = 0
    for sid in valid_sids:
        try:
            df = load_subject(sid)
            pe = post_error_analysis(df)
            if 'error' not in pe and pe['post_error_n'] >= 1:
                valid_pe += 1
                for k in ['post_error_n', 'post_correct_n']:
                    all_pe[k] += pe[k]
                all_pe['post_error_wm'] += pe['post_error_wm'] * pe['post_error_n']
                all_pe['post_correct_wm'] += pe['post_correct_wm'] * pe['post_correct_n']
        except Exception:
            pass
    if valid_pe > 0 and all_pe['post_error_n'] > 0:
        all_pe['post_error_wm'] /= all_pe['post_error_n']
        all_pe['post_correct_wm'] /= all_pe['post_correct_n']
        print(f"\n  Post-error WM: {all_pe['post_error_wm']:.2f} (n={all_pe['post_error_n']})")
        print(f"  Post-correct WM: {all_pe['post_correct_wm']:.2f} (n={all_pe['post_correct_n']})")
    else:
        print(f"\n  无有效 post-error 数据")

    print(); print('=' * 72); print("  报告结束"); print('=' * 72)
