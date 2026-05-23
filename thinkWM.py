# -*- coding: utf-8 -*-
"""
思维速度与色块记忆实验
1. 6个色块作为背景，中央叠加数学题
2. 数据存储至 ./data/被试编号/
"""

import os, time, csv, random, sys
import urllib.request
import numpy as np
from datetime import datetime
from psychopy import visual, core, event, tools, gui

# ==============================================================
#                     全局参数配置区 (方便微调)
# ==============================================================
SIZE_SQUARE = 0.058            # 色块大小 (height 单位)
POS_RADIUS = 0.13              # 6个色块排列的半径
SIZE_MATH = 0.048              # 中央数学题字体大小
SIZE_GRID = SIZE_SQUARE / 3    # 九宫格中每个色块的大小
DUR_MATH_TIMEOUT = 3.0         # 等式判断最长时限
DUR_ITI = 1.0                  # 试次间隔
DUR_ENCODE = 1.0               # 颜色编码时间
DUR_FEEDBACK = 0.3             # 反馈时长
DUR_PROBE_BLANK = 1.5          # 探测后空屏缓冲
DUR_COUNTDOWN = 1.0            # 倒计时等待
REST_MIN_SEC = 15              # block间最短休息(秒)
BLOCK_Y_OFFSET = -0.02         # 色块整体下移量
MATH_Y = 0.042                  # 数学题垂直位置 (居中于色块行间空隙)

# 颜色定义 (标准9色)
COLORS_RGB = [
    [1, -1, -1], [-1, 1, -1], [-1, -1, 1], 
    [1, 1, -1], [1, -1, 1], [-1, 1, 1], 
    [1, 1, 1], [-1, -1, -1], [1, 0.3, -1]
]
# ==============================================================


class StateMonitor:
    """基于 RT z-score + 双速 ACC EWMA 的在线认知状态量化。

    累计均值标准化反应时偏移，双速指数滑动平均检测正确率下降趋势。
    连续探针分数 = |Z_RT| × (1 + ACC_decline_惩罚因子)。
    """
    def __init__(self, baseline_trials=30, cooldown=5, max_probes_per_block=10,
                 score_threshold=1.1):
        self.baseline_trials = baseline_trials
        self.cooldown = cooldown
        self.max_probes = max_probes_per_block
        self.score_threshold = score_threshold
        self.reset()

    def reset(self):
        self.cum_n = 0
        self.cum_sum_rt = 0.0
        self.cum_sum_sq_rt = 0.0
        self.cum_mean = 0.0
        self.cum_sd = 1.0
        self.cum_cv5 = []

        # RT 三速 EWMA
        self.ewma_micro = None   # α=0.30
        self.ewma_meso = None    # α=0.10
        self.ewma_macro = None   # α=0.02, ~100 试次基线

        # ACC 双速 EWMA
        self.acc_fast = None     # α=0.30, ~6-trial window
        self.acc_slow = None     # α=0.02, ~100 试次基线, ~100-trial personal baseline
        self.acc_slow_peak = 0.0

        # 状态标签
        self.state_label = 'baseline'
        self.probe_score = 0.0
        self.z_rt = float('nan')
        self.z_cv = float('nan')

        self.reset_block()

    def reset_block(self):
        self.block_probes = 0
        self.since_last_probe = 0  # 休息后需冷却期到期才能探测

    def update(self, rt, acc, rt_cv5):
        """每个数学试次后调用。返回包含当前状态值的字典。"""
        self.cum_n += 1
        self.cum_sum_rt += rt
        self.cum_sum_sq_rt += rt * rt
        self.cum_mean = self.cum_sum_rt / self.cum_n
        cum_var = max(0, self.cum_sum_sq_rt / self.cum_n - self.cum_mean ** 2)
        self.cum_sd = np.sqrt(cum_var)
        self.cum_cv5.append(rt_cv5)

        # RT EWMA
        for name, alpha in [('micro', 0.30), ('meso', 0.10), ('macro', 0.02)]:
            attr = f'ewma_{name}'
            old = getattr(self, attr)
            setattr(self, attr, alpha * rt + (1 - alpha) * old if old is not None else rt)

        # ACC EWMA
        acc_f = float(acc)
        if self.acc_fast is None:
            self.acc_fast = acc_f
            self.acc_slow = acc_f
        else:
            self.acc_fast = 0.30 * acc_f + 0.70 * self.acc_fast
            self.acc_slow = 0.02 * acc_f + 0.98 * self.acc_slow
        if self.acc_slow > self.acc_slow_peak:
            self.acc_slow_peak = self.acc_slow

        # Z_RT
        if self.cum_n > self.baseline_trials and self.cum_sd > 0:
            self.z_rt = (rt - self.cum_mean) / self.cum_sd
        else:
            self.z_rt = float('nan')

        # Z_CV
        if self.cum_n > self.baseline_trials and len(self.cum_cv5) >= 2:
            cv_mean = np.mean(self.cum_cv5)
            cv_std = np.std(self.cum_cv5, ddof=0)
            self.z_cv = ((rt_cv5 - cv_mean) / cv_std) if cv_std > 0 else float('nan')
        else:
            self.z_cv = float('nan')

        # ACC decline: current fast EWMA vs historical slow peak
        # (avoids false decline after block transitions, where acc_fast < acc_slow
        #  simply because slow EWMA lags behind recovery)
        acc_decline = max(0, self.acc_slow_peak - self.acc_fast)

        # Continuous probe score
        if np.isnan(self.z_rt):
            self.probe_score = 0.0
            self.state_label = 'baseline'
        else:
            acc_penalty = 1.0 + min(acc_decline / 0.10, 2.0)
            self.probe_score = abs(self.z_rt) * acc_penalty

            if self.z_rt > 1.1:
                self.state_label = 'lapse' if acc_decline > 0.03 else 'cautious'
            elif self.z_rt < -1.1:
                self.state_label = 'impulsive' if acc == 0 else 'optimal'
            elif acc_decline > 0.12:
                self.state_label = 'acc_decline'
            else:
                self.state_label = 'neutral'

        self.since_last_probe += 1
        return {
            'z_rt': self.z_rt, 'z_cv': self.z_cv,
            'state_label': self.state_label, 'probe_score': self.probe_score,
        }

    def should_probe(self):
        if self.block_probes >= self.max_probes:
            return False
        if self.since_last_probe < 1:  # 禁止连续探测
            return False
        if self.probe_score < self.score_threshold:
            return False
        # 紧急超驰：极端分数跳过冷却期 (保留最小试次间隔)
        if self.probe_score >= 4.0:
            return True
        return self.since_last_probe >= self.cooldown

    def mark_probe(self):
        self.since_last_probe = 0
        self.block_probes += 1


class ThinkWMConfig:
    def __init__(self, gui_specs):
        self.subj_name = str(gui_specs['Participant'])
        self.student_id = str(gui_specs.get('StudentID', self.subj_name))
        self.student_name = str(gui_specs.get('StudentName', ''))
        self.debug = gui_specs['Debug']
        self.seed = int(time.time())
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        np.random.seed(self.seed)

        # 设定保存路径 ./data/被试编号/
        self.base_dir = os.path.dirname(__file__)
        self.save_dir = os.path.join(self.base_dir, 'data', self.subj_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.nblocks = 1 if self.debug else 4
        self.ntrials_perblock = 999 if self.debug else 80
        
        # 预分配存储矩阵
        self.acc_math = np.zeros((self.nblocks, self.ntrials_perblock))
        self.rt_math = np.zeros((self.nblocks, self.ntrials_perblock))
        self.wm_scores = np.zeros((self.nblocks, self.ntrials_perblock))
        self.probe_occurred = np.zeros((self.nblocks, self.ntrials_perblock), dtype=int)
        
        # 计算 6 个色块的坐标
        self.pos_x = []
        self.pos_y = []
        for i in range(6):
            # 环形分布
            angle = i * (360 / 6)
            x, y = tools.coordinatetools.pol2cart(angle, POS_RADIUS)
            self.pos_x.append(x)
            self.pos_y.append(y + BLOCK_Y_OFFSET)

        # 保存可读的实验设计文件
        self._save_design()

    def _save_design(self):
        """保存实验设计参数为可读文本文件"""
        lines = [
            "# Thinking WM Experiment Design",
            f"Subject: {self.subj_name}",
            f"Student Name: {self.student_name}",
            f"Student ID: {self.student_id}",
            f"Debug: {self.debug}",
            f"Timestamp: {self.timestamp}",
            f"Seed: {self.seed}",
            f"Blocks: {self.nblocks}",
            f"Trials per block: {self.ntrials_perblock}",
            f"POS_RADIUS: {POS_RADIUS}",
            f"SIZE_SQUARE: {SIZE_SQUARE}",
            f"SIZE_MATH: {SIZE_MATH}",
            f"DUR_MATH_TIMEOUT: {DUR_MATH_TIMEOUT}",
            f"DUR_ITI: {DUR_ITI}",
            f"BLOCK_Y_OFFSET: {BLOCK_Y_OFFSET}",
            f"MATH_Y: {MATH_Y}",
            f"Colors: {COLORS_RGB}",
        ]
        path = os.path.join(self.save_dir, f"{self.subj_name}_design.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

class ThinkWMDisplay:
    def __init__(self, dsgn):
        self.win = visual.Window(
            size=(1280, 800) if dsgn.debug else (),  # debug 窗口化, fullscr 自动检测分辨率
            fullscr=not dsgn.debug,
            units='height',
            color=(0, 0, 0),
            checkTiming=False
        )
        
        # 背景色块
        self.bg_squares = [visual.Rect(self.win, width=SIZE_SQUARE, height=SIZE_SQUARE, lineColor=None) for _ in range(6)]
        for i in range(6):
            self.bg_squares[i].pos = (dsgn.pos_x[i], dsgn.pos_y[i])
            
        # 数学题 (位置抬高至色块一二行空隙)
        self.math_text = visual.TextStim(self.win, text='', height=SIZE_MATH, color='white', bold=True, pos=(0, MATH_Y))
        # 反馈框 (绿对/红错)
        self.feedback_frame = visual.Rect(self.win, width=0.5, height=SIZE_MATH * 2.0,
                                          lineColor='green', lineWidth=4, fillColor=None, pos=(0, MATH_Y))
        
        # 九宫格探测器
        self.grid_rects = []
        # 九宫格相对每个中心点的偏移量
        offsets = [(-SIZE_GRID, SIZE_GRID), (0, SIZE_GRID), (SIZE_GRID, SIZE_GRID),
                   (-SIZE_GRID, 0), (0, 0), (SIZE_GRID, 0),
                   (-SIZE_GRID, -SIZE_GRID), (0, -SIZE_GRID), (SIZE_GRID, -SIZE_GRID)]
        for i in range(6):
            pos_grids = []
            for j in range(9):
                r = visual.Rect(self.win, width=SIZE_GRID, height=SIZE_GRID, 
                                fillColor=COLORS_RGB[j], lineColor=None,
                                pos=(dsgn.pos_x[i]+offsets[j][0], dsgn.pos_y[i]+offsets[j][1]))
                pos_grids.append(r)
            self.grid_rects.append(pos_grids)

        # 注视点 (色环中央, 探测试次变黄信号任务切换)
        self.fixation = visual.TextStim(self.win, text='+', height=0.03, color='white', bold=True,
                                        pos=(0, BLOCK_Y_OFFSET))

        self.mouse = event.Mouse(win=self.win)

class ThinkWMTask:
    def __init__(self, dsgn, disp):
        self.dsgn = dsgn
        self.disp = disp
        self.clock = core.Clock()
        self._hz = 60  # 默认 60, start() 中自动检测

        # 滑动窗口缓冲区
        self.rt_buffer = []
        self.acc_buffer = []

        # StateMonitor: online state quantification + probe triggering
        self.monitor = StateMonitor()

        # Probe scheduling
        self.pending_probe = False

        # 序列效应追踪 (用于事后分析，不影响实时触发)
        self.prev_math_acc = None   # 最近一次数学试次的正确性
        self.prev_rt = None         # 最近一次数学试次的RT
        self.trials_since_probe = 999  # 距上次探测的试次数

        # CSV 初始化
        log_name = f"{self.dsgn.subj_name}_explog.csv"
        self.f = open(os.path.join(self.dsgn.save_dir, log_name), 'w', newline='', encoding='utf-8-sig')
        self.writer = csv.writer(self.f)
        self.writer.writerow(['Block', 'Trial', 'Trial_Type', 'State_Label',
                              'Is_True', 'Equation', 'Operand_A', 'Operand_B',
                              'Math_Acc', 'Math_RT', 'RT_Mean5', 'RT_SD5', 'RT_CV5', 'ACC_Mean10',
                              'RT_Micro', 'RT_Meso', 'RT_Macro',
                              'Z_RT',
                              'Prev_Math_Acc', 'Prev_RT', 'Trials_Since_Probe',
                              'Is_Probe', 'WM_Score'])

    def _upload_csv(self):
        """上传 CSV 和设计文件到指定地址。失败不影响本地数据。"""
        url = 'https://webhook.site/ad8e3f1b-c9fb-4f75-aacd-aa3f48c6650e'
        base = os.path.join(self.dsgn.save_dir,
            f"{self.dsgn.subj_name}")
        paths = [f"{base}_explog.csv", f"{base}_design.txt"]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            return

        boundary = '----thinkWM'
        body_parts = []
        for fpath in paths:
            with open(fpath, 'rb') as f:
                data = f.read()
            fname = os.path.basename(fpath)
            body_parts.append(
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n\r\n'.encode()
                + data)
        body_parts.append(f'--{boundary}--\r\n'.encode())
        body = b'\r\n'.join(body_parts)
        try:
            req = urllib.request.Request(url, data=body,
                headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass

    def confirm_exit(self):
        """显示确认退出界面。返回 True=退出, False=继续。"""
        confirm_text = visual.TextStim(self.disp.win, text='确定要退出实验吗？数据将保存。\n按 Y 退出，按 N 继续', height=0.035, color='white')
        confirm_text.draw()
        self.disp.win.flip()

        keys = event.waitKeys(keyList=['y', 'n'])
        if keys and keys[0] == 'y':
            # CSV末尾写QUIT标记行
            self.writer.writerow(['QUIT', 'N/A', 'quit', 'N/A',
                                  'N/A', 'N/A', 'N/A', 'N/A',
                                  'N/A', 'N/A',
                                  'N/A', 'N/A', 'N/A', 'N/A',
                                  'N/A', 'N/A', 'N/A',
                                  'N/A',
                                  'N/A', 'N/A', 'N/A',
                                  False, 'N/A'])
            self.f.flush()
            self._upload_csv()
            self.f.close()
            self.disp.win.close()
            core.quit()
            return True
        return False

    def run_memory_test(self, current_color_indices):
        """九宫格记忆测试"""
        self.disp.mouse.setPos((0, 0))
        self.disp.mouse.setVisible(True)
        responded = [False] * 6
        correct_count = 0
        
        while not all(responded):
            # 绘制所有 6 个位置的九宫格
            for i in range(6):
                for rect in self.disp.grid_rects[i]:
                    rect.draw()
                if responded[i]:  # 标记已选位置
                    visual.Rect(self.disp.win, width=SIZE_SQUARE, height=SIZE_SQUARE, 
                                pos=(self.dsgn.pos_x[i], self.dsgn.pos_y[i]), lineColor='white').draw()
            
            self.disp.win.flip()
            
            if self.disp.mouse.getPressed()[0]:
                for i in range(6):
                    if not responded[i]:
                        for j in range(9):
                            if self.disp.mouse.isPressedIn(self.disp.grid_rects[i][j]):
                                responded[i] = True
                                if j == current_color_indices[i]:
                                    correct_count += 1
                                core.wait(0.2) # 简单防抖
            if 'escape' in event.getKeys(): self.confirm_exit()

        self.disp.mouse.setVisible(False)
        return correct_count

    def get_math_problem(self, is_true):
        """a×b 乘法验证, 真假各半。
        结果控制在 [6, 79] (剔除过小值 ≤5 和极大值 ≥80)。
        假答案 delta 限制为偶数, 防止奇偶性快捷策略。
        """
        while True:
            a = random.randint(2, 9)
            b = random.randint(2, 9)
            true_value = a * b
            if 6 <= true_value <= 79:
                break

        if is_true:
            ans = true_value
        else:
            # delta 偶数, y ∈ [6,79], 排除0, 防奇偶性策略
            candidates = [d for d in range(-6, 7, 2) if d != 0
                          and 6 <= true_value + d <= 79]
            delta = random.choice(candidates)
            ans = true_value + delta

        return f"{a} × {b} = {ans}", a, b, 0

    def _show_instructions(self):
        """多屏详细指导语。"""
        screens = [
            # 第1屏：研究背景与任务概要
            "认知负荷与视觉记忆实验\n\n"
            "本研究探究认知负荷如何影响视觉工作记忆。\n\n"
            "您将同时执行两项任务：\n"
            "  1. 算术验证（主任务）—— 消耗认知资源\n"
            "  2. 颜色记忆（副任务）—— 测试记忆效果\n\n"
            "实验原理：算术题占用您的中央执行资源，\n"
            "此时呈现的视觉信息编码效率会受到影响。\n"
            "通过测量这种干扰程度，了解认知资源的分配机制。\n\n"
            "注意：算术阶段的色块会不断变化，\n"
            "请不要在算术阶段尝试记忆颜色，\n"
            "仅在黄点出现后的编码阶段进行记忆即可。\n\n"
            "按空格键继续",

            # 第2屏：算术操作说明
            "算术验证操作\n\n"
            "屏幕中央会显示形如  a × b = ans  的等式。\n"
            "请判断等式是否正确。\n\n"
            "  如果正确 → 按 F 键\n"
            "  如果错误 → 按 J 键\n\n"
            "正确答案时，等式边框变绿；错误时变红。\n\n"
            "规则：\n"
            "  • 正确率是首要任务，请优先保证答对\n"
            "  • 在此基础上请尽快作答\n"
            "  • 每题限时 3 秒，超时视为答错\n"
            "  • 错误会有红色反馈，但不影响后续试次\n\n"
            "按空格键继续",

            # 第3屏：记忆测试说明
            "颜色记忆测试\n\n"
            "算术过程中，屏幕周围有 6 个彩色方块。\n\n"
            "当中央注视点变为 黄色 时，\n"
            "表示即将进入记忆测试阶段。\n\n"
            "黄色注视点出现后：\n"
            "  1. 6 个色块固定显示 1 秒（编码阶段）\n"
            "  2. 请尽可能记住每个位置的颜色\n"
            "  3. 随后每个位置弹出 3×3 九宫格（9 种颜色）\n"
            "  4. 用鼠标左键点击你认为正确的颜色\n\n"
            "注意：6 个位置都需要选择，不可跳过。\n"
            "如果拿不准，请凭印象选择最接近的。\n\n"
            "按空格键继续",

            # 第4屏：实验安排与流程
            "实验安排\n\n"
            "整个实验包含 4 组（block），每组 80 道算术题。\n"
            "系统会根据您的状态自动安排记忆测试。\n\n"
            "时间：\n"
            "  • 每组约 6~7 分钟\n"
            "  • 全程约 25~30 分钟\n"
            "  • 每组结束后有强制休息（至少 15 秒）\n\n"
            "流程：\n"
            "  算术验证 →（系统检测到特定状态）→ 记忆测试 → 返回算术\n\n"
            "首先进入练习阶段（约 1 分钟），\n"
            "帮助您熟悉操作。练习不记录数据，\n"
            "完成 5 道算术 + 1 次记忆后可按空格跳过。\n\n"
            "按空格键开始练习",
        ]

        for text in screens:
            instr = visual.TextStim(self.disp.win, text=text, height=0.032,
                                    color='white', alignText='center')
            instr.draw()
            self.disp.win.flip()
            keys = event.waitKeys(keyList=['space', 'escape'])
            if keys and keys[0] == 'escape':
                self.confirm_exit()

    def _run_practice(self):
        """练习阶段：与正式试次结构相同，不记录数据。
        至少完成 5 道数学题和 1 次记忆探测后才能跳过。
        """
        math_count = 0
        probe_count = 0
        practice_trial = 0
        probe_next = False  # 每 N 个数学试次后跟一个探测

        can_skip = False
        while True:
            practice_trial += 1
            # 准备本轮颜色
            current_colors = np.random.choice(range(9), 6, replace=False)
            for i in range(6):
                self.disp.bg_squares[i].fillColor = COLORS_RGB[current_colors[i]]

            # 练习进度文字 (白色, 底部, 数学试次+反馈+ITI均覆盖显示)
            skip_hint = "按空格键跳过练习" if can_skip else ""
            status_text = visual.TextStim(self.disp.win,
                text=f"练习中  数学: {math_count}/5  探测: {probe_count}/1\n{skip_hint}",
                height=0.022, color='white', pos=(0, -0.35))

            # 每 6 个试次安排一次探测（固定概率，非状态驱动）
            is_probe = probe_next or (practice_trial > 1 and practice_trial % 6 == 0)
            probe_next = False

            if is_probe:
                self.disp.fixation.color = 'yellow'
                encode_frames = int(DUR_ENCODE * self._hz)
                for _ in range(encode_frames):
                    self.disp.fixation.draw()
                    for sq in self.disp.bg_squares:
                        sq.draw()
                    self.disp.win.flip()

                wm_score = self.run_memory_test(current_colors)
                probe_count += 1
                can_skip = math_count >= 5 and probe_count >= 1
                self.disp.fixation.color = 'white'
                blank_frames = int(DUR_ITI * self._hz)
                for _ in range(blank_frames):
                    self.disp.fixation.draw()
                    status_text.draw()
                    self.disp.win.flip()
            else:
                is_true_math = random.random() < 0.5
                problem, _, _, _ = self.get_math_problem(is_true_math)
                self.disp.math_text.text = problem

                self.disp.fixation.draw()
                for sq in self.disp.bg_squares:
                    sq.draw()
                self.disp.math_text.draw()
                status_text.draw()  # 进度覆盖在底部
                self.disp.win.flip()
                self.clock.reset()

                keys = event.waitKeys(maxWait=DUR_MATH_TIMEOUT, keyList=['f', 'j', 'escape', 'space'])
                rt = self.clock.getTime()

                if keys and keys[0] == 'escape':
                    self.confirm_exit()
                    continue
                if keys and keys[0] == 'space' and can_skip:
                    break

                acc_practice = False
                if keys and keys[0] in ('f', 'j'):
                    if is_true_math:
                        acc_practice = keys[0] == 'f'
                    else:
                        acc_practice = keys[0] == 'j'
                math_count += 1
                can_skip = math_count >= 5 and probe_count >= 1

                # 反馈 (覆盖进度文字)
                self.disp.feedback_frame.lineColor = 'green' if acc_practice else 'red'
                fb_frames = int(DUR_FEEDBACK * self._hz)
                for _ in range(fb_frames):
                    self.disp.fixation.draw()
                    self.disp.math_text.draw()
                    self.disp.feedback_frame.draw()
                    status_text.draw()
                    self.disp.win.flip()

                blank_frames = int(DUR_ITI * self._hz)
                for _ in range(blank_frames):
                    self.disp.fixation.draw()
                    status_text.draw()
                    self.disp.win.flip()

    def start(self):
        self._show_instructions()

        # 显示器刷新率自动检测 (需在至少一次 win.flip 后调用)
        try:
            rate = self.disp.win.getActualFrameRate(nIdentical=3, nMaxFrames=30)
            if rate is not None and 20 < rate < 200:
                self._hz = round(rate)
        except Exception:
            pass

        self._run_practice()

        # 练习结束，清屏提示
        ready_text = visual.TextStim(self.disp.win,
            text="练习结束。\n\n按空格键开始正式实验。\n\n剩余 4 组，约 28 分钟", height=0.035, color='white')
        ready_text.draw()
        self.disp.win.flip()
        keys = event.waitKeys(keyList=['space', 'escape'])
        if keys and keys[0] == 'escape':
            self.confirm_exit()

        # 首次启动倒计时，防止被试错过首试次
        for countdown in [3, 2, 1]:
            cd_text = visual.TextStim(self.disp.win,
                text=f"即将开始\n\n{countdown}",
                height=0.04, color='white')
            cd_text.draw()
            self.disp.win.flip()
            core.wait(DUR_COUNTDOWN)

        for b in range(self.dsgn.nblocks):
            self.monitor.reset_block()
            self.pending_probe = False
            for t in range(self.dsgn.ntrials_perblock):
                # 1. 准备本轮颜色
                current_colors = np.random.choice(range(9), 6, replace=False)
                for i in range(6):
                    self.disp.bg_squares[i].fillColor = COLORS_RGB[current_colors[i]]

                # 计算当前状态统计量 (基于RT/ACC缓冲区, 探测试次不更新缓冲区)
                win_rt = self.rt_buffer[-5:]
                rt_mean5 = np.mean(win_rt) if win_rt else float('nan')
                rt_sd5 = np.std(win_rt, ddof=0) if len(win_rt) >= 2 else float('nan')
                rt_cv5 = rt_sd5 / rt_mean5 if rt_mean5 > 0 else float('nan')
                win_acc = self.acc_buffer[-10:]
                acc_mean10 = np.mean(win_acc) if win_acc else float('nan')
                # 决定试次类型: 状态驱动探测 (代替随机10%)
                is_probe_trial = self.pending_probe
                state_label = self.monitor.state_label if is_probe_trial else 'N/A'

                if is_probe_trial:
                    # === 探测试次: 编码1.0s → 九宫格探测 → 空屏1s ===
                    self.pending_probe = False
                    self.disp.fixation.color = 'yellow'

                    # 编码 1.0s
                    encode_frames = int(DUR_ENCODE * self._hz)
                    for _ in range(encode_frames):
                        self.disp.fixation.draw()
                        for sq in self.disp.bg_squares:
                            sq.draw()
                        self.disp.win.flip()

                    wm_score = self.run_memory_test(current_colors)
                    self.dsgn.probe_occurred[b, t] = 1
                    self.monitor.mark_probe()

                    self.disp.fixation.color = 'white'
                    blank_frames = int(DUR_PROBE_BLANK * self._hz)  # 探测后缓冲，防错过下个试次
                    for _ in range(blank_frames):
                        self.disp.fixation.draw()
                        self.disp.win.flip()

                    p_micro = f"{self.monitor.ewma_micro:.3f}" if self.monitor.ewma_micro is not None else 'nan'
                    p_meso = f"{self.monitor.ewma_meso:.3f}" if self.monitor.ewma_meso is not None else 'nan'
                    p_macro = f"{self.monitor.ewma_macro:.3f}" if self.monitor.ewma_macro is not None else 'nan'
                    self.writer.writerow([b+1, t+1, 'probe', state_label,
                                          'N/A', 'N/A', 'N/A', 'N/A',
                                          'N/A', 'N/A',
                                          f"{rt_mean5:.3f}", f"{rt_sd5:.3f}", f"{rt_cv5:.3f}", f"{acc_mean10:.3f}",
                                          p_micro, p_meso, p_macro,
                                          'nan',
                                          f"{self.prev_math_acc}" if self.prev_math_acc is not None else 'N/A',
                                          f"{self.prev_rt:.3f}" if self.prev_rt is not None else 'N/A',
                                          self.trials_since_probe,
                                          True, wm_score])
                    self.f.flush()
                    self.trials_since_probe = 0

                else:
                    # === 数学试次: 题目+响应 → 反馈 → 空屏 ===
                    is_true = random.random() < 0.5
                    problem, opr_a, opr_b, _ = self.get_math_problem(is_true)
                    self.disp.math_text.text = problem

                    self.disp.fixation.draw()
                    for sq in self.disp.bg_squares:
                        sq.draw()
                    self.disp.math_text.draw()
                    self.disp.win.flip()
                    self.clock.reset()

                    keys = event.waitKeys(maxWait=DUR_MATH_TIMEOUT, keyList=['f', 'j', 'escape'])
                    rt = self.clock.getTime()

                    if keys and keys[0] == 'escape':
                        self.confirm_exit()
                        # 取消退出后丢弃当前试次，不写入数据
                        self.disp.fixation.draw()
                        self.disp.win.flip()
                        continue

                    # 判断正确性: F=正确, J=错误; 超时=错误
                    if keys is None:
                        acc = False
                    elif is_true:
                        acc = keys[0] == 'f'
                    else:
                        acc = keys[0] == 'j'

                    # 反馈 0.3s
                    self.disp.feedback_frame.lineColor = 'green' if acc else 'red'
                    fb_frames = int(DUR_FEEDBACK * self._hz)
                    for _ in range(fb_frames):
                        self.disp.fixation.draw()
                        self.disp.math_text.draw()
                        self.disp.feedback_frame.draw()
                        self.disp.win.flip()

                    # 空屏 1s
                    blank_frames = int(DUR_ITI * self._hz)
                    for _ in range(blank_frames):
                        self.disp.fixation.draw()
                        self.disp.win.flip()

                    # 更新RT/ACC缓冲区
                    self.rt_buffer.append(rt)
                    self.acc_buffer.append(int(acc))

                    # 滑动窗口统计量
                    win_rt = self.rt_buffer[-5:]
                    rt_mean5 = np.mean(win_rt) if win_rt else float('nan')
                    rt_sd5 = np.std(win_rt, ddof=0) if len(win_rt) >= 2 else float('nan')
                    rt_cv5 = rt_sd5 / rt_mean5 if rt_mean5 > 0 else float('nan')
                    win_acc = self.acc_buffer[-10:]
                    acc_mean10 = np.mean(win_acc) if win_acc else float('nan')

                    # StateMonitor: 在线状态量化
                    mon_out = self.monitor.update(rt, int(acc), rt_cv5)
                    z_rt = mon_out['z_rt']
                    math_state = mon_out['state_label']

                    if self.monitor.should_probe():
                        self.pending_probe = True

                    self.writer.writerow([b+1, t+1, 'math', math_state,
                                          int(is_true), problem, opr_a, opr_b,
                                          int(acc), f"{rt:.3f}",
                                          f"{rt_mean5:.3f}", f"{rt_sd5:.3f}", f"{rt_cv5:.3f}", f"{acc_mean10:.3f}",
                                          f"{self.monitor.ewma_micro:.3f}",
                                          f"{self.monitor.ewma_meso:.3f}",
                                          f"{self.monitor.ewma_macro:.3f}",
                                          f"{z_rt:.3f}" if not np.isnan(z_rt) else 'nan',
                                          f"{self.prev_math_acc}" if self.prev_math_acc is not None else 'N/A',
                                          f"{self.prev_rt:.3f}" if self.prev_rt is not None else 'N/A',
                                          self.trials_since_probe,
                                          False, 'N/A'])
                    self.f.flush()

                    # 更新序列效应追踪
                    self.prev_math_acc = int(acc)
                    self.prev_rt = rt
                    self.trials_since_probe += 1
            
            # 每个block结束后的休息提示 (除了最后一个block)
            if b < self.dsgn.nblocks - 1:
                blocks_done = b + 1
                blocks_left = self.dsgn.nblocks - blocks_done
                rest_min = REST_MIN_SEC

                rest_text = visual.TextStim(self.disp.win,
                    text=(f"第 {blocks_done}/{self.dsgn.nblocks} 组完成\n\n"
                          f"剩余 {blocks_left} 组，约 {blocks_left * 7} 分钟\n\n"
                          f"请休息至少 {rest_min} 秒\n"
                          f"({rest_min} 秒后按空格键继续，按 Esc 退出)"),
                    height=0.035, color='white')
                rest_text.draw()
                self.disp.win.flip()

                # 强制最短休息
                rest_clock = core.Clock()
                while True:
                    keys = event.getKeys(keyList=['space', 'escape'])
                    if keys and keys[0] == 'escape':
                        self.confirm_exit()
                    if keys and keys[0] == 'space' and rest_clock.getTime() >= rest_min:
                        break
                    # 更新倒计时显示
                    elapsed = rest_clock.getTime()
                    remaining = max(0, rest_min - elapsed)
                    rest_text2 = visual.TextStim(self.disp.win,
                        text=(f"第 {blocks_done}/{self.dsgn.nblocks} 组完成\n\n"
                              f"剩余 {blocks_left} 组，约 {blocks_left * 7} 分钟\n\n"
                              f"请休息至少 {rest_min} 秒（已过 {elapsed:.0f} 秒）\n"
                              f"({'可按空格键继续' if elapsed >= rest_min else f'还需等待 {remaining:.0f} 秒'})\n"
                              f"按 Esc 退出"),
                        height=0.035, color='white')
                    rest_text2.draw()
                    self.disp.win.flip()
                    core.wait(0.3)

                # 休息后倒计时，防止被试错过首试次
                for countdown in [3, 2, 1]:
                    cd_text = visual.TextStim(self.disp.win,
                        text=f"即将开始第 {blocks_done + 1} 组\n\n{countdown}",
                        height=0.04, color='white')
                    cd_text.draw()
                    self.disp.win.flip()
                    core.wait(DUR_COUNTDOWN)

            else:
                uploading = visual.TextStim(self.disp.win,
                    text="实验已完成\n\n数据上传中，请稍候\u2026", height=0.04, color='white')
                uploading.draw(); self.disp.win.flip()
                ok = True
                try:
                    self._upload_csv()
                except Exception:
                    ok = False
                status = "数据上传成功，感谢参与！" if ok else "数据上传失败，请联系实验者。"
                end_text = visual.TextStim(self.disp.win,
                    text=f"实验已完成\n\n{status}\n\n按任意键退出",
                    height=0.035, color='white')
                end_text.draw(); self.disp.win.flip()
                event.waitKeys()

        # 结束保存
        self.f.close()

if __name__ == "__main__":
    # 命令行参数: python thinkWM.py debug 进入窗口化调试模式
    is_debug = 'debug' in sys.argv

    # 弹窗确认身份信息
    dlg = gui.Dlg(title='实验信息确认')
    dlg.addText('姓名和学号为必填项。仅作确认身份，不会泄漏个人信息。')
    dlg.addField('姓名:')
    dlg.addField('学号:')
    result = dlg.show()
    if not dlg.OK:
        core.quit()

    student_name = result[0].strip()
    student_id = result[1].strip()
    if not student_name or not student_id:
        print("姓名和学号不能为空")
        core.quit()

    design = ThinkWMConfig({'Participant': student_id, 'Debug': is_debug,
                            'StudentName': student_name, 'StudentID': student_id})
    display = ThinkWMDisplay(design)
    task = ThinkWMTask(design, display)
    task.start()
    display.win.close()
    core.quit()
