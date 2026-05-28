# thinkWM — Cognitive Load & Visual Working Memory

**English | [中文](README.zh-CN.md)**

A PsychoPy-based closed-loop dual-task experiment platform investigating how cognitive load fluctuations affect visual working memory encoding in real time.

## Background

In dual-task paradigms, the relationship between moment-to-moment fluctuations in primary-task cognitive load and secondary-task working memory encoding remains largely unexplored with online measures. DeBettencourt et al. (2019) established a closed-loop attention monitoring paradigm based on RT, but their primary task was a low-perceptual-load simple detection. When the primary task demands high cognitive load (arithmetic verification), RT variability reflects **task engagement** rather than attentional level.

According to **Load Theory** (Lavie 1995/2004), high cognitive load consumes central executive resources, impairing active encoding of non-target stimuli. Thus, high engagement (fast RT / high accuracy) may paradoxically predict *worse* WM performance — the opposite direction from DeBettencourt. Pilot data (5 repeated sessions + 2 independent subjects) consistently supports this reversed pattern.

## Experimental Design

### Dual-Task Structure

| Task | Description | Role |
|------|-------------|------|
| Arithmetic verification (primary) | Judge `a × b = ans`, F=correct J=wrong, 3s timeout | Consume executive resources |
| Color memory (secondary) | Encode 6 color positions → recall via 3×3 grid mouse click | Measure WM encoding |

### Parameters

- 4 blocks × 80 trials (full) / 1 block × 999 trials (debug)
- Math format: `a × b` (a,b ∈ [2,9], result ∈ [6,79]), 50% true/false
- False-answer delta constrained to even numbers (±2, ±4, ±6) — prevents parity-based shortcuts
- Color blocks change dynamically during math phase — no informative encoding before probe
- Encoding: 1.0s (fixation turns yellow → colors displayed)
- Total duration: ~25-30 min

### StateMonitor Algorithm

Cumulative z-score (30-trial baseline) + dual-speed ACC EWMA for real-time cognitive state quantification:

```
Probe score S = |Z_RT| × (1 + min(ACC_decline / 0.12, 2.0))
```

| State Label | Condition | Interpretation |
|-------------|-----------|---------------|
| optimal | Z_RT < -1.1, ACC=1 | Fast + correct → deep engagement |
| cautious | Z_RT > 1.1, ACC_decline ≤ 0.03 | Slow but correct → disengaged/cautious |
| lapse | Z_RT > 1.1, ACC_decline > 0.03 | Slow + declining → attentional lapse |
| impulsive | Z_RT < -1.1, ACC=0 | Fast + error → impulsive |
| acc_decline | ACC_decline > 0.12 | Sustained accuracy drop |

Trigger: S ≥ 1.1 + cooldown ≥ 5 trials + budget ≤ 10/block; emergency S ≥ 4.0 overrides cooldown.

## Installation

```bash
pip install -r requirements.txt
```

Core: `psychopy`, `numpy`, `scipy`. Python 3.10 recommended (see `.python-version`).

Analysis extras: `pandas`, `matplotlib` (included in requirements.txt).

## Usage

```bash
# Debug mode: windowed, 1 block × 999 trials
python thinkWM.py debug

# Full mode: fullscreen, 4 blocks × 80 trials
python thinkWM.py
```

Startup flow:
1. Dialog: enter name & student ID (identity confirmation only)
2. 4 instruction screens (space to advance)
3. Practice phase (5 math + 1 probe trials, skippable)
4. Main experiment with inter-block rests (≥ 15s)
5. Auto-save & upload to GitLab (Deploy Token built-in, just works™)

## Data Output

## Data Upload

Experiment data is automatically uploaded to the USTC GitLab Generic Packages registry via a built-in **Deploy Token** (scoped to `write_package_registry` only — can only upload data, cannot read code or modify repository settings).

**No setup required.** Just run the experiment and data gets uploaded:

```cmd
thinkWM.exe
```

Data appears at:
```
https://git.ustc.edu.cn/YinXiran/thinkwm/-/packages
```
→ **thinkwm-data** package → grouped by date (`YYYYMMDD`)

### Override with custom token

If you want to use your own token (e.g., for a fork), set `GITLAB_TOKEN` environment variable in `username:token` format:

```cmd
set GITLAB_TOKEN=your-username:your-token
thinkWM.exe
```

When `GITLAB_TOKEN` is set, it takes precedence over the built-in token.

`data/<timestamp>/<timestamp>_explog.csv`, 28 columns (timestamp = session start time):

| Column | Description |
|--------|-------------|
| Block, Trial | Block and trial number |
| Trial_Type | math / probe |
| State_Label | Cognitive state at trigger |
| Equation | Math equation string |
| Operand_A, B | Operands |
| Math_Acc, Math_RT | Accuracy (0/1) and reaction time (s) |
| RT_Mean5, RT_SD5, RT_CV5 | Rolling 5-trial RT window |
| ACC_Mean10 | Rolling 10-trial accuracy |
| RT_Micro/Meso/Macro | Triple EWMA (α=0.30/0.10/0.02) |
| Z_RT | Cumulative z-score |
| Prev_Math_Acc, Prev_RT | Previous trial accuracy and RT |
| Trials_Since_Probe | Trials since last probe |
| Is_Probe, WM_Score | Probe trial flag, WM score (/6) |

## Data Analysis

### CLI (quick report)

```bash
# Single subject
python analysis/thinkwm_analysis.py subject_id

# Multiple subjects
python analysis/thinkwm_analysis.py subject_id1 subject_id2
```

### Interactive (Python/ipython)

```python
from analysis.thinkwm_analysis import load_subject, wm_by_state

df = load_subject('subject_id')
result = wm_by_state(df)       # WM by cognitive state
diff_analysis(df)               # Engaged vs disengaged comparison
```

Features: multi-session merge, WM-by-state stats, engaged/disengaged difference tests, difficulty-stratified analysis, post-error analysis.

## Key Findings (Pilot Data)

1. **Reversed pattern**: Disengaged states (cautious/acc_decline) show consistently higher WM than engaged (optimal) — opposite to attention-paradigm direction
2. **Difficulty modulation**: Effect driven by hard problems (hard: Δ=+1.73, p=.028); easy/medium non-significant
3. **|Z_RT| does not independently predict WM**: r = -0.40 ~ 0.07
4. **Ceiling effect**: High-WM subject (009: 5.27/6) showed no state differentiation
5. **No post-error WM effect**: Δ=+0.006, p=.73, but state labels shifted (χ²=36.8, p<.0001)

## Shortcuts

| Key | Function |
|-----|----------|
| F | Equation correct |
| J | Equation wrong |
| ESC | Exit experiment (confirm dialog) |
| Space | Advance instructions / skip practice |

## Tests

```bash
python -m pytest tests/ -v    # 45 tests, ~1 sec
```

- **StateMonitor tests** (32): initialization, z-score, ACC EWMA, state labels, trigger logic, edge cases
- **Math problem tests** (13): operand range, true/false answers, even-delta validation, format

## Tech Stack

- Python 3.10 + PsychoPy + NumPy + SciPy
- 45 pytest tests with MockPsychopy isolation
- PyInstaller packaging (`build_exe.py`)

## Project Structure

```
thinkWM.py              — Main experiment script
build_exe.py            — PyInstaller build script
README.md               — This file (English)
README.zh-CN.md         — Chinese version
requirements.txt        — Dependencies
LICENSE                 — MIT License
data/                   — Experiment data (gitignored)
analysis/               — Analysis scripts
tests/                  — pytest suite
```

## Citation

DeBettencourt, M. T., Keene, P. A., Awh, E., & Vogel, E. K. (2019). Real-time monitoring of attention fluctuations in the visual working memory system. *Nature Human Behaviour*, 3(8), 792–800.

## Disclaimer

This project is **self-directed research** conducted independently. It has not been supervised, reviewed, or endorsed by any academic advisor or institution. The experimental design, analysis, and interpretations are solely the author's own work and may contain errors or oversights. **Not peer-reviewed.** Use the findings with appropriate caution.

## License

MIT License. See [LICENSE](LICENSE).
