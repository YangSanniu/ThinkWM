"""Unit tests for StateMonitor — pure logic, mocked PsychoPy."""
import sys, math, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Install mock before importing thinkWM
from conftest import MockPsychopy
MockPsychopy.install()
import thinkWM
StateMonitor = thinkWM.StateMonitor


class TestInit:
    def test_default_init(self):
        sm = StateMonitor()
        assert sm.baseline_trials == 30
        assert sm.cooldown == 5
        assert sm.max_probes == 10
        assert sm.score_threshold == 1.1
        assert sm.cum_n == 0
        assert sm.state_label == 'baseline'
        assert sm.probe_score == 0.0
        assert math.isnan(sm.z_rt)
        assert sm.block_probes == 0
        assert sm.since_last_probe == 0

    def test_custom_params(self):
        sm = StateMonitor(baseline_trials=10, cooldown=3, max_probes_per_block=5, score_threshold=2.0)
        assert sm.baseline_trials == 10
        assert sm.cooldown == 3
        assert sm.max_probes == 5
        assert sm.score_threshold == 2.0


class TestBaseline:
    def test_no_z_score_in_baseline(self):
        sm = StateMonitor(baseline_trials=30)
        for i in range(30):
            rt = 1.0 + np.random.uniform(-0.1, 0.1)  # need variance for cum_sd>0
            out = sm.update(rt=rt, acc=1, rt_cv5=0.2)
            assert np.isnan(out['z_rt'])
            assert out['state_label'] == 'baseline'
            assert out['probe_score'] == 0.0

    def test_boundary_trial_30_baseline(self):
        sm = StateMonitor(baseline_trials=30)
        for i in range(30):
            out = sm.update(rt=1.0, acc=1, rt_cv5=0.2)
        # After 30 trials (cum_n=30), cum_n > 30 is False, so still baseline
        assert out['state_label'] == 'baseline'

    def test_trial_31_exits_baseline(self):
        sm = StateMonitor(baseline_trials=30)
        for i in range(30):
            rt = 1.0 + np.random.uniform(-0.1, 0.1)  # variance so cum_sd>0
            sm.update(rt=rt, acc=1, rt_cv5=0.2)
        out = sm.update(rt=1.0, acc=1, rt_cv5=0.2)
        assert out['state_label'] != 'baseline'


class TestZScore:
    def test_zero_at_mean(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            rt = 1.0 + np.random.uniform(-0.15, 0.15)  # variance needed
            sm.update(rt=rt, acc=1, rt_cv5=0.2)
        # Feed exactly the mean — z should be ~0
        out = sm.update(rt=sm.cum_mean, acc=1, rt_cv5=0.2)
        assert abs(out['z_rt']) < 0.001

    def test_sign(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            rt = 1.0 + np.random.uniform(-0.15, 0.15)
            sm.update(rt=rt, acc=1, rt_cv5=0.2)
        fast = sm.update(rt=sm.cum_mean - sm.cum_sd, acc=1, rt_cv5=0.2)
        slow = sm.update(rt=sm.cum_mean + sm.cum_sd, acc=1, rt_cv5=0.2)
        assert fast['z_rt'] < 0
        assert slow['z_rt'] > 0


class TestAccEWMA:
    def test_convergence(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        assert sm.acc_fast > 0.9
        assert sm.acc_slow > 0.9

    def test_decline_detection(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(10):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        for i in range(5):
            sm.update(rt=1.0, acc=0, rt_cv5=0.1)
        assert sm.acc_fast < sm.acc_slow

    def test_peak_tracking(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(10):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        peak = sm.acc_slow_peak
        for i in range(5):
            sm.update(rt=1.0, acc=0, rt_cv5=0.1)
        assert sm.acc_slow_peak == peak


class TestStateLabels:
    def test_lapse(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        for i in range(5):
            sm.update(rt=1.0, acc=0, rt_cv5=0.1)
        out = sm.update(rt=3.0, acc=0, rt_cv5=0.1)
        assert out['state_label'] == 'lapse'

    def test_cautious(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        out = sm.update(rt=3.0, acc=1, rt_cv5=0.1)
        assert out['state_label'] == 'cautious'

    def test_optimal(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        out = sm.update(rt=0.3, acc=1, rt_cv5=0.1)
        assert out['state_label'] == 'optimal'

    def test_impulsive(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        out = sm.update(rt=0.3, acc=0, rt_cv5=0.1)
        assert out['state_label'] == 'impulsive'

    def test_valid_labels_only(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        valid = {'baseline', 'lapse', 'cautious', 'impulsive', 'optimal',
                 'acc_decline', 'neutral'}
        for i in range(200):
            rt = np.random.uniform(0.3, 3.0)
            acc = np.random.choice([0, 1])
            cv = np.random.uniform(0.0, 0.5)
            out = sm.update(rt=rt, acc=acc, rt_cv5=cv)
            assert out['state_label'] in valid, f"Invalid label: {out['state_label']}"


class TestShouldProbe:
    def test_below_threshold(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        assert not sm.should_probe()

    def test_budget(self):
        sm = StateMonitor(max_probes_per_block=2)
        sm.block_probes = 2
        assert not sm.should_probe()

    def test_cooldown_active(self):
        sm = StateMonitor(cooldown=5)
        sm.since_last_probe = 2
        sm.probe_score = 2.0
        assert not sm.should_probe()

    def test_cooldown_passed(self):
        sm = StateMonitor(cooldown=5)
        sm.since_last_probe = 5
        sm.probe_score = 2.0
        assert sm.should_probe()

    def test_emergency_override(self):
        sm = StateMonitor(cooldown=5)
        sm.since_last_probe = 1
        sm.probe_score = 6.0
        assert sm.should_probe()

    def test_min_gap_always_respected(self):
        sm = StateMonitor(cooldown=5)
        sm.since_last_probe = 0
        sm.probe_score = 10.0
        assert not sm.should_probe()

    def test_mark_probe(self):
        sm = StateMonitor(cooldown=5)
        sm.since_last_probe = 10
        sm.mark_probe()
        assert sm.since_last_probe == 0
        assert sm.block_probes == 1


class TestReset:
    def test_full_reset(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(10):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        sm.mark_probe()
        sm.reset()
        assert sm.cum_n == 0
        assert sm.acc_fast is None
        assert sm.state_label == 'baseline'
        assert sm.block_probes == 0

    def test_block_reset(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(10):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        cum_n_before = sm.cum_n
        sm.mark_probe()
        sm.reset_block()
        assert sm.block_probes == 0
        assert sm.since_last_probe == 0
        assert sm.cum_n == cum_n_before


class TestEdgeCases:
    def test_stable_subject_neutral(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            rt = 1.0 + np.random.uniform(-0.15, 0.15)
            sm.update(rt=rt, acc=1, rt_cv5=0.1)
        for i in range(50):
            out = sm.update(rt=sm.cum_mean + np.random.uniform(-0.02, 0.02), acc=1, rt_cv5=0.1)
            assert out['state_label'] == 'neutral'

    def test_all_errors_no_crash(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        for i in range(20):
            out = sm.update(rt=1.0, acc=0, rt_cv5=0.1)
            assert 'state_label' in out

    def test_extreme_rt(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        out = sm.update(rt=0.1, acc=1, rt_cv5=0.1)
        assert out['z_rt'] < -2.0
        out = sm.update(rt=3.0, acc=1, rt_cv5=0.1)
        assert out['z_rt'] > 2.0

    def test_nan_cv5(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=float('nan'))
        out = sm.update(rt=1.0, acc=1, rt_cv5=float('nan'))
        assert 'state_label' in out

    def test_zero_rt(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        out = sm.update(rt=0.001, acc=1, rt_cv5=0.1)
        assert not math.isinf(out['probe_score'])

    def test_rapid_switching(self):
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        for i in range(50):
            if i % 2 == 0:
                out = sm.update(rt=0.3, acc=1, rt_cv5=0.05)
            else:
                out = sm.update(rt=2.5, acc=0, rt_cv5=0.3)
            assert out['state_label'] in ('optimal', 'lapse', 'impulsive', 'cautious',
                                           'acc_decline', 'neutral')

    def test_ewma_not_nan(self):
        sm = StateMonitor(baseline_trials=5)
        sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        assert sm.acc_fast is not None
        assert not math.isnan(sm.acc_fast)

    def test_probe_score_returns(self):
        """Verify update returns expected keys."""
        sm = StateMonitor(baseline_trials=5)
        for i in range(5):
            sm.update(rt=1.0, acc=1, rt_cv5=0.1)
        out = sm.update(rt=1.5, acc=1, rt_cv5=0.1)
        assert 'z_rt' in out
        assert 'z_cv' in out
        assert 'state_label' in out
        assert 'probe_score' in out
