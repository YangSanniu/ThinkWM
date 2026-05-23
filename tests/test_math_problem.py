"""Unit tests for get_math_problem — a×b only format with even delta."""
import sys, random, os, tempfile, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from conftest import MockPsychopy
MockPsychopy.install()
import thinkWM

_tmpdir = tempfile.mkdtemp()
_mock_dsgn = types.SimpleNamespace(
    subj_name='test', timestamp='000000_000000', save_dir=_tmpdir
)
task = thinkWM.ThinkWMTask(dsgn=_mock_dsgn, disp=None)


class TestMathProblemGeneration:
    def test_true_equation(self):
        for _ in range(100):
            eq, a, b, c = task.get_math_problem(is_true=True)
            assert a * b == int(eq.split('=')[-1].strip())

    def test_false_equation_mismatch(self):
        mismatches = 0
        for _ in range(200):
            eq, a, b, c = task.get_math_problem(is_true=False)
            result = int(eq.split('=')[-1].strip())
            if a * b != result:
                mismatches += 1
        assert mismatches >= 190  # delta is always non-zero

    def test_false_answer_within_6_to_79(self):
        for _ in range(200):
            eq, a, b, c = task.get_math_problem(is_true=False)
            ans = int(eq.split('=')[-1].strip())
            assert 6 <= ans <= 79, f"False answer {ans} out of [6, 79]"

    def test_true_answer_within_6_to_79(self):
        for _ in range(200):
            eq, a, b, c = task.get_math_problem(is_true=True)
            ans = int(eq.split('=')[-1].strip())
            assert 6 <= ans <= 79, f"True answer {ans} out of [6, 79]"

    def test_operand_a_range(self):
        for _ in range(200):
            eq, a, b, c = task.get_math_problem(is_true=True)
            assert 2 <= a <= 9

    def test_operand_b_range(self):
        for _ in range(200):
            eq, a, b, c = task.get_math_problem(is_true=True)
            assert 2 <= b <= 9

    def test_c_always_zero(self):
        for _ in range(200):
            eq, a, b, c = task.get_math_problem(is_true=True)
            assert c == 0

    def test_format_contains_multiply_symbol(self):
        for _ in range(10):
            eq, a, b, c = task.get_math_problem(is_true=True)
            assert '×' in eq

    def test_returns_4_values(self):
        result = task.get_math_problem(is_true=True)
        assert len(result) == 4
        eq, a, b, c = result
        assert isinstance(eq, str)
        assert isinstance(a, int)
        assert isinstance(b, int)
        assert isinstance(c, int)

    def test_deterministic_true(self):
        random.seed(42)
        r1 = task.get_math_problem(is_true=True)
        random.seed(42)
        r2 = task.get_math_problem(is_true=True)
        assert r1 == r2

    def test_false_delta_even(self):
        """Delta must be even (prevents parity shortcut strategy)."""
        for _ in range(200):
            eq, a, b, c = task.get_math_problem(is_true=False)
            true_val = a * b
            ans = int(eq.split('=')[-1].strip())
            delta = ans - true_val
            assert delta != 0, f"Delta is zero for {eq}"
            assert delta % 2 == 0, f"Delta {delta} not even for {eq} (true={true_val})"
            assert abs(delta) <= 6, f"Delta {delta} exceeds ±6 for {eq}"

    def test_no_edge_cases(self):
        """True value must be in [6, 79], not 4, 5, 80, 81."""
        for _ in range(500):
            eq, a, b, c = task.get_math_problem(is_true=True)
            ans = int(eq.split('=')[-1].strip())
            assert 6 <= a * b <= 79, f"a*b={a * b} out of [6, 79]"
            assert ans == a * b

    def test_no_invalid_operands(self):
        random.seed(0)
        for _ in range(500):
            eq, a, b, c = task.get_math_problem(is_true=random.random() < 0.5)
            assert a > 0 and b > 0
            assert a * b >= 6
