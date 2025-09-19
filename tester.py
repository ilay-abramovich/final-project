import decimal
from decimal import Decimal
import enum
import io
from itertools import combinations
import os
import re
import subprocess
import tempfile
from typing import Optional, Any, IO

import numpy as np


VALGRIND_ERRCODE = 99
EPS = 1e-4
TRIALS_PROGRAMS = 5
TRIALS_SYMNMF_LIB = 5
TRIALS_VALGRIND_C = 3
TRIALS_VALGRIND_PY_SYMNMF = 6
TRIALS_ANALYSIS_PY = 5
TEST_PYTHON_MEMORY = True

REGEX_NUMBER_FMT = r"-?(?:0|[1-9]\d*)\.\d{4}"
REGEX_ANALYSIS_PY_OUTPUT = re.compile(
    rf"nmf: {REGEX_NUMBER_FMT}\nkmeans: {REGEX_NUMBER_FMT}"
)

__author__ = "Yahel Caspi"
__version__ = "0.4.0"


class ProgramType(enum.Enum):
    PYTHON = "python"
    C = "c"


class TestData:
    def __init__(self, round=True, dedup=False):
        rng = np.random.default_rng()

        retry = True  # Used to trigger retry in case of division by zero
        while retry:
            retry = False

            n = rng.integers(50, 801)
            dim = rng.integers(2, 11)
            k = rng.integers(2, n // 2)

            if rng.choice(2):
                centers = rng.normal(scale=10.0, size=(k, dim))
            else:
                centers = rng.uniform(-10, 11, size=(k, dim))

            self.X = rng.choice(centers, n) + rng.standard_normal((n, dim))
            if round:
                self.X = np.round(self.X, 4)

            if dedup:
                self.X = np.unique(self.X, axis=0)

            self.A = similarity_matrix(self.X)

            np_old_err_settings = np.seterr(divide="raise")
            try:
                self.D = ddg(self.A)
            except FloatingPointError:
                retry = True
                continue
            finally:
                np.seterr(**np_old_err_settings)

            self.W = normalized_similarity_matrix(self.A, self.D)


def print_green(msg: str, prefix: str = ""):
    if prefix:
        msg = f"[{prefix}] {msg}"

    print(f"\033[32m{msg}\033[0m")


def print_yellow(msg: str, prefix: str = ""):
    if prefix:
        msg = f"[{prefix}] {msg}"

    print(f"\033[33m{msg}\033[0m")


def print_red(msg: str, prefix: str = ""):
    if prefix:
        msg = f"[{prefix}] {msg}"

    print(f"\033[31m{msg}\033[0m")


def print_white_on_red(msg: str):
    print(f"\033[97;41m{msg}\033[0m")


def format_goal_name(goal_name: str) -> str:
    return f"\033[1;3m{goal_name}\033[22;23m"


def similarity_matrix(X: np.ndarray):
    n = X.shape[0]
    A = np.zeros((n, n), dtype=np.float64)

    for i, j in combinations(range(n), 2):
        A[i, j] = A[j, i] = np.exp(-np.linalg.norm(X[i] - X[j]) ** 2 / 2)

    return A


def ddg(A: np.ndarray):
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]

    D = np.diag(A.sum(axis=1))

    return D


def normalized_similarity_matrix(A: np.ndarray, D: np.ndarray):
    D_inv_sqrt = np.diag(D.diagonal() ** (-1 / 2))
    return D_inv_sqrt @ A @ D_inv_sqrt


def initialize_H(W: np.ndarray, k: int, set_seed=False):
    from symnmf import init_H

    if set_seed:
        np.random.seed(1234)

    return init_H(W, k)


def symnmf_main(W: np.ndarray, k: int, set_seed=False):
    MAX_ITER = 300
    BETA = 0.5

    initial_H = initialize_H(W, k, set_seed)
    H = initial_H.copy()

    for _ in range(MAX_ITER):
        H_next = H * (1 - BETA + BETA * ((W @ H) / (H @ H.T @ H)))
        H, H_next = H_next, H
        if np.linalg.norm(H - H_next) ** 2 < EPS:
            break

    return initial_H, H


def generate_data(K=20, points_num=None, round=False):
    rng = np.random.default_rng()
    dim = rng.integers(2, 10)
    N = points_num if points_num else rng.integers(100, 700)

    centroids = rng.uniform(-11, 11, (K, dim))
    data = rng.choice(centroids, N) + rng.standard_normal((N, dim))

    # Remove duplicate points
    data = np.unique(data, axis=0)

    while data.shape[0] < N:
        extra = rng.choice(centroids, N - data.shape[0]) + rng.standard_normal(
            (N - data.shape[0], dim)
        )
        data = np.unique(np.vstack([data, extra]), axis=0)

    # Round data to 4 decimal places for a fair comparison
    if round:
        data = np.round(data, 4)

    return data.astype(np.float64)


def make_stub_file(data):
    file = tempfile.NamedTemporaryFile(suffix=".txt", delete=True)
    buf = io.TextIOWrapper(file)

    for row in data:
        print(",".join(f"{x:.4f}" for x in row), file=buf)
    buf.detach()

    return file


def execute_python_program(
    k, goal, filename, use_valgrind=False
) -> tuple[subprocess.CompletedProcess[str], Optional[IO[Any]]]:
    args = ["python3", "symnmf.py", str(k), goal, filename]

    pass_fds = []
    env = None
    if use_valgrind:
        logfile = tempfile.TemporaryFile()
        fd = logfile.fileno()
        # Ensures the fd is inherited by child processes
        os.set_inheritable(fd, True)

        args = [
            "valgrind",
            "--leak-check=full",
            f"--log-fd={fd}",
            f"--error-exitcode={VALGRIND_ERRCODE}",
            "--suppressions=python.supp",
            "--show-leak-kinds=definite,indirect",
            "--errors-for-leak-kinds=definite,indirect",
        ] + args
        env = os.environ.copy()
        env["PYTHONMALLOC"] = "debug"

        pass_fds.append(fd)

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        pass_fds=pass_fds,
        env=env,
    )

    if use_valgrind:
        logfile.seek(0)
        return result, logfile

    return result, None


def test_analysis_py():
    successes = 0
    for trial in range(1, TRIALS_ANALYSIS_PY + 1):
        success = True
        print(f"trial {trial}/{TRIALS_ANALYSIS_PY}")

        test_data = TestData(dedup=True)
        k = np.random.default_rng().integers(2, 11)
        with make_stub_file(test_data.X) as tmpfile:
            args = ["python3", "analysis.py", str(k), tmpfile.name]
            result = subprocess.run(args, capture_output=True, text=True)

        if result.returncode != 0:
            print_red(
                f"failure: process had a non-zero return code [{result.returncode}]"
            )
            success = False

        if result.stderr:
            print_red("failure: process had a non-empty stderr")
            print_white_on_red(result.stderr)
            success = False

        if REGEX_ANALYSIS_PY_OUTPUT.fullmatch(result.stdout.removesuffix("\n")) is None:
            print_red("failure: bad format for analysis.py output")
            success = False

        if success:
            successes += 1

    if successes == TRIALS_ANALYSIS_PY:
        print_green("success")
    elif successes > 0:
        print_red(
            f"failure: succeeded in {successes} out of {TRIALS_ANALYSIS_PY} trials"
        )
    else:
        print_red("failure: no trial succeeded")


def execute_c_program(
    goal, filename, use_valgrind=False
) -> tuple[subprocess.CompletedProcess[str], Optional[IO[Any]]]:
    args = ["./symnmf", goal, filename]

    pass_fds = []
    if use_valgrind:
        logfile = tempfile.TemporaryFile()
        fd = logfile.fileno()
        # Ensures the fd is inherited by child processes
        os.set_inheritable(fd, True)

        args = [
            "valgrind",
            "--leak-check=full",
            f"--log-fd={fd}",
            f"--error-exitcode={VALGRIND_ERRCODE}",
        ] + args

        pass_fds.append(fd)

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        pass_fds=pass_fds,
    )

    if use_valgrind:
        logfile.seek(0)
        return result, logfile

    return result, None


def verify_output(result: str, matrix: np.ndarray, prefix_msg: str = ""):
    PERMITTED_DELTA = Decimal("0.0001")

    lines = result.rstrip("\n").splitlines()
    if len(lines) == 1 and lines[0] == "An Error Has Occurred":
        print_red('failure: process returned "An Error Has Occurred"', prefix_msg)
        return False

    if len(lines) != matrix.shape[0]:
        print_red(
            "failure: incorrect number of lines. "
            f"expected {matrix.shape[0]}, got {len(lines)}",
            prefix_msg,
        )
        return False

    different_entries = 0
    for i, line in enumerate(lines):
        coordinates = line.split(",")
        if len(coordinates) != matrix.shape[1]:
            print_red(
                "failure: incorrect number of columns. "
                f"expected {matrix.shape[1]}, got {len(coordinates)}",
                prefix_msg,
            )
            return False

        try:
            coordinates = [Decimal(coordinate) for coordinate in coordinates]
        except decimal.InvalidOperation:
            print_red("failure: coordinates couldn't be parsed", prefix_msg)
            return False

        for j, coordinate in enumerate(coordinates):
            expected = Decimal.from_float(matrix[i, j]).quantize(PERMITTED_DELTA)
            if expected != coordinate:
                different_entries += 1

            if (coordinate - expected).copy_abs() > PERMITTED_DELTA:
                print_red(
                    "failure: incorrect output. "
                    f"expected {expected}, got {coordinate}",
                    prefix_msg,
                )
                return False

    if different_entries:
        print_yellow(
            f"found {different_entries} out of {matrix.size} entries "
            "with mismatching output (within tolerance)",
            prefix_msg,
        )

    return True


def test_c_program(
    goal: str, filename: str, reference_matrix: np.ndarray, use_valgrind: bool
):
    success = True

    result, valgrind_logfile = execute_c_program(goal, filename, use_valgrind)
    if use_valgrind and result.returncode == VALGRIND_ERRCODE:
        valgrind_log = valgrind_logfile.read()
        print_red("memory leak detected by valgrind")
        print(valgrind_log.decode())
        success = False
    elif result.returncode != 0:
        print_red(f"failure: process had a non-zero return code [{result.returncode}]")
        success = False

    if result.stderr:
        print_red("failure: process had a non-empty stderr")
        print(result.stderr)
        success = False

    return success and verify_output(result.stdout, reference_matrix)


def test_goal(
    program_type: ProgramType,
    goal: str,
    filename: str,
    use_valgrind: bool,
    target_matrix,
    k=0,
) -> bool:
    if program_type is ProgramType.PYTHON and goal == "symnmf":
        assert k > 1

    success = True
    prefix_msg = f"prog={program_type.value},goal={goal}"

    result, valgrind_logfile = (
        execute_c_program(goal, filename, use_valgrind)
        if program_type is ProgramType.C
        else execute_python_program(k, goal, filename, use_valgrind)
    )
    if use_valgrind and result.returncode == VALGRIND_ERRCODE:
        valgrind_log = valgrind_logfile.read()
        print_red("memory leak detected by valgrind", prefix_msg)
        print_white_on_red(valgrind_log.decode())
        success = False
    elif result.returncode != 0:
        print_red(
            f"failure: process had a non-zero return code [{result.returncode}]",
            prefix_msg,
        )
        success = False

    if result.stderr:
        print_red("failure: process had a non-empty stderr", prefix_msg)
        print_white_on_red(result.stderr)
        success = False

    return success and verify_output(result.stdout, target_matrix, prefix_msg)


def test_symnmf_lib():
    import symnmfmodule as symnmf

    test_data = TestData(round=False)
    rng = np.random.default_rng()
    k = rng.integers(2, 13)

    err_msg = (
        "failure: goal {} returned a result that's too distant from the expected one"
    )

    goal_name = format_goal_name("sym")
    A = np.array(symnmf.sym(test_data.X))
    if not np.all(np.linalg.norm(test_data.A - A, axis=1) < EPS):
        print_red(err_msg.format(goal_name))
        return False

    goal_name = format_goal_name("ddg")
    D = np.array(symnmf.ddg(test_data.X))
    if not np.all(np.linalg.norm(test_data.D - D, axis=1) < EPS):
        print_red(err_msg.format(goal_name))
        return False

    goal_name = format_goal_name("norm")
    W_target = normalized_similarity_matrix(test_data.A, test_data.D)
    W = np.array(symnmf.norm(test_data.X))
    if not np.all(np.linalg.norm(W_target - W, axis=1) < EPS):
        print_red(err_msg.format(goal_name))
        return False

    goal_name = format_goal_name("symnmf")
    initial_H, final_H_target = symnmf_main(W, k)
    final_H = np.array(symnmf.symnmf(initial_H, W))
    if not np.all(np.linalg.norm(final_H_target - final_H, axis=1) < EPS):
        print_red(err_msg.format(goal_name))
        return False

    if np.linalg.norm(W - (final_H @ final_H.T)) < EPS:
        print_yellow(f"warning: {goal_name} failed to converge")
        return False

    return True


def test_programs():
    rng = np.random.default_rng()

    stats = {
        "python": {
            "symnmf": 0,
            "sym": 0,
            "ddg": 0,
            "norm": 0,
        },
        "c": {
            "sym": 0,
            "ddg": 0,
            "norm": 0,
        },
    }

    for _ in range(TRIALS_PROGRAMS):
        test_data = TestData()
        goals = (
            ("sym", test_data.A),
            ("ddg", test_data.D),
            ("norm", test_data.W),
        )

        with make_stub_file(test_data.X) as tmpfile:
            for goal, target_matrix in goals:
                if not test_goal(
                    ProgramType.C, goal, tmpfile.name, False, target_matrix
                ):
                    stats["c"][goal] += 1

                if not test_goal(
                    ProgramType.PYTHON, goal, tmpfile.name, False, target_matrix
                ):
                    stats["python"][goal] += 1

            # Test symnmf goal (python only)
            k = rng.integers(2, 11)
            _, target_H = symnmf_main(test_data.W, k, True)
            if not test_goal(
                ProgramType.PYTHON, "symnmf", tmpfile.name, False, target_H, k
            ):
                stats["python"]["symnmf"] += 1

    print("Summary - C:")
    for goal in ("sym", "ddg", "norm"):
        goal_name = format_goal_name(goal)
        if stats["c"][goal]:
            print_red(
                f"goal {goal_name} failed {stats['c'][goal]} out of {TRIALS_PROGRAMS} times"
            )
        else:
            print_green(f"goal {goal_name} succeeded")

    print("Summary - Python:")
    for goal in ("sym", "ddg", "norm", "symnmf"):
        goal_name = format_goal_name(goal)
        if stats["python"][goal]:
            print_red(
                f"goal {goal_name} failed {stats['python'][goal]} out of {TRIALS_PROGRAMS} times"
            )
        else:
            print_green(f"goal {goal_name} succeeded")


def test_with_valgrind():
    print("Testing C")
    stats = {
        "sym": 0,
        "ddg": 0,
        "norm": 0,
    }
    for _ in range(TRIALS_VALGRIND_C):
        test_data = TestData()
        goals = (
            ("sym", test_data.A),
            ("ddg", test_data.D),
            ("norm", test_data.W),
        )

        with make_stub_file(test_data.X) as tmpfile:
            for goal, target_matrix in goals:
                if not test_goal(
                    ProgramType.C, goal, tmpfile.name, True, target_matrix
                ):
                    stats[goal] += 1

    for goal in stats:
        goal_name = format_goal_name(goal)
        if stats[goal]:
            print_red(
                f"goal {goal_name} failed {stats[goal]} out of {TRIALS_VALGRIND_C} times"
            )
        else:
            print_green(f"goal {goal_name} succeeded")

    if TEST_PYTHON_MEMORY:
        print("\nTesting python 'simple' goals")
        test_data = TestData()
        goals = (
            ("sym", test_data.A),
            ("ddg", test_data.D),
            ("norm", test_data.W),
        )

        with make_stub_file(test_data.X) as tmpfile:
            for goal, target_matrix in goals:
                goal_name = format_goal_name(goal)
                print(f"Testing {goal_name}:")
                if test_goal(ProgramType.PYTHON, goal, tmpfile.name, True, target_matrix):
                    print_green("success")
                else:
                    print_red(f"goal {goal_name} failed")

        del goals, test_data

        print(f"\nTesting python {format_goal_name('synmnf')} goal")
        rng = np.random.default_rng()
        successes = 0
        for trial in range(1, TRIALS_VALGRIND_PY_SYMNMF + 1):
            print(f"trial {trial}/{TRIALS_VALGRIND_PY_SYMNMF}")
            test_data = TestData()
            k = rng.integers(2, 11)
            _, target_H = symnmf_main(test_data.W, k, True)

            with make_stub_file(test_data.X) as tmpfile:
                if test_goal(ProgramType.PYTHON, "symnmf", tmpfile.name, True, target_H, k):
                    successes += 1

        if successes == TRIALS_VALGRIND_PY_SYMNMF:
            print_green("success")
        elif successes > 0:
            print_red(
                f"failure: succeeded in {successes} out of {TRIALS_VALGRIND_PY_SYMNMF} trials"
            )
        else:
            print_red("failure: no trial succeeded")
    else:
        print()
        print_yellow("\033[3mTesting python is disabled")


if __name__ == "__main__":
    print("--------")
    print("Testing programs")
    print("--------")
    test_programs()

    print("\n--------")
    print("Testing symnmf extension directly")
    print("--------")
    symnmflib_sucesses = 0
    for trial in range(1, TRIALS_SYMNMF_LIB + 1):
        print(f"trial {trial}/{TRIALS_PROGRAMS}")
        if test_symnmf_lib():
            symnmflib_sucesses += 1

    if symnmflib_sucesses == TRIALS_SYMNMF_LIB:
        print_green("success")
    elif symnmflib_sucesses > 0:
        print_red(
            f"failure: succeeded in {symnmflib_sucesses} out of {TRIALS_SYMNMF_LIB} trials"
        )
    else:
        print_red("failure: no trial succeeded")

    print("\n--------")
    print("Testing with valgrind")
    print("--------")
    test_with_valgrind()

    print("\n--------")
    print("Testing analysis.py (format only)")
    print("--------")
    test_analysis_py()
