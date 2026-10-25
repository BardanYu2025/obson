"""Execute the real shell entrypoint against a recording fake Python command."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/babel_input_factorial768_autodl.sh"


@pytest.fixture
def shell(tmp_path):
    directory = tmp_path / "workspace with spaces"
    (directory / "scripts").mkdir(parents=True)
    script = directory / "scripts" / SCRIPT.name
    shutil.copy2(SCRIPT, script)
    recorder = directory / "fake python"
    recorder.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as f:\n"
        "    f.write(json.dumps({'args':sys.argv[1:], 'cwd':os.getcwd(), "
        "'omp':os.environ.get('OMP_NUM_THREADS'), 'pythonpath':os.environ.get('PYTHONPATH')})+'\\n')\n"
        "exporting = sys.argv[2].endswith('_export')\n"
        "raise SystemExit(int(os.environ.get('EXPORT_EXIT' if exporting else 'RUN_EXIT', '0')))\n"
    )
    recorder.chmod(0o755)
    log = directory / "calls.jsonl"
    environment = {
        **os.environ,
        "PYTHON_BIN": str(recorder),
        "CALL_LOG": str(log),
        "OMP_NUM_THREADS": "invalid inherited value",
        "BABEL_MULTISCALE_RUN": "must-not-use-old-experiment",
        "BABEL_OBSERVED_HISTORY_RUN": "must-not-use-old-attention-experiment",
        "BABEL_CONTENT_GUARD_RUN": "must-not-use-old-feedback-experiment",
    }

    def invoke(*arguments, run_exit=0, export_exit=0, extra=None):
        env = {
            **environment,
            "RUN_EXIT": str(run_exit),
            "EXPORT_EXIT": str(export_exit),
            **(extra or {}),
        }
        result = subprocess.run(
            ["bash", str(script), *arguments], env=env, capture_output=True, text=True, timeout=10
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return invoke, directory


@pytest.mark.parametrize("status", [0, 1, 124, 130])
def test_success_failure_timeout_and_stop_all_export_original_exit(shell, status):
    invoke, directory = shell
    result, calls = invoke("all", run_exit=status)
    assert result.returncode == status
    assert len(calls) == 2
    command, export = calls
    assert command["args"][:2] == ["-m", "obson.babel.input_factorial_run"]
    assert command["args"][2:] == [
        "--source",
        "checkpoints/babel_input_patch768",
        "--out",
        "checkpoints/babel_input_factorial768",
    ]
    assert export["args"][:2] == ["-m", "obson.babel.input_factorial_export"]
    assert export["args"][export["args"].index("--command-exit-code") + 1] == str(status)
    assert export["args"][export["args"].index("--download") + 1] == "/root/autodl-tmp/download"
    assert all(call["omp"] == "4" and call["cwd"] == str(directory) for call in calls)
    assert all(call["pythonpath"].split(":")[0] == str(directory / "src") for call in calls)


def test_export_only_never_starts_worker_and_quotes_paths(shell):
    invoke, _ = shell
    result, calls = invoke(
        "export",
        extra={"BABEL_INPUT_FACTORIAL_RUN": "checkpoints/a b", "BABEL_DOWNLOAD_DIR": "/tmp/a b"},
    )
    assert result.returncode == 0 and len(calls) == 1
    assert calls[0]["args"][1] == "obson.babel.input_factorial_export"
    assert calls[0]["args"][3] == "checkpoints/a b"
    assert calls[0]["args"][5] == "/tmp/a b"


def test_evaluate_only_is_explicit_and_export_failure_is_nonzero(shell):
    invoke, _ = shell
    result, calls = invoke("evaluate", export_exit=9)
    assert result.returncode == 9 and "Report export failed" in result.stderr
    assert calls[0]["args"][-1] == "--evaluate-only"


def test_export_error_does_not_hide_original_worker_error(shell):
    invoke, _ = shell
    result, calls = invoke("all", run_exit=7, export_exit=9)
    assert result.returncode == 7 and len(calls) == 2


@pytest.mark.parametrize("args", [("train",), ("all", "--epochs", "240")])
def test_unknown_modes_or_extra_training_overrides_fail_before_execution(shell, args):
    invoke, _ = shell
    result, calls = invoke(*args)
    assert result.returncode == 2 and not calls
