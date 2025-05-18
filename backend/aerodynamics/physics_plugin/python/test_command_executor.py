# File: backend/aerodynamics/physics_plugin/python/test_command_executor.py
import os
import sys
import stat
import json
import tempfile
import time
import pytest

# Ensure we import command_executor from this directory
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from command_executor import (
    run_and_parse_json,
    CLIExecutionError,
    JSONDecodingError,
)


def make_executable_script(dirpath, name, content):
    """Helper: write content to a script and mark it executable."""
    path = os.path.join(dirpath, name)
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def test_successful_json_roundtrip(tmp_path):
    script = make_executable_script(
        tmp_path,
        "ok.sh",
        "#!/usr/bin/env bash\n" 'echo \'{"status":"ok","value":42}\'\n' "exit 0\n",
    )
    result = run_and_parse_json([script])
    assert result == {"status": "ok", "value": 42}


def test_cli_execution_error_includes_stderr(tmp_path):
    script = make_executable_script(
        tmp_path,
        "fail.sh",
        "#!/usr/bin/env bash\n" "echo 'oops, error occurred' >&2\n" "exit 7\n",
    )
    with pytest.raises(CLIExecutionError) as exc:
        run_and_parse_json([script])
    msg = str(exc.value)
    assert "exit code 7" in msg
    assert "oops, error occurred" in msg  # stderr should appear in the exception


def test_json_decoding_error_on_stdout(tmp_path):
    script = make_executable_script(
        tmp_path, "badjson.sh", "#!/usr/bin/env bash\n" "echo 'not a json'\n" "exit 0\n"
    )
    with pytest.raises(JSONDecodingError):
        run_and_parse_json([script])


def test_ignore_stderr_with_valid_json(tmp_path):
    script = make_executable_script(
        tmp_path,
        "mixed.sh",
        "#!/usr/bin/env bash\n"
        "echo 'warning: be careful' >&2\n"
        "echo '{\"ok\":true}'\n"
        "exit 0\n",
    )
    result = run_and_parse_json([script])
    assert result == {"ok": True}


def test_json_on_stderr_only(tmp_path):
    # JSON on stderr, nothing on stdout -> should still JSONDecodeError
    script = make_executable_script(
        tmp_path,
        "to_stderr.sh",
        "#!/usr/bin/env bash\n" "echo '{\"a\":1}' >&2\n" "exit 0\n",
    )
    with pytest.raises(JSONDecodingError):
        run_and_parse_json([script])


def test_nonexistent_command_raises(tmp_path):
    # Command doesn't exist
    with pytest.raises(CLIExecutionError) as exc:
        run_and_parse_json(["nonexistent-command-xyz"])
    assert "failed to execute" in str(exc.value)


def test_large_json_output(tmp_path):
    large = list(range(5000))
    js = json.dumps({"big": large})
    script = make_executable_script(
        tmp_path, "large.sh", "#!/usr/bin/env bash\n" f"printf '%s' '{js}'\n"
    )
    result = run_and_parse_json([script])
    assert isinstance(result["big"], list)
    assert len(result["big"]) == 5000
    assert result["big"][0] == 0 and result["big"][-1] == 4999


def test_unicode_output(tmp_path):
    # Script prints Unicode JSON
    data = {"emoji": "🚀", "text": "αβγ"}
    js = json.dumps(data, ensure_ascii=False)
    script = make_executable_script(
        tmp_path, "unicode.sh", "#!/usr/bin/env bash\n" f"echo '{js}'\n"
    )
    result = run_and_parse_json([script])
    assert result["emoji"] == "🚀"
    assert result["text"] == "αβγ"


def test_timeout_raises(tmp_path):
    # Script sleeps longer than our timeout
    script = make_executable_script(
        tmp_path,
        "sleep.sh",
        "#!/usr/bin/env bash\n" "sleep 2\n" "echo '{\"done\":true}'\n",
    )
    with pytest.raises(CLIExecutionError) as exc:
        # assume run_and_parse_json supports a timeout kwarg in seconds
        run_and_parse_json([script], timeout=0.5)
    assert "timed out" in str(exc.value).lower()


def test_env_var_passing(tmp_path):
    # Script inspects an env var and echoes it back in JSON
    script = make_executable_script(
        tmp_path,
        "env.sh",
        "#!/usr/bin/env bash\n" 'echo "{\\"MYVAR\\": \\"$MYVAR\\"}"\n',
    )
    result = run_and_parse_json([script], env={"MYVAR": "HelloWorld"})
    assert result["MYVAR"] == "HelloWorld"
