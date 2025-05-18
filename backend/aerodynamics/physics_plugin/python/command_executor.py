#!/usr/bin/env python3
"""
command_executor.py

Enterprise-grade wrapper for invoking the aerodynamics_physics_plugin
binary (or any external CLI) with JSON I/O, retries, async support,
and structured logging.
"""
import os
import json
import subprocess
import logging
import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union


# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("CommandExecutor")
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(levelname)8s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ──────────────────────────────────────────────────────────────────────────────
class CommandExecutorError(Exception):
    """
    Base exception for all errors raised by CommandExecutor.

    Attributes:
        message: human–readable error message
        returncode: (optional) process exit code if applicable
        stderr: (optional) captured stderr bytes or string from the process
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr: bytes | str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.returncode = returncode
        self.stderr = stderr

    def __str__(self) -> str:
        base = self.message

        if self.returncode is not None:
            base += f" [exit code: {self.returncode}]"

        if self.stderr:
            # Decode bytes if necessary.
            err_txt = (
                self.stderr.decode("utf-8", errors="ignore")
                if isinstance(self.stderr, (bytes, bytearray))
                else str(self.stderr)
            )
            base += f"\nSTDERR:\n{err_txt}"

        return base


class JSONDecodingError(CommandExecutorError):
    """
    Raised when the CLI returns invalid or malformed JSON.

    Inherits:
        returncode (sometimes None, since parse may occur after success)
        stderr (raw bytes, if captured)
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr: bytes | str | None = None,
        raw_output: bytes | None = None,
    ) -> None:
        full_msg = message

        if raw_output is not None:
            snippet = raw_output[:200].decode("utf-8", errors="ignore")
            full_msg += f"\n--- Raw output snippet ---\n{snippet!r}"

        super().__init__(full_msg, returncode=returncode, stderr=stderr)


class CLIExecutionError(CommandExecutorError):
    """
    Raised when the subprocess returns a non-zero exit code.

    Attributes:
        returncode: the integer exit code (non-zero)
        stderr: captured stderr from the subprocess
    """

    def __init__(
        self, message: str, *, returncode: int, stderr: bytes | str | None = None
    ) -> None:
        # if stderr is bytes, we'll let the base class decode on __str__
        super().__init__(message, returncode=returncode, stderr=stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Utility Decorators
# ──────────────────────────────────────────────────────────────────────────────
def retry(
    *,
    attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    exceptions: tuple = (subprocess.SubprocessError, CommandExecutorError),
) -> Callable:
    """
    Decorator to retry a function on failure.
    """

    def decorator(fn):
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == attempts:
                        logger.error("All %d attempts failed.", attempts)
                        raise
                    logger.warning(
                        "Attempt %d/%d failed with '%s', retrying in %.2f s...",
                        attempt,
                        attempts,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor

        return wrapper

    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────
def run_and_parse_json(
    cmd: list[str], *, timeout: float | None = None, env: dict[str, str] | None = None
) -> dict:
    """Run a command, enforce timeout, and parse its stdout as JSON."""
    proc_env = os.environ.copy()

    if env:
        proc_env.update(env)

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            env=proc_env,
        )
    except subprocess.TimeoutExpired as e:
        raise CLIExecutionError(
            f"Command timed out after {e.timeout} seconds",
            returncode=None,
            stderr=e.stderr if hasattr(e, "stderr") else None,
        ) from e
    except OSError as e:
        # e.g. FileNotFoundError when the command doesn't exist
        raise CLIExecutionError(
            f"failed to execute command: {e}",
            returncode=None,
            stderr=None
        ) from e

    if completed.returncode != 0:
        raise CLIExecutionError(
            f"Command failed (exit code {completed.returncode})",
            returncode=completed.returncode,
            stderr=completed.stderr,
        )

    try:
        return json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as e:
        raise JSONDecodingError(
            "Failed to parse JSON from command output",
            returncode=completed.returncode,
            stderr=completed.stderr,
        ) from e


# ──────────────────────────────────────────────────────────────────────────────
# CommandExecutor Class
# ──────────────────────────────────────────────────────────────────────────────
class CommandExecutor:
    """
    Wraps invocation of an external executable with structured JSON I/O,
    retry logic, async interface, and pluggable validation.
    """

    def __init__(
        self,
        exe_path: Union[str, Path],
        *,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
        validate_input: Optional[Callable[[Dict], None]] = None,
        validate_output: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        """
        :param exe_path: path to the binary
        :param timeout: timeout per invocation (in seconds)
        :param env: extra environment variables
        :param validate_input: optional function to validate input dict
        :param validate_output: optional function to validate output dict
        """
        self.exe_path = Path(exe_path)
        self.timeout = timeout
        self.env = {**subprocess.os.environ, **(env or {})}
        self.validate_input = validate_input
        self.validate_output = validate_output

        if not self.exe_path.exists():
            logger.warning("Executable '%s' not found.", self.exe_path)

    def _encode_input(self, data: Optional[Dict[str, Any]]) -> Optional[bytes]:
        """Validate (if configured) then JSON-serialize input dict."""
        if data is None:
            return None

        if self.validate_input:
            self.validate_input(data)

        try:
            return json.dumps(data).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise CommandExecutorError(f"Input JSON serialization failed: {e}") from e

    def _parse_output(self, stdout: bytes) -> Dict[str, Any]:
        """JSON-parse stdout into a Python dict, with optional validation."""
        try:
            out_text = stdout.decode("utf-8").strip()
            result = json.loads(out_text)
        except Exception as e:
            raise JSONDecodingError(
                f"Invalid JSON from CLI: {e}\nRaw: {stdout!r}"
            ) from e

        if self.validate_output:
            self.validate_output(result)

        return result

    @retry(attempts=3, initial_delay=0.5)
    def run(
        self,
        args: Optional[list[str]] = None,
        data: Optional[Dict[str, Any]] = None,
        capture_output: bool = True,
        check: bool = True,
    ) -> Dict[str, Any]:
        """
        Synchronous invocation.

        :param args: additional CLI arguments
        :param data: JSON-serializable input dict
        :param capture_output: capture & parse stdout/stderr
        :param check: raise on non-zero return code
        :returns: parsed JSON dict from stdout
        """
        cmd = [str(self.exe_path)] + (args or [])
        logger.info("Running: %s", " ".join(cmd))

        inp = self._encode_input(data)

        try:
            proc = subprocess.run(
                cmd,
                input=inp,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                timeout=self.timeout,
                env=self.env,
                check=check,
            )
        except subprocess.CalledProcessError as e:
            raise CLIExecutionError(f"CLI exited {e.returncode}: {e.stderr}") from e

        if capture_output and proc.stderr:
            logger.debug("STDERR: %s", proc.stderr.decode("utf-8", errors="ignore"))

        return self._parse_output(proc.stdout or b"")

    async def run_async(
        self,
        args: Optional[list[str]] = None,
        data: Optional[Dict[str, Any]] = None,
        capture_output: bool = True,
        check: bool = True,
    ) -> Dict[str, Any]:
        """
        Asynchronous invocation using asyncio.create_subprocess_exec.
        """
        cmd = [str(self.exe_path)] + (args or [])
        logger.info("Async running: %s", " ".join(cmd))

        inp = self._encode_input(data)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if inp else None,
            stdout=asyncio.subprocess.PIPE if capture_output else None,
            stderr=asyncio.subprocess.PIPE if capture_output else None,
            env=self.env,
        )

        stdout, stderr = await proc.communicate(inp)

        if check and proc.returncode != 0:
            raise CLIExecutionError(f"Async CLI exited {proc.returncode}")

        if capture_output and stderr:
            logger.debug("Async STDERR: %s", stderr.decode())

        return self._parse_output(stdout or b"")


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synchronous example
    exe = CommandExecutor(
        exe_path="../build/libaerodynamics_physics_plugin.dylib",
        timeout=30.0,
    )
    inp = {
        "mesh_file": "mesh.vtk",
        "solver": "k_epsilon",
        "dt": 0.01,
    }
    try:
        out = exe.run(args=["--simulate"], data=inp)
        print(json.dumps(out, indent=2))
    except CommandExecutorError as err:
        logger.error("Sync execution failed: %s", err)

    # Asynchronous example
    async def main():
        exe_async = CommandExecutor("../build/libaerodynamics_physics_plugin.dylib")
        res = await exe_async.run_async(args=["--simulate"], data=inp)
        print("Async result:", res)

    asyncio.run(main())
