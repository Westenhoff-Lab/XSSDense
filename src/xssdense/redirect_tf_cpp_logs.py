"""Logging redirect for cubersom cpp warnings.
It sends TensorFlow's C++ log output (fd 2) to a file.
Must run before `import tensorflow`."""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def redirect(log_path="tf_cpp.log"):
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    orig_stderr_fd = os.dup(2)  # keep the real stderr
    os.dup2(log_fd, 2)  # fd 2 now goes to the file
    try:
        get_ipython  # notebook / VS Code interactive window: Python errors don't use fd 2
    except NameError:
        sys.stderr = os.fdopen(orig_stderr_fd, "w", buffering=1)  # script: keep tracebacks on the terminal
