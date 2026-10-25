"""Structured Python test tools. No shell command strings are accepted."""
import ast
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

IGNORED = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'build', 'dist'}


def project_path(root, target):
    root = Path(root).resolve()
    path = (root / target).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Target must be inside the project workspace.')
    if not path.exists():
        raise ValueError(f'Target does not exist: {target}')
    return path


def python_files(root, target='.'):
    path = project_path(root, target)
    if path.is_file():
        if path.suffix != '.py':
            raise ValueError('Choose a Python file or directory.')
        yield path
        return
    for folder, directories, files in os.walk(path, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in IGNORED and not Path(folder, d).is_symlink())
        for name in sorted(files):
            candidate = Path(folder, name)
            if candidate.suffix == '.py' and not candidate.is_symlink():
                yield candidate


def discover_tests(root, target='.'):
    """Find conventionally named tests without importing project code."""
    found = []
    scanned = 0
    for path in python_files(root, target):
        scanned += 1
        if path.name.startswith('test_') or path.name.endswith('_test.py'):
            found.append(str(path.relative_to(root)))
        if scanned >= 5000 or len(found) >= 200:
            return {'status': 'partial', 'files': found, 'note': 'Discovery limit reached.'}
    return {'status': 'found' if found else 'none_found', 'files': found,
            'note': 'Filename discovery only; use run_tests to execute tests.'}


def check_syntax(root, target='.'):
    """Parse Python sources without executing them or producing bytecode."""
    checked = 0
    errors = []
    for path in python_files(root, target):
        checked += 1
        try:
            ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, ValueError, OSError) as error:
            errors.append({'file': str(path.relative_to(root)),
                           'line': getattr(error, 'lineno', None), 'error': str(error)})
        if checked >= 5000:
            return {'status': 'partial', 'checked': checked, 'errors': errors}
    return {'status': ('failed' if errors else 'passed') if checked else 'none_found',
            'checked': checked, 'errors': errors,
            'note': 'Syntax checks do not establish runtime correctness.'}


def run_tests(root, runner='pytest', target='.', timeout_seconds=60, confirm=input):
    """Run pytest or unittest after approval, returning exit code and output."""
    if runner not in {'pytest', 'unittest'}:
        raise ValueError('runner must be pytest or unittest')
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 300:
        raise ValueError('timeout_seconds must be between 1 and 300')
    path = project_path(root, target)
    if runner == 'pytest':
        command = [sys.executable, '-m', 'pytest', str(path), '-q', '--tb=short']
    else:
        directory = path if path.is_dir() else path.parent
        pattern = 'test*.py' if path.is_dir() else path.name
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(directory), '-p', pattern, '-v']
    if confirm(f'Run {runner} on {target}? This executes project code. [y/N] ').strip().lower() != 'y':
        return {'status': 'cancelled', 'runner': runner, 'target': target}
    environment = {key: os.environ[key] for key in
                   ('PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP') if key in os.environ}
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONIOENCODING='utf-8')
    started = time.monotonic()
    # Capture to a temporary file rather than growing an in-memory pipe buffer.
    with tempfile.TemporaryFile() as output:
        try:
            completed = subprocess.run(command, cwd=root, env=environment, shell=False,
                                       stdout=output, stderr=subprocess.STDOUT, timeout=timeout_seconds)
            code = completed.returncode
            status = 'passed' if code == 0 else ('none_found' if runner == 'pytest' and code == 5 else 'failed')
        except subprocess.TimeoutExpired:
            status, code = 'timed_out', None
        except OSError as error:
            return {'status': 'error', 'error': str(error), 'runner': runner}
        size = output.tell()
        output.seek(max(0, size - 12000))
        log = output.read().decode('utf-8', errors='replace')
    if runner == 'pytest' and 'No module named pytest' in log:
        status = 'missing_dependency'
    if runner == 'unittest' and code in {0, 5} and 'Ran 0 tests' in log:
        status = 'none_found'
    return {'status': status, 'runner': runner, 'target': target, 'exit_code': code,
            'duration_seconds': round(time.monotonic() - started, 2),
            'output': ('[Earlier output omitted]\n' if size > 12000 else '') + log}