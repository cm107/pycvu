import os
import sys
import io

f = open('file.txt', 'w')

class RedirectStdToFile(object):
    """
    Usage:
        >>> f_stdout = open('stdout.log', 'w')
        >>> f_stderr = open('stderr.log', 'w')
        >>> print('Fubar')
        >>> with RedirectStdToFile(stdout=f_stdout, stderr=f_stderr):
        >>>     print("You can only see me in the log.")
        >>> print("I'm back!")
    
    Based on post: https://stackoverflow.com/a/6796752
    """
    def __init__(
        self,
        stdout: io.TextIOWrapper=None,
        stderr: io.TextIOWrapper=None
    ):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
    
    @staticmethod
    def debug():
        f_stdout = open('stdout.log', 'w')
        f_stderr = open('stderr.log', 'w')
        print('Fubar')

        with RedirectStdToFile(stdout=f_stdout, stderr=f_stderr):
            print("You can only see me in the log.")

        print("I'm back!")

class SuppressStd(RedirectStdToFile):
    """
    Usage:
        >>> print('Fubar')
        >>> with SuppressStd():
        >>>     print("You'll never see me")
        >>> print("I'm back!")
    
    Based on post: https://stackoverflow.com/a/6796752
    """

    def __init__(self, showError: bool=True):
        devnull = open(os.devnull, 'w')
        super().__init__(
            stdout=devnull,
            stderr=devnull if not showError else None
        )

    @staticmethod
    def debug():
        print('Fubar')
        with SuppressStd():
            print("You'll never see me")

        print("I'm back!")


class RedirectStdToVariable(object):
    """
    Usage:
        >>> print('Fubar')
        >>> stdout_var = io.StringIO()
        >>> with RedirectStdToVariable(stdout_var):
        >>>     print('This line will get stored in the variable.')
        >>>     print('Hello World!')
        >>> print("I'm back!")
        >>> lines = stdout_var.getvalue().split('\\n')[:-1]
        >>> print(f"{lines}")
        ['This line will get stored in the variable.', 'Hello World!']
    """

    def __init__(
        self,
        stdout: io.StringIO=None,
        stderr: io.StringIO=None
    ):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
    
    @staticmethod
    def debug():
        print('Fubar')
        stdout_var = io.StringIO()
        with RedirectStdToVariable(stdout_var):
            print('This line will get stored in the variable.')
            print('Hello World!')
        
        print("I'm back!")

        lines = stdout_var.getvalue().split('\n')[:-1]
        print(f"{lines}")
