"""
A wrap of python's print() function. So it can save all prints to a log file.
Copyright 2019 Yang Kaiyu yku12cn@gmail.com
"""

import sys
from pathlib import Path
from io import StringIO
from time import time, strftime, localtime


def _timestamp(ftime):
    r"""Generate a time stamp from seconds

    Args:
        ftime (flot): seconds

    Returns:
        str : a time stamp YYYY/MM/DD hh:mm:ss.sss
    """
    return "[%s.%03d] " % (strftime("%Y/%b/%d %H:%M:%S", localtime(ftime)),
                           ftime * 1000 % 1000)


def tStamp():
    r"""Generate a compact time string

    Returns:
        str: YYYYMMDDhhmmss
    """
    return "%04d%02d%02d%02d%02d%02d" % localtime()[0:6]


class printLog():
    r"""Logging while Printing!

    Keyword Arguments:
        log (str): Path of .log file (default: {""})
                   Leave empty if you don't need .log
        mkdirF (bool): create dir for .log (default: {True})
    """
    def __init__(self, log="", mkdirF=True, encoding="utf-8"):
        if log == "":
            self._logfile = None
        else:
            self.logfile(log=log, mkdirF=mkdirF, encoding=encoding)
            # self.logfile = open(Path(log), "a")

    def logfile(self, log=None, mkdirF=True, encoding="utf-8"):
        r"""Assign or re-assign the log file

        Args:
            log (str, optional): Path of .log file. Defaults to None.
            mkdirF (bool, optional): create dir for .log. Defaults to True.
            encoding (str, optional): encoding scheme. Defaults to "utf-8".

        Returns:
            str: call logfile() without Arguments will return
                 the path of current logfile
        """
        if not log:
            if not self._logfile:
                return None
            return self._logfile.name
        if hasattr(self, "_logfile"):
            # check if logfile() is called by __init__
            if self._logfile is not None:
                # if a file instance already exists, close the instance
                self._logfile.close()

        logpath = Path(log)
        if mkdirF:
            if not logpath.parent.exists():
                Path.mkdir(logpath.parent, parents=True)
        self._logfile = open(logpath, "a", encoding=encoding)
        return self._logfile.name

    def __call__(self, *args, t=False, redirect=False,
                 file=None, flush=False, **kwargs):
        r"""A wrap of python built-in print()

        Args:
            t (bool, optional): gen timestamp for each log. Defaults to False.
            redirect (bool, optional): if true, the print() will be muted and
                               its results will be returned. Defaults to False.
            file (write-able, optional): inherent from print().
            flush (bool, optional): inherent from print()

        Returns:
            str: return results if redirect is True
        """
        if self._logfile:
            # print to log
            if t:  # Print time stamp
                print(_timestamp(time()), end="", file=self._logfile)
            print(*args, file=self._logfile, flush=True, **kwargs)

        if redirect:
            # handle redirect flag
            temp = StringIO()
            print(*args, file=temp, flush=flush, **kwargs)
            return temp.getvalue()

        if not file:
            file = sys.stdout
        print(*args, file=file, flush=flush, **kwargs)
        return None


if __name__ == "__main__":
    # This is a short demo
    from time import sleep
    printL = printLog("./aaa/%s.log" % tStamp(), mkdirF=True)
    print(printL.logfile())
    for i in range(10):
        printL("hello from iter", i, t=True, end="")
        printL(" balabala?")
        sleep(0.1)
