from __future__ import print_function

import time


class Timer(object):

    clean = True

    def __init__(self, prefix="", precision=2, verbose=True):
        self.start_time = None
        self.sub_timer = None
        self.prefix = prefix
        self.precision = precision
        self.verbose = verbose

    def _write(self, string, *args, **kwargs):
        """
        If the verbose setting is true: print(string.format(args), kwargs)
        :param str string:
        :param args:
        :param kwargs:
        """
        if self.verbose:
            print(string.format(*args), **kwargs)

    def start(self, title):
        if self.start_time is not None:
            self.stop()

        self.start_time = time.time()
        if not Timer.clean:
            self._write("")
        self._write("{}{}...", self.prefix, title, end="")
        Timer.clean = False

    def stop(self):
        if self.start_time is not None:
            time_taken = time.time() - self.start_time
            self.start_time = None
            if self.sub_timer is not None:
                self.sub_timer.stop()
                self.sub_timer = None
            if Timer.clean:
                self._write(self.prefix, end="")
            self._write((" done (took {:." + str(self.precision) + "f}s)"), time_taken)
            Timer.clean = True
            return time_taken
        return None

    def read(self):
        if self.start_time is not None:
            return time.time() - self.start_time
        return None

    def sub_time(self):
        """
        :rtype: Timer|None
        """
        if self.start_time is None:
            return None
        if self.sub_timer is None:
            self.sub_timer = Timer(prefix=self.prefix + "\t", precision=self.precision, verbose=self.verbose)
        return self.sub_timer

    def log(self, message):
        if self.start_time is None:
            self._write("{}{}", self.prefix, message)
        else:
            if not Timer.clean:
                self._write("")
                Timer.clean = True
            self._write("{}\t{}", self.prefix, message)
