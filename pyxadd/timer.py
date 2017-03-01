from __future__ import print_function

import time


class Timer(object):

    clean = True

    def __init__(self, prefix=""):
        self.start_time = None
        self.sub_timer = None
        self.prefix = prefix

    def start(self, title):
        if self.start_time is not None:
            self.stop()

        self.start_time = time.time()
        if not Timer.clean:
            print()
        print("{}{}...".format(self.prefix, title), end="")
        Timer.clean = False

    def stop(self):
        if self.start_time is not None:
            time_taken = time.time() - self.start_time
            self.start_time = None
            if self.sub_timer is not None:
                self.sub_timer.stop()
                self.sub_timer = None
            if Timer.clean:
                print(self.prefix, end="")
            print(" done (took {:.2f}s)".format(time_taken))
            Timer.clean = True
            return time_taken
        return None

    def sub_time(self):
        """
        :rtype: Timer|None
        """
        if self.start_time is None:
            return None
        if self.sub_timer is None:
            self.sub_timer = Timer(prefix=self.prefix + "\t")
        return self.sub_timer

    def log(self, message):
        if self.start_time is None:
            print("{}{}".format(self.prefix, message))
        else:
            if not Timer.clean:
                print()
                Timer.clean = True
            print("{}\t{}".format(self.prefix, message))
