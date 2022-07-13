import logging
import time
import datetime

def getFileNameBasedTime():
    return datetime.datetime.now().__str__().replace(':', '_')

def saveStr2File(filename, contents):
    fh = open(filename, 'w')
    fh.write(contents)
    fh.close()

def getStrFromFile(filename):
    fh = open(filename, 'r')
    content = fh.readline().strip()
    fh.close()
    return content

class Logger(object):
    """
    set logger

        https://www.cnblogs.com/CJOKER/p/8295272.html

    """

    def __init__(self, logger_path):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logfile = logging.FileHandler(logger_path)
        #
        self.logfile.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(
            '%(asctime)s -%(filename)s:%(lineno)s - %(levelname)s - %(message)s')
        self.logfile.setFormatter(formatter)
        self.logdisplay = logging.StreamHandler()
        #
        self.logdisplay.setLevel(logging.DEBUG)
        self.logdisplay.setFormatter(formatter)
        self.logger.addHandler(self.logfile)
        self.logger.addHandler(self.logdisplay)

    def get_logger(self):
        return self.logger


class Timer:
    last_time = 0
    current_time = 0
    def __init__(self):
        self.last_time = self.current_time = time.perf_counter()
        return

    def clear(self):
        self.last_time = self.current_time
        
    def get_run_time(self, desc=''):
        self.current_time = time.perf_counter()
        print(desc, ' + ' , self.current_time-self.last_time)
        self.last_time = self.current_time
        return
