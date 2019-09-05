import datetime
def TicTocGenerator():
    tf = datetime.datetime.now()
    while True:
        ti = tf
        tf = datetime.datetime.now()
        yield (tf - ti).total_seconds()

class TicToc(object):
    def __init__(self):
        self.tictoc = TicTocGenerator()

    def toc(self):
        return next(self.tictoc) * 1000.0

    def tic(self):
        next(self.tictoc)
        