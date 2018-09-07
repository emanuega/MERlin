from abc import ABC, abstractmethod
import multiprocessing

from . import analysistask

class Executor(object):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def run(self):
        pass 


class LocalExecutor(Executor):

    def __init__(self, coreCount=None):
        super().__init__()

        if coreCount is None:
            self.coreCount = int(multiprocessing.cpu_count()*0.7)
        else:
            self.coreCount = coreCount

    def run(self, task, callback=None):
        if isinstance(task, analysistask.ParallelAnalysisTask):
            pool = multiprocessing.Pool(processes=self.coreCount)
            pool.map_async(
                    task.run, range(task.fragment_count()),
                    callback=callback)

        else:
            pool = multiprocessing.Pool(processes=1)
            pool.apply_async(task.run, callback=callback)