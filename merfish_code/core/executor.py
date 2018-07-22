from abc import ABC, abstractmethod
import multiprocessing

from . import analysistask

class Executor(object):

    def __init__(self, task):
        self.task = task
    
    @abstractmethod
    def run(self):
        pass 

    def is_complete(self):
        return self.task.is_complete()


class LocalExecutor(Executor):

    def __init__(self, task):
        super().__init__(task)

    def run(self):
        if isinstance(self.task, analysistask.ParallelAnalysisTask):
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            pool.map_async(
                    self.task.run, range(self.task.fragment_count()))
            print('done with pool')

        else:
            process = multiprocessing.Process(target=self.task.run)
            process.start()
