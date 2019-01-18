from abc import abstractmethod
import multiprocessing
import threading

from merlin.core import analysistask


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

    def run(self, task, index=None, callback=None, join=False):
        pool = None
        thread = None
        if isinstance(task, analysistask.ParallelAnalysisTask):
            pool = multiprocessing.Pool(processes=self.coreCount)
            pool.map_async(
                    task.run, range(task.fragment_count()),
                    callback=callback)
        elif isinstance(task, analysistask.InternallyParallelAnalysisTask) \
                and index is None:
            task.set_core_count(self.coreCount)
            thread = threading.Thread(target=task.run)
            thread.daemon = True
            thread.start()
        elif isinstance(task, analysistask.InternallyParallelAnalysisTask) \
                and index is not None:
            pool = multiprocessing.Pool(processes=1)
            pool.apply_async(task.run, index, callback=callback)
        else:
            pool = multiprocessing.Pool(processes=1)
            pool.apply_async(task.run, callback=callback)

        if join:
            if pool:
                pool.close()
                pool.join()
            if thread:
                thread.join()
