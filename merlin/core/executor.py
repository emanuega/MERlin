from abc import abstractmethod
import multiprocessing
import threading
from typing import Callable

from merlin.core import analysistask


class Executor(object):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def run(self, task: analysistask.AnalysisTask, index: int=None,
            callback: Callable=None, join: bool=False,
            rerunCompleted: bool=False) -> None:
        """Run an analysis task.

        This method will not run analysis tasks that are already currently
        running and analysis is terminated early due to error or otherwise
        will not be restarted.

        Args:
            task: the analysis task to run.
            index: index of the analysis to run for a parallel analysis task.
            callback: function ta call upon completion of the analysis.
            join: flag indicating if the execution threads should be joined to
                the current thread. If True, this function doesn't return
                until the analysis is complete.
            rerunCompleted: flag indicating if previous analysis should be
                run again even if it has previously completed. If overwrite
                is True, analysis will be run on the task regardless of its
                status. If overwrite is False, analysis will only be run on
                the task or fragments of the task that have either not been
                started or have previously completed in error.
        """
        pass


class LocalExecutor(Executor):

    def __init__(self, coreCount=None):
        super().__init__()

        if coreCount is None:
            self.coreCount = int(multiprocessing.cpu_count()*0.7)
        else:
            self.coreCount = coreCount

    def run(self, task: analysistask.AnalysisTask, index: int=None,
            callback: Callable=None, join: bool=False,
            rerunCompleted: bool=False) -> None:
        if task.is_complete() and not rerunCompleted:
            return

        pool = None
        thread = None
        if isinstance(task, analysistask.ParallelAnalysisTask):
            pool = multiprocessing.Pool(processes=self.coreCount)
            fragmentsToRun = [x for x in range(task.fragment_count())
                              if (rerunCompleted and not task.is_running(x))
                              or (not task.is_complete(x)
                                  and not task.is_running(x))]
            pool.map_async(task.run, fragmentsToRun, callback=callback)

        elif isinstance(task, analysistask.InternallyParallelAnalysisTask):
            if task.is_running():
                return
            task.set_core_count(self.coreCount)
            thread = threading.Thread(target=task.run)
            thread.daemon = True
            thread.start()

        else:
            if task.is_running():
                return
            pool = multiprocessing.Pool(processes=1)
            pool.apply_async(task.run, callback=callback)

        if join:
            if pool:
                pool.close()
                pool.join()
            if thread:
                thread.join()
