import json
import importlib
import networkx
import time
from . import analysistask


class Scheduler():

    def __init__(self, dataSet, executor, parameters, schedulerName=None):
        self.dataSet = dataSet
        self.parameters = parameters
        self.executor = executor

        self.analysisTasks = self._parse_parameters()
        self.dependencyGraph = self._parse_dependency_graph()

        self.tasksStarted = []
        self.tasksCompleted = []

    def _parse_parameters(self):
        analysisTasks = {} 
        for tDict in self.parameters['analysis_tasks']:
            analysisModule = importlib.import_module(
                    'merlin.analysis.' + tDict['module'])
            analysisClass = getattr(analysisModule, tDict['task'])
            analysisParameters = tDict.get('parameters')
            analysisName = tDict.get('analysis_name')
            newTask = analysisClass(
                    self.dataSet, analysisParameters, analysisName)
            if newTask.get_analysis_name() in analysisTasks:
                raise Exception('Analysis tasks must have unique names. ' + \
                        newTask.get_analysis_name() + ' is redundant.')
            # TODO This should be more careful to not overwrite an existing
            # analysis task that has already been run.
            newTask.save()
            analysisTasks[newTask.get_analysis_name()] = newTask
        return analysisTasks

    def _parse_dependency_graph(self):
        graph = networkx.DiGraph()
        roots = set()
        for task in self.analysisTasks.values():
            requirements = task.get_dependencies()
            if len(requirements) == 0:
                roots.add(task.get_analysis_name())
            else:
                for r in requirements:
                    graph.add_edge(task.get_analysis_name(), r)
        return graph

    def _start_task(self, taskName):
        print('Starting ' + taskName)
        self.tasksStarted.append(taskName)
        self.executor.run(self.analysisTasks[taskName])

    def start(self):
        self._run_ready_tasks()
        self._begin_master_loop()

    def _begin_master_loop(self, waitTime=30):
        while not self._check_status():
            time.sleep(waitTime)

    def _check_status(self):
        statusChanged = False
        for a in self.tasksStarted:
            if self.analysisTasks[a].is_complete():
                self.tasksStarted.remove(a)
                self.tasksCompleted.append(a)
                statusChanged = True
                print('Completed ' + a)
        if statusChanged:
            self._run_ready_tasks()

        if not statusChanged and not self.tasksStarted:
            return True
        return False

    def _run_ready_tasks(self):
        tasksComplete = {k: a for k,a in self.analysisTasks.items() \
                if a.is_complete()}
        tasksWaiting = {k: a for k,a in self.analysisTasks.items() \
                if not a.is_complete() and not a.is_running()}
        tasksRunning = {k: a for k,a in self.analysisTasks.items() \
                if a.is_running()}
        tasksReady = {k: a for k,a in tasksWaiting.items() \
                if all([a2 in tasksComplete for a2 in self.dependencyGraph[k]])}

        parallelTaskRunning = any(
                [a.is_parallel() for a in tasksRunning.values()])

        for k,a in tasksReady.items():
            if k not in self.tasksStarted:
                if isinstance(a, analysistask.ParallelAnalysisTask):
                    if not parallelTaskRunning:
                        self._start_task(k)
                        parallelTaskRunning = True
                else:
                    self._start_task(k)
