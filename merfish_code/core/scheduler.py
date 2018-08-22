import json
import importlib
import networkx
from . import analysistask

class Scheduler():

    def __init__(self, dataSet, executor, parameters, scheduleName=None):
        self.dataSet = dataSet
        self.parameters = parameters
        self.executor = executor

        self.analysisTasks = self._parse_parameters()
        self.dependencyGraph = self._parse_dependency_graph()

        self.tasksStarted = []

    def _parse_parameters(self):
        analysisTasks = {} 
        for tDict in self.parameters['analysis_tasks']:
            analysisModule = importlib.import_module(
                    'merfish_code.analysis.' + tDict['module'])
            analysisClass = getattr(analysisModule, tDict['task'])
            analysisParameters = tDict.get('parameters')
            analysisName = tDict.get('analysis_name')
            newTask = analysisClass(
                    self.dataSet, analysisParameters, analysisName)
            if newTask.get_analysis_name() in analysisTasks:
                raise Exception('Analysis tasks must have unique names. ' + \
                        newTask.get_analysis_name() + ' is redundant.')
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

    def _on_done(self, result):
        self.run()

    def _start_task(self, taskName):
        self.tasksStarted.append(taskName)
        self.executor.run(self.analysisTasks[taskName], callback=self._on_done)

    def run(self):
        tasksComplete = {k: a for k,a in self.analysisTasks.items() \
                if a.is_complete()}
        tasksWaiting = {k: a for k,a in self.analysisTasks.items() \
                if not a.is_complete() and not a.is_running()}
        tasksRunning = {k: a for k,a in self.analysisTasks.items() \
                if a.is_running()}
        tasksReady = {k: a for k,a in tasksWaiting.items() \
                if all([a2 in tasksComplete for a2 in self.dependencyGraph[k]])}

        parallelTaskRunning = any([isinstance(
            a, analysistask.ParallelAnalysisTask) \
                    for a in tasksRunning.values()])

        for k,a in tasksReady.items():
            if k not in self.tasksStarted:
                if isinstance(a, analysistask.ParallelAnalysisTask):
                    if not parallelTaskRunning:
                        self._start_task(k)
                        parallelTaskRunning = True
                else:
                    self._start_task(k)

