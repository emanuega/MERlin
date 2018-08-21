import json
import importlib
import networkx

class Scheduler():

    def __init__(self, dataSet, parameters, scheduleName=None):
        self.dataSet = dataSet
        self.parameters = parameters

        self._parse_parameters()

    def _parse_parameters(self):
        self.analysisTasks = {} 
        for tDict in self.parameters['analysis_tasks']:
            analysisModule = importlib.import_module(
                    'merfish_code.analysis.' + tDict['module'])
            analysisClass = getattr(analysisModule, tDict['task'])
            analysisParameters = tDict.get('parameters')
            analysisName = tDict.get('analysis_name')
            newTask = analysisClass(
                    self.dataSet, analysisParameters, analysisName)
            if newTask.get_analysis_name() in self.analysisTasks:
                raise Exception('Analysis tasks must have unique names. ' + \
                        newTask.get_analysis_name() + ' is redundant.')
            self.analysisTasks[newTask.get_analysis_name()] = newTask

    def _parse_dependency_graph(self):
        graph = networkx.DiGraph()
        roots = set()
        for task in self.analysisTasks.values():
            requirements = task.get_dependencies()
            if len(requirements) == 0:
                roots.add(task.get_analysis_name())
            else:
                for r in requirements:
                    graph.add_edge(r, task.get_analysis_name())
        return graph, roots
        
