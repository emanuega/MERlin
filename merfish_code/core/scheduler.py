import json
import importlib

class Scheduler():

    def __init__(self, dataSet, parameters, scheduleName=None):
        self.dataSet = dataSet
        self.parameters = parameters

        self._parse_parameters()

    def _parse_parameters(self):
        self.analysisTasks = []
        for tDict in parameters['analysis_tasks']:
            analysisModule = importlib.import_module(
                    'merfish_code.analysis.' + tDict['module'])
            analysisClass = getattr(analysisModule, tDict['task'])
            analysisParameters = tDict.get('parameters')
            analysisName = tDict.get('analysis_name')
            self.analysisTasks.append(
                    analysisClass(dataSet, analysisParameters, analysisName))




