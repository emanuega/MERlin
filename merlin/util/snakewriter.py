import importlib
import networkx

from merlin.core import analysistask
from merlin.core import dataset


class SnakemakeRule(object):

    def __init__(self, analysisTask: analysistask.AnalysisTask,
                 pythonPath=None):
        self._analysisTask = analysisTask
        self._pythonPath = pythonPath

    @staticmethod
    def _add_quotes(stringIn):
        return '\'%s\'' % stringIn

    @staticmethod
    def _clean_string(stringIn):
        return stringIn.replace('\\', '/')

    def _expand_as_string(self, taskName, indexCount) -> str:
        return 'expand(%s, g=list(range(%i)))' % (self._add_quotes(
            self._analysisTask.dataSet.analysis_done_filename(taskName, '{g}')),
            indexCount)


    def _generate_input_names(self, task):
        if isinstance(task, analysistask.ParallelAnalysisTask):
            return self._clean_string(self._expand_as_string(
                task.get_analysis_name(), task.fragment_count()))
        else:
            return self._clean_string(self._add_quotes(
                self._analysisTask.dataSet.analysis_done_filename(task)))

    def _generate_input(self) -> str:
        inputTasks = [self._analysisTask.dataSet.load_analysis_task(x)
                      for x in self._analysisTask.get_dependencies()]
        inputString = ','.join([self._generate_input_names(x)
                                for x in inputTasks])
        return self._clean_string(inputString)

    def _generate_done_input(self) -> str:
        inputTasks = [self._analysisTask.dataSet.load_analysis_task(x)
                      for x in self._analysisTask.get_dependencies()]
        inputDone= ','.join([self._generate_done_output(x) for x in inputTasks])

        return inputDone

    def _generate_output(self) -> str:
        if isinstance(self._analysisTask, analysistask.ParallelAnalysisTask):
            return self._clean_string(
                self._add_quotes(
                    self._analysisTask.dataSet.analysis_done_filename(
                        self._analysisTask, '{i}')))
        else:
            return self._clean_string(
                self._add_quotes(
                    self._analysisTask.dataSet.analysis_done_filename(
                        self._analysisTask)))

    def _generate_done_output(self,
                         analysisTask: analysistask.AnalysisTask = None) -> str:
        if analysisTask is None:
            analysisTask = self._analysisTask

        return self._clean_string(
            self._add_quotes(
                analysisTask.dataSet.analysis_full_completion_filename(
                    analysisTask)))

    def _generate_touch_output(self,
                         analysisTask: analysistask.AnalysisTask = None) -> str:
        if analysisTask is None:
            analysisTask = self._analysisTask

        return self._clean_string(
            analysisTask.dataSet.analysis_full_completion_filename(
                    analysisTask))


    def _generate_message(self) -> str:
        messageString = \
            ''.join(['Running ', self._analysisTask.get_analysis_name()])

        if isinstance(self._analysisTask, analysistask.ParallelAnalysisTask):
            messageString += ' {wildcards.i}'

        return self._add_quotes(messageString)

    def _generate_shell(self) -> str:
        if self._pythonPath is None:
            shellString = 'python '
        else:
            shellString = self._clean_string(self._pythonPath) + ' '
        shellString += ''.join(
            ['-m merlin -t ',
             self._clean_string(self._analysisTask.analysisName),
             ' -e \"',
             self._clean_string(self._analysisTask.dataSet.dataHome), '\"',
             ' -s \"',
             self._clean_string(self._analysisTask.dataSet.analysisHome),
             '\"'])
        if isinstance(self._analysisTask, analysistask.ParallelAnalysisTask):
            shellString += ' -i {wildcards.i}'
        shellString += ' ' + self._clean_string(
            self._analysisTask.dataSet.dataSetName)

        return self._add_quotes(shellString)

    def as_string(self) -> str:
        fullString = ('rule %s:\n\tinput: %s\n\toutput: %s\n\tmessage: %s\n\t'
                      + 'shell: %s\n\n') \
                      % (self._analysisTask.get_analysis_name(),
                         self._generate_done_input(), self._generate_output(),
                         self._generate_message(),  self._generate_shell())
        fullString += ('rule %sDone:\n\tinput: %s\n\toutput: %s\n\t' +
                       'shell: \'touch %s\'\n') \
                      % (self._analysisTask.get_analysis_name(),
                         self._generate_input_names(self._analysisTask),
                         self._generate_done_output(self._analysisTask),
                         self._generate_touch_output(self._analysisTask))

        return fullString

    def full_output(self) -> str:
        return self._generate_input_names(self._analysisTask)

class SnakefileGenerator(object):

    def __init__(self, analysisParameters, dataSet: dataset.DataSet,
                 pythonPath: str=None):
        self._analysisParameters = analysisParameters
        self._dataSet = dataSet
        self._pythonPath = pythonPath

    def _parse_parameters(self):
        analysisTasks = {}
        for tDict in self._analysisParameters['analysis_tasks']:
            analysisModule = importlib.import_module(tDict['module'])
            analysisClass = getattr(analysisModule, tDict['task'])
            analysisParameters = tDict.get('parameters')
            analysisName = tDict.get('analysis_name')
            newTask = analysisClass(
                    self._dataSet, analysisParameters, analysisName)
            if newTask.get_analysis_name() in analysisTasks:
                raise Exception('Analysis tasks must have unique names. ' +
                                newTask.get_analysis_name() + ' is redundant.')
            # TODO This should be more careful to not overwrite an existing
            # analysis task that has already been run.
            newTask.save()
            analysisTasks[newTask.get_analysis_name()] = newTask
        return analysisTasks

    def _identify_terminal_tasks(self, analysisTasks):
        taskGraph = networkx.DiGraph()
        for x in analysisTasks.keys():
            taskGraph.add_node(x)

        for x, a in analysisTasks.items():
            for d in a.get_dependencies():
                taskGraph.add_edge(d, x)

        return [k for k, v in taskGraph.out_degree if v == 0
                and not analysisTasks[k].is_complete()]

    def generate_workflow(self) -> str:
        """Generate a snakemake workflow for the analysis parameters
        of this SnakemakeGenerator and save the workflow into the dataset.

        Returns:
            the path to the generated snakemake workflow
        """
        analysisTasks = self._parse_parameters()
        ruleList = {k: SnakemakeRule(v, self._pythonPath)
                    for k, v in analysisTasks.items() if not v.is_complete()}

        terminalTasks = self._identify_terminal_tasks(analysisTasks)
        workflowString = 'rule all: \n\tinput: ' + \
            ','.join([ruleList[x]._generate_done_output()
                      for x in terminalTasks]) + '\n\n'
        workflowString += '\n'.join([x.as_string() for x in ruleList.values()])

        return self._dataSet.save_workflow(workflowString)
