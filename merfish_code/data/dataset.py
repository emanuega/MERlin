import os
import dotenv
import errno
import cPickle as pickle


class DataSet(object):

    def __init__(self, dataDirectoryName, 
            dataName=None, dataHome=None, analysisHome=None):

        dotenvPath = dotenv.find_dotenv()
        dotenv.load_dotenv(dotenvPath)

        if dataHome is None:
            dataHome = os.environ.get('DATA_HOME')

        if analysisHome is None:
            analysisHome = os.environ.get('ANALYSIS_HOME')

        self.rawDataPath = os.sep.join([dataHome, dataDirectoryName])
        if not os.path.isdir(self.rawDataPath):
            raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), self.rawDataPath)
            
        self.analysisPath = os.sep.join([analysisHome, dataDirectoryName])
        os.makedirs(self.analysisPath, exist_ok=True)

        self.logPath = os.sep.join([self.analysisPath, 'logs'])
        os.makedirs(self.logPath, exist_ok=True)

        self.figurePath = os.sep.join([self.analysisPath, 'figures'])
        os.makedirs(self.figurePath, exist_ok=True)

    def save_figure(figure, figureName, subDirectory=None):

        if subDirectory is not None:
            savePath = os.sep.join([self.figurePath, subDirectory, figureName])
            os.makedirs(savePath, exist_ok=True)
        else:
            savePath = os.sep.join([self.figurePath, figureName])

        figure.savefig(savePath + '.png', pad_inches=0)
        figure.savefig(savePath + '.pdf', transparent=True, pad_inches=0)

    def get_analysis_subdirectory(self, analysisName):
        subdirectoryPath = os.sep.join(
                [self.analysisPath, subdirectoryName])
        os.makedirs(subdirectoryPath, exist_ok=True)

        return subdirectoryPath

    def get_task_subdirectory(self, analysisName):
        taskDirectoryPath = os.sep.join(
                [self.get_analysis_subdirectory(analysisName), 'tasks'])
        os.makedirs(taskDirectoryPath, exist_ok=True)

        return taskDirectoryPath
        
    def save_analysis_task(self, analysisTask):
        saveName = os.sep.join([get_task_subdirectory(
            analysisTask.get_analysis_name()), 'task.pkl'])
        
        with open(saveName, 'w') as outFile:
            pickle.dump(analysisTask, outFile)

    def load_analysis_task(self, analysisTaskName):
        loadName = os.sep.join([get_task_subdirectory(
            analysisTask.get_analysis_name()), 'task.pkl'])

        with open(loadName, 'r') as inFile:
            return pickle.load(loadName)

    def record_analysis_running(self, analysisTask, fragmentIndex=None):
        self._record_analysis_event(analysisTask, 'run', fragmentIndex)

    def record_analysis_complete(self, analysisTask, fragmentIndex=None):
        self._record_analysis_event(analysisTask, 'done', fragmentIndex)

    def _record_analysis_event(
            self, analysisTask, eventName, fragmentIndex=None):    
        if fragmentIndex is None:
            fileName = analysisTask.get_analysis_name() + '.' + eventName
        else:
            fileName = analysisTask.get_analysis_name() + \
                    '_' + str(fragmentIndex) + '.' + eventName

        fullName = os.sep.join([get_task_subdirectory(
            analysisTask.get_analysis_name()), fileName])
        open(fullName, 'a').close()

    def check_analysis_running(self, analysisTask, fragmentIndex=None):
        return self._check_analysis_event(analysisTask, 'run', fragmentIndex)

    def check_analysis_done(self, analysisTask, fragmentIndex=None):
        return self._check_analysis_event(analysisTask, 'done', fragmentIndex)

    def _check_analysis_event(
            self, analysisTask, eventName, fragmentIndex=None):
        if fragmentIndex is None:
            fileName = analysisTask.get_analysis_name() + '.' + eventName
        else:
            fileName = analysisTask.get_analysis_name() + \
                    '_' + str(fragmentIndex) + '.' + eventName
    
        fullName = os.sep.join([get_task_subdirectory(
            analysisTask.get_analysis_name()), fileName])
        return os.paith.exists(fullName)

class ImageDataSet(DataSet):

    def __init__(self, dataDirectoryName, 
            dataName=None, dataHome=None, analysisHome=None):
        super().__init__(dataDirectoryName, dataName, dataHome, analysisHome)

    def get_image_files(self):
        return sorted(
                [os.sep.join([self.rawDataPath, currentFile]) \
                    for currentFile in os.listdir(self.rawDataPath) \
                if currentFile.endswith('.dax') \
                or currentFile.endswith('.tif')])



class MERFISHDataSet(ImageDataSet):

    def __init__(self, dataDirectoryName, codebookName=None, 
            dataOrganizationName=None,
            dataName=None, dataHome=None, analysisHome=None):
        super().__init__(dataDirectoryName, dataName, dataHome, analysisHome)

        if codebookName is not None:
            self._import_codebook(codebookName)

        if dataOrganizationname is not None:
            self._import_dataorganization(dataOrganizationName)

    def _import_codebook(self, codebookName):
        pass

    def _import_dataorganization(self, dataOrganizationName):
        pass

