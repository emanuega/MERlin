import os
import dotenv
import errno

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

