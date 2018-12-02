import os
import dotenv
import errno
import json
import shutil
import pandas
import numpy as np
import re
import csv
import sqlalchemy
import fnmatch
import tifffile
import importlib
import time
import logging

from storm_analysis.sa_library import datareader
import merlin
from merlin.core import analysistask


class DataSet(object):

    def __init__(self, dataDirectoryName, 
            dataName=None, dataHome=None, analysisHome=None):

        if dataHome is None:
            dataHome = merlin.DATA_HOME

        if analysisHome is None:
            analysisHome = merlin.ANALYSIS_HOME

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

    def save_figure(self, analysisTask, figure, figureName):
        savePath = os.sep.join(
                [self.get_analysis_subdirectory(analysisTask, 'figures'),
                    figureName])

        figure.savefig(savePath + '.png', pad_inches=0)
        figure.savefig(savePath + '.pdf', transparent=True, pad_inches=0)

    def get_analysis_image_set(self, analysisTask, imageBaseName, imageIndex):
        return tifffile.imread(self._analysis_image_name(
            analysisTask, imageBaseName, imageIndex))

    def get_analysis_image(self, analysisTask, imageBaseName, imageIndex,
            imagesPerSlice, sliceIndex, frameIndex):
        '''

        Args:
            analysisTask - 
            imageBaseName - 
            imageIndex - the index of the image file
            imagesPerSlice - the number of images in each slice of the image
                file
            sliceIndex - the index of the slice to get the image
            frameIndex - the index of the frame in the specified slice
        '''
        #TODO - It may be useful to add a function that gets all 
        #frames in a slice
        imageFile = tifffile.TiffFile(self._analysis_image_name(
            analysisTask, imageBaseName, imageIndex))
        indexInFile = sliceIndex*imagesPerSlice + frameIndex
        return imageFile.asarray(key=int(indexInFile))
    
    def _writer_for_analysis_images(
            self, analysisTask, imageBaseName, imageIndex=None):
        return tifffile.TiffWriter(self._analysis_image_name(
            analysisTask, imageBaseName, imageIndex), imagej=True)

    def _analysis_tiff_description(self, sliceCount, frameCount):
        imageDescription = {'ImageJ': '1.47a\n',
                'images': sliceCount*frameCount,
                'channels': 1,
                'slices': sliceCount,
                'frames': frameCount,
                'hyperstack': True,
                'loop': False}
        return imageDescription

    def _analysis_image_name(self, analysisTask, imageBaseName, imageIndex):
        destPath = self.get_analysis_subdirectory(
                analysisTask, subdirectory='images')
        if imageIndex is None:
            return os.sep.join([destPath, imageBaseName+'.tif'])
        else:
            return os.sep.join([destPath, imageBaseName+str(imageIndex)+'.tif'])

    def _analysis_result_save_path(self, resultName, analysisName,
            resultIndex=None, subdirectory=None):
        saveName = resultName
        if resultIndex is not None:
            saveName += '_' + str(resultIndex)
        return os.sep.join([self.get_analysis_subdirectory(
            analysisName, subdirectory), saveName])

    def save_dataframe_to_csv(self, dataframe, resultName, analysisTask):
        savePath = self._analysis_result_save_path(
                resultName, analysisTask.get_analysis_name()) + '.csv'

        with open(savePath, 'w') as f:
            dataframe.to_csv(f)

    def save_analysis_result(self, analysisResult, resultName, 
            analysisName, resultIndex=None, subdirectory=None):
        #TODO - only implemented currently for ndarray
        if not isinstance(analysisResult, np.ndarray):
            raise TypeError('analysisResult must be a numpy array')

        savePath = self._analysis_result_save_path(
                resultName, analysisName, resultIndex, subdirectory)
        np.save(savePath, analysisResult)
    
    def load_analysis_result(self, resultName, analysisName, 
            resultIndex=None, subdirectory=None):
        #TODO - This should determine the file extension based on the 
        #files that are present
        savePath = self._analysis_result_save_path(
                resultName, analysisName, resultIndex, subdirectory) + '.npy'
        return np.load(savePath)

    def get_analysis_subdirectory(self, analysisTask, subdirectory=None,
            create=True):
        '''
        analysisTask can either be the class or a string containing the
        class name.

        create - Flag indicating if the analysis subdirectory should be
            created if it does not already exist.
        '''
        if isinstance(analysisTask , analysistask.AnalysisTask):
            analysisName = analysisTask.get_analysis_name()
        else:
            analysisName = analysisTask

        if subdirectory is None:
            subdirectoryPath = os.sep.join(
                    [self.analysisPath, analysisName])
        else:
            subdirectoryPath = os.sep.join(
                    [self.analysisPath, analysisName, \
                            subdirectory])

        if create:
            os.makedirs(subdirectoryPath, exist_ok=True)

        return subdirectoryPath

    def get_task_subdirectory(self, analysisTask):
        return self.get_analysis_subdirectory(
                analysisTask, subdirectory='tasks')

    def get_log_subdirectory(self, analysisTask):
        return self.get_analysis_subdirectory(
                analysisTask, subdirectory='log')
        
    def save_analysis_task(self, analysisTask):
        saveName = os.sep.join([self.get_task_subdirectory(
            analysisTask), 'task.json'])
        
        with open(saveName, 'w') as outFile:
            json.dump(analysisTask.get_parameters(), outFile, indent=4) 

    def load_analysis_task(self, analysisTaskName):
        loadName = os.sep.join([self.get_task_subdirectory(
            analysisTaskName), 'task.json'])

        with open(loadName, 'r') as inFile:
            parameters = json.load(inFile)
            analysisModule = importlib.import_module(parameters['module'])
            analysisTask = getattr(analysisModule, parameters['class'])
            return analysisTask(self, parameters, analysisTaskName)
            
    def delete_analysis(self, analysisTask):
        '''
        Remove all files associated with the provided analysis 
        from this data set.

        Before deleting an analysis task, it must be verified that the
        analysis task is not running.
        '''
        analysisDirectory = self.get_analysis_subdirectory(analysisTask)
        shutil.rmtree(analysisDirectory)

    def analysis_exists(self, analysisTaskName):
        '''
        Determine if an analysis task with the specified name exists in this 
        dataset.
        '''
        analysisPath = self.get_analysis_subdirectory(
                analysisTaskName, create=False)
        return os.path.exists(analysisPath)

    def get_logger(self, analysisTask, fragmentIndex=None):
        loggerName = analysisTask.get_analysis_name()
        if fragmentIndex is not None:
            loggerName += '.' + str(fragmentIndex)

        logger = logging.getLogger(loggerName)
        logger.setLevel(logging.DEBUG)
        fileHandler = logging.FileHandler(
                self._log_path(analysisTask, fragmentIndex))
        fileHandler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        return logger

    def close_logger(self, analysisTask, fragmentIndex=None):
        loggerName = analysisTask.get_analysis_name()
        if fragmentIndex is not None:
            loggerName += '.' + str(fragmentIndex)

        logger = logging.getLogger(loggerName)

        handlerList = list(logger.handlers)
        for handler in handlerList:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

    def _log_path(self, analysisTask, fragmentIndex=None):
        logName = analysisTask.get_analysis_name()
        if fragmentIndex is not None:
            logName += '_' + str(fragmentIndex)
        logName += '.log'

        return os.sep.join([self.get_log_subdirectory(analysisTask), logName])

    def _analysis_status_file(self, analysisTask, eventName, 
            fragmentIndex=None):
        if fragmentIndex is None:
            fileName = analysisTask.get_analysis_name() + '.' + eventName
        else:
            fileName = analysisTask.get_analysis_name() + \
                    '_' + str(fragmentIndex) + '.' + eventName
        return os.sep.join([self.get_task_subdirectory(analysisTask), \
                fileName])

    def record_analysis_started(self, analysisTask, fragmentIndex=None):
        self._record_analysis_event(analysisTask, 'start', fragmentIndex)

    def record_analysis_running(self, analysisTask, fragmentIndex=None):
        self._record_analysis_event(analysisTask, 'run', fragmentIndex)

    def record_analysis_complete(self, analysisTask, fragmentIndex=None):
        self._record_analysis_event(analysisTask, 'done', fragmentIndex)

    def record_analysis_error(self, analysisTask, fragmentIndex=None):
        self._record_analysis_event(analysisTask, 'error', fragmentIndex)

    def _record_analysis_event(
            self, analysisTask, eventName, fragmentIndex=None):    
        fileName = self._analysis_status_file(
                analysisTask, eventName, fragmentIndex)
        with open(fileName, 'w') as f:
            f.write('%s' %time.time())

    def is_analysis_idle(self, analysisTask, fragmentIndex=None):
        fileName = self._analysis_status_file(
                analysisTask, 'run', fragmentIndex)
        return time.time() - os.path.getmtime(fileName) > 120

    def check_analysis_started(self, analysisTask, fragmentIndex=None):
        return self._check_analysis_event(analysisTask, 'start', fragmentIndex)

    def check_analysis_running(self, analysisTask, fragmentIndex=None):
        return self._check_analysis_event(analysisTask, 'run', fragmentIndex)

    def check_analysis_done(self, analysisTask, fragmentIndex=None):
        return self._check_analysis_event(analysisTask, 'done', fragmentIndex)

    def check_analysis_error(self, analysisTask, fragmentIndex=None):
        return self._check_analysis_event(analysisTask, 'error', fragmentIndex)

    def _check_analysis_event(
            self, analysisTask, eventName, fragmentIndex=None):
        fileName = self._analysis_status_file(
                analysisTask, eventName, fragmentIndex)
        return os.path.exists(fileName)

    def get_database_engine(self, analysisTask=None, index=None):
        if analysisTask is None:
            return sqlalchemy.create_engine('sqlite:///' + \
                    os.sep.join([self.analysisPath, 'analysis_data.db']))
        else:
            dbPath = os.sep.join(
                        [self.analysisPath, analysisTask.get_analysis_name(), \
                                'db'])
            os.makedirs(dbPath, exist_ok=True)

            if index is None:
                dbName = 'analysis_data.db'
            else:
                dbName = 'analysis_data' + str(index) + '.db'


            return sqlalchemy.create_engine(os.sep.join(
                        ['sqlite:///' + dbPath, dbName]))


class ImageDataSet(DataSet):

    def __init__(self, dataDirectoryName, dataName=None, dataHome=None, 
            analysisHome=None, microscopeParametersName=None):
        super().__init__(dataDirectoryName, dataName, dataHome, analysisHome)


        if microscopeParametersName is not None:
            self._import_microscope_parameters(microscopeParametersName)
    
        self._load_microscope_parameters()

    def get_image_file_names(self):
        return sorted(
                [os.sep.join([self.rawDataPath, currentFile]) \
                    for currentFile in os.listdir(self.rawDataPath) \
                if currentFile.endswith('.dax') \
                or currentFile.endswith('.tif')])

    def load_image(self, imagePath, frameIndex):
        with datareader.inferReader(imagePath) as reader:
            imageIn = reader.loadAFrame(frameIndex)
            if self.transpose:
                imageIn = np.transpose(imageIn)
            if self.flipHorizontal:
                imageIn = np.flip(imageIn, axis=1)
            if self.flipVertical:
                imageIn = np.flip(imageIn, axis=0)
            return imageIn 

    def _import_microscope_parameters(self, microscopeParametersName):
        sourcePath = os.sep.join([merlin.MICROSCOPE_PARAMETERS_HOME,
                microscopeParametersName + '.json'])
        destPath = os.sep.join(
                [self.analysisPath, 'microscope_parameters.json'])

        shutil.copyfile(sourcePath, destPath) 

    def _load_microscope_parameters(self): 
        path = os.sep.join(
                [self.analysisPath, 'microscope_parameters.json'])
        
        if os.path.exists(path):
            with open(path) as inputFile:
                self.microscopeParameters = json.load(inputFile)
        else:
            self.microscopeParameters = {}

        self.flipHorizontal = self.microscopeParameters.get(
            'flip_horizontal', True)
        self.flipVertical = self.microscopeParameters.get(
            'flip_vertical', False)
        self.transpose = self.microscopeParameters.get('transpose', True)
        self.micronsPerPixel = self.microscopeParameters.get(
                'microns_per_pixel', 106)

class MERFISHDataSet(ImageDataSet):

    def __init__(self, dataDirectoryName, codebookName=None, 
            dataOrganizationName=None, positionFileName=None,
            dataName=None, dataHome=None, analysisHome=None,
            dataParameters=None, microscopeParametersName=None):
        super().__init__(dataDirectoryName, dataName, dataHome, analysisHome, 
                microscopeParametersName)

        if codebookName is not None:
            self._import_codebook(codebookName)

        if dataOrganizationName is not None:
            self._import_dataorganization(dataOrganizationName)

        if positionFileName is not None:
            self._import_positions(positionFileName)

        self._load_dataorganization()
        self._load_codebook()
        self._load_positions()
        self._map_images()
        #TODO - this should check for consistency between all the
        #different components

        self.dataParameters = dataParameters
        #TODO - This should not be hard coded
        self.imageDimensions = (2048, 2048)

    def get_bit_count(self):
        '''Get the number of bits used for MERFISH barcodes in this data set.
        '''

        return len(self.bitNames)

    def get_bit_names(self):
        '''Get the names of the bits for this MERFISH data set.

        Returns:
            A list of the names of the bits in order from the lowest to highest
        '''
        return self.bitNames

    def get_image_dimensions(self):
        '''Get the dimensions of the images in this MERFISH data set.

        Returns:
            A tuple containing the width and height of each image in pixels.
        '''
        return self.imageDimensions

    def get_microns_per_pixel(self):
        '''Get the conversion factor to convert pixels to microns.
        '''

        return self.micronsPerPixel

    def get_fov_offset(self, fov):
        '''Get the offset of the specified fov in the global coordinate system.
        This offset is based on the anticipated stage position.

        Args:
            fov: The fov 
        Returns:
            A tuple specificing the x and y offset of the top right corner 
            of the specified fov in pixels.
        '''
        #TODO - this should be implemented using the position of the fov. 
        return (self.positions.loc[fov]['X'], self.positions.loc[fov]['Y'])

    def get_stage_positions(self):
        return self.positions

    def get_data_channels(self):
        '''Get the data channels for the MERFISH data set.

        Returns:
            A list of the data channels
        '''
        return self.dataOrganization.index

    def get_data_channel_name(self, dataChannelIndex):
        '''Get the name for the data channel with the specified index.
        '''

        return self.dataOrganization['bitName'][dataChannelIndex]


    def get_data_channel_for_bit(self, bitName):
        '''Get the data channel associated with the specified bit.

        Args:
            bitName: the name of the bit to search for
        Returns:
            The associated data channel
        '''
        return self.dataOrganization[\
                self.dataOrganization['bitName'] == bitName].index.item()

    def z_index_to_position(self, zIndex):
        '''Get the z position associated with the provided z index.'''

        return self.get_z_positions()[zIndex]

    def get_z_positions(self):
        '''Get the z positions present in this dataset.

        Returns:
            A sorted list of all unique z positions
        '''
        return(sorted(np.unique(
            [y for x in self.dataOrganization['zPos'] for y in x])))

    def get_image_path(self, imageType, fov, imagingRound):
        selection = self.fileMap[(self.fileMap['imageType'] == imageType) & \
                (self.fileMap['fov'] == fov) & \
                (self.fileMap['imagingRound'] == imagingRound)]

        return selection['imagePath'].values[0]

    def get_fovs(self):
        return np.unique(self.fileMap['fov'])

    def get_imaging_rounds(self):
        return np.unique(self.fileMap['imagingRound'])

    def get_fiducial_filename(self, dataChannel, fov):
        imageType = self.dataOrganization.loc[dataChannel, 'fiducialImageType']
        imagingRound = \
                self.dataOrganization.loc[dataChannel, 'fiducialImagingRound']
        return(self.get_image_path(imageType, fov, imagingRound))

    def get_fiducial_frame(self, dataChannel):
        '''Get the index of the frame containing the fiducial image
        for the specified data channel.
        '''
        return self.dataOrganization.loc[dataChannel, 'fiducialFrame']

    def get_raw_image(self, dataChannel, fov, zPosition):
        channelInfo = self.dataOrganization.loc[dataChannel]
        imagePath = self.get_image_path(
                channelInfo['imageType'], fov, channelInfo['imagingRound'])

        channelZ = channelInfo['zPos']
        if isinstance(channelZ, np.ndarray):
            zIndex = np.where(channelZ == zPosition)[0]
            if len(zIndex) == 0:
                frameIndex = 0
            else:
                frameIndex = zIndex[0]
        else:
            frameIndex = 0

        frames = channelInfo['frame']
        if isinstance(frames, np.ndarray):
            frame = frames[frameIndex]
        else:
            frame = frames

        return self.load_image(imagePath, frame)

    def get_fiducial_image(self, dataChannel, fov):
        channelInfo = self.dataOrganization.loc[dataChannel]
        imagePath = self.get_image_path(
                channelInfo['fiducialImageType'], 
                fov, 
                channelInfo['fiducialImagingRound'])
        frame = channelInfo['fiducialFrame']
        return self.load_image(imagePath, frame)

    def _map_images(self):
        #TODO: This doesn't map the fiducial image types and currently assumes
        #that the fiducial image types and regular expressions are part of the 
        #standard image types.
        #TODO: This doesn't verify that all files are present
        mapPath = os.sep.join([self.analysisPath, 'filemap.csv'])

        if not os.path.exists(mapPath):
            uniqueTypes, uniqueIndexes = np.unique(
                self.dataOrganization['imageType'], return_index=True)

            fileNames = self.get_image_file_names()
            fileData = []
            for currentType, currentIndex in zip(uniqueTypes, uniqueIndexes):
                matchRE = re.compile(
                        self.dataOrganization.imageRegExp[currentIndex])


                for currentFile in fileNames:
                    matchedName = matchRE.match(os.path.split(currentFile)[-1])
                    if matchedName is not None:
                        transformedName = matchedName.groupdict()
                        if transformedName['imageType'] == currentType:
                            if 'imagingRound' not in transformedName:
                                transformedName['imagingRound'] = -1
                            transformedName['imagePath'] = currentFile
                            fileData.append(transformedName)
        
            self.fileMap = pandas.DataFrame(fileData)
            self.fileMap[['imagingRound', 'fov']] = \
                    self.fileMap[['imagingRound', 'fov']].astype(int)
    
            self.fileMap.to_csv(mapPath)

        else:
            self.fileMap = pandas.read_csv(mapPath)

    def get_codebook(self):
        return self.codebook

    def _parse_list(self, inputString, dtype=float):
        return np.fromstring(inputString.strip('[]'), dtype=dtype, sep=',')

    def _parse_int_list(self, inputString):
        return self._parse_list(inputString, dtype=int)

    def _parse_barcode_from_string(self, inputString):
        return np.array([int(x) for x in inputString if x is not ' '])

    def _import_codebook(self, codebookName):
        sourcePath = os.sep.join([merlin.CODEBOOK_HOME,
                codebookName + '_codebook.csv'])
        destPath = os.sep.join([self.analysisPath, 'codebook.csv'])

        shutil.copyfile(sourcePath, destPath)    

    def _load_codebook(self):
        path = os.sep.join([self.analysisPath, 'codebook.csv'])

        headerLength = 3
        self.codebook = pandas.read_csv(
                path, header=headerLength, skipinitialspace=True,
                usecols=['name','id','barcode'],
                converters={'barcode': self._parse_barcode_from_string})
        with open(path, 'r') as inFile:
            csvReader = csv.reader(inFile, delimiter=',')
            header = [row for i,row in enumerate(csvReader) if i<headerLength]
        self.bitNames = [x.strip() for x in header[2][1:]]

    def _load_positions(self):
        positionPath = os.sep.join([self.analysisPath, 'positions.csv'])
        #TODO - this is messy searching for the position file
        #TODO - I should check to make sure the number of positions 
        # matches the number of FOVs
        if not os.path.exists(positionPath):
            for f in os.listdir(self.rawDataPath):
                if fnmatch.fnmatch(f, '*position*'):
                    shutil.copyfile(
                            os.sep.join([self.rawDataPath, f]), positionPath)
        
        if not os.path.exists(positionPath):
            for f in os.listdir(os.sep.join([self.rawDataPath, '..'])):
                if fnmatch.fnmatch(f, '*position*'):
                    shutil.copyfile(
                            os.sep.join([self.rawDataPath, '..', f]), 
                            positionPath)

        self.positions = pandas.read_csv(positionPath, header=None,
                names=['X','Y'])

    def _import_positions(self, positionFileName):
        sourcePath = os.sep.join([merlin.POSITION_HOME, \
                positionFileName + '.csv'])
        destPath = os.sep.join([self.analysisPath, 'positions.csv'])
            
        shutil.copyfile(sourcePath, destPath)    

    def _convert_parameter_list(self, listIn, castFunction, delimiter=';'):
        return [castFunction(x) for x in listIn.split(delimiter) if len(x)>0]

    def _import_dataorganization(self, dataOrganizationName):
        sourcePath = os.sep.join([merlin.DATA_ORGANIZATION_HOME, \
                dataOrganizationName + '.csv'])
        destPath = os.sep.join([self.analysisPath, 'dataorganization.csv'])
            
        shutil.copyfile(sourcePath, destPath)    

    def _load_dataorganization(self):
        path = os.sep.join([self.analysisPath, 'dataorganization.csv'])
        self.dataOrganization = pandas.read_csv(
                path, 
                converters={'frame': self._parse_int_list,
                    'zPos': self._parse_list})

