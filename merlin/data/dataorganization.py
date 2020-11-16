import os
import re
from typing import List
from typing import Tuple
import pandas
import numpy as np

import merlin
from merlin.core import dataset


def _parse_list(inputString: str, dtype=float):
    if ',' in inputString:
        return np.fromstring(inputString.strip('[] '), dtype=dtype, sep=',')
    else:
        return np.fromstring(inputString.strip('[] '), dtype=dtype, sep=' ')


def _parse_int_list(inputString: str):
    return _parse_list(inputString, dtype=int)


class InputDataError(Exception):
    pass


class DataOrganization(object):

    """
    A class to specify the organization of raw images in the original
    image files.
    """

    def __init__(self, dataSet, filePath: str = None):
        """
        Create a new DataOrganization for the data in the specified data set.

        If filePath is not specified, a previously stored DataOrganization
        is loaded from the dataSet if it exists. If filePath is specified,
        the DataOrganization at the specified filePath is loaded and
        stored in the dataSet, overwriting any previously stored
        DataOrganization.

        Raises:
            InputDataError: If the set of raw data is incomplete or the
                    format of the raw data deviates from expectations.
        """

        self._dataSet = dataSet

        if filePath is not None:
            if not os.path.exists(filePath):
                filePath = os.sep.join(
                        [merlin.DATA_ORGANIZATION_HOME, filePath])

            self.data = pandas.read_csv(
                filePath,
                converters={'frame': _parse_int_list, 'zPos': _parse_list})
            self.data['readoutName'] = self.data['readoutName'].str.strip()
            self._dataSet.save_dataframe_to_csv(
                    self.data, 'dataorganization', index=False)

        else:
            self.data = self._dataSet.load_dataframe_from_csv(
                'dataorganization',
                converters={'frame': _parse_int_list, 'zPos': _parse_list})

        stringColumns = ['readoutName', 'channelName', 'imageType',
                         'imageRegExp', 'fiducialImageType', 'fiducialRegExp']
        self.data[stringColumns] = self.data[stringColumns].astype('str')
        self._map_image_files()

    def get_data_channels(self) -> np.array:
        """Get the data channels for the MERFISH data set.

        Returns:
            A list of the data channel indexes
        """
        return np.array(self.data.index)

    def get_data_channel_readout_name(self, dataChannelIndex: int) -> str:
        """Get the name for the data channel with the specified index.

        Args:
            dataChannelIndex: The index of the data channel
        Returns:
            The name of the specified data channel
        """
        return self.data.iloc[dataChannelIndex]['readoutName']

    def get_data_channel_name(self, dataChannelIndex: int) -> str:
        """Get the name for the data channel with the specified index.

        Args:
            dataChannelIndex: The index of the data channel
        Returns:
            The name of the specified data channel,
            primarily relevant for non-multiplex measurements
        """
        return self.data.iloc[dataChannelIndex]['channelName']

    def get_data_channel_index(self, dataChannelName: str) -> int:
        """Get the index for the data channel with the specified name.

        Args:
            dataChannelName: the name of the data channel. The data channel
                name is not case sensitive.
        Returns:
            the index of the data channel where the data channel name is
                dataChannelName
        Raises:
            # TODO this should raise a meaningful exception if the data channel
            # is not found
        """
        return self.data[self.data['channelName'].apply(
            lambda x: str(x).lower()) == str(dataChannelName).lower()]\
            .index.values.tolist()[0]

    def get_data_channel_color(self, dataChannel: int) -> str:
        """Get the color used for measuring the specified data channel.

        Args:
            dataChannel: the data channel index
        Returns:
            the color for this data channel as a string
        """
        return str(self.data.at[dataChannel, 'color'])

    def get_data_channel_for_bit(self, bitName: str) -> int:
        """Get the data channel associated with the specified bit.

        Args:
            bitName: the name of the bit to search for
        Returns:
            The index of the associated data channel
        """
        return self.data[self.data['readoutName'] ==
                         bitName].index.values.item()

    def get_data_channel_with_name(self, channelName: str) -> int:
        """Get the data channel associated with a gene name.

        Args:
            channelName: the name of the gene to search for
        Returns:
            The index of the associated data channel
        """
        return self.data[self.data['channelName'] ==
                         channelName].index.values.item()

    def get_fiducial_filename(self, dataChannel: int, fov: int) -> str:
        """Get the path for the image file that contains the fiducial
        image for the specified dataChannel and fov.

        Args:
            dataChannel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """

        imageType = self.data.loc[dataChannel, 'fiducialImageType']
        imagingRound = \
            self.data.loc[dataChannel, 'fiducialImagingRound']
        return self._get_image_path(imageType, fov, imagingRound)

    def get_fiducial_frame_index(self, dataChannel: int) -> int:
        """Get the index of the frame containing the fiducial image
        for the specified data channel.

        Args:
            dataChannel: index of the data channel
        Returns:
            The index of the fiducial frame in the corresponding image file
        """
        return self.data.iloc[dataChannel]['fiducialFrame']

    def get_image_filename(self, dataChannel: int, fov: int) -> str:
        """Get the path for the image file that contains the
        images for the specified dataChannel and fov.

        Args:
            dataChannel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """
        channelInfo = self.data.iloc[dataChannel]
        imagePath = self._get_image_path(
                channelInfo['imageType'], fov, channelInfo['imagingRound'])
        return imagePath

    def get_image_frame_index(self, dataChannel: int, zPosition: float) -> int:
        """Get the index of the frame containing the image
        for the specified data channel and z position.

        Args:
            dataChannel: index of the data channel
            zPosition: the z position
        Returns:
            The index of the frame in the corresponding image file
        """
        channelInfo = self.data.iloc[dataChannel]
        channelZ = channelInfo['zPos']
        if isinstance(channelZ, np.ndarray):
            zIndex = np.where(channelZ == zPosition)[0]
            if len(zIndex) == 0:
                raise Exception('Requested z position not found. Position ' +
                                'z=%0.2f not found for channel %i'
                                % (zPosition, dataChannel))
            else:
                frameIndex = zIndex[0]
        else:
            frameIndex = 0

        frames = channelInfo['frame']
        if isinstance(frames, np.ndarray):
            frame = frames[frameIndex]
        else:
            frame = frames

        return frame

    def get_z_positions(self) -> List[float]:
        """Get the z positions present in this data organization.

        Returns:
            A sorted list of all unique z positions
        """
        return sorted(np.unique([y for x in self.data['zPos'] for y in x]))

    def get_fovs(self) -> np.ndarray:
        return np.unique(self.fileMap['fov'])

    def get_sequential_rounds(self) -> Tuple[List[int], List[str]]:
        """ Get the rounds that are not present in your codebook

        Returns:
            A tuple of two lists, the first list contains the channel number
            for all the rounds not contained in the codebook, the second list
            contains the name associated with that channel in the data
            organization file.
        """
        multiplexBits = {b for x in self._dataSet.get_codebooks()
                         for b in x.get_bit_names()}
        sequentialChannels = [i for i in self.get_data_channels()
                              if self.get_data_channel_readout_name(i)
                              not in multiplexBits]
        sequentialGeneNames = [self.get_data_channel_name(x) for
                               x in sequentialChannels]
        return sequentialChannels, sequentialGeneNames

    def _get_image_path(
            self, imageType: str, fov: int, imagingRound: int) -> str:
        selection = self.fileMap[(self.fileMap['imageType'] == imageType) &
                                 (self.fileMap['fov'] == fov) &
                                 (self.fileMap['imagingRound'] == imagingRound)]
        filemapPath = selection['imagePath'].values[0]
        return os.path.join(self._dataSet.dataHome, self._dataSet.dataSetName,
                            filemapPath)

    def _truncate_file_path(self, path) -> None:
        head, tail = os.path.split(path)
        return tail

    def _map_image_files(self) -> None:
        # TODO: This doesn't map the fiducial image types and currently assumes
        # that the fiducial image types and regular expressions are part of the
        # standard image types.

        try:
            self.fileMap = self._dataSet.load_dataframe_from_csv('filemap')
            self.fileMap['imagePath'] = self.fileMap['imagePath'].apply(
                self._truncate_file_path)

        except FileNotFoundError:
            uniqueEntries = self.data.drop_duplicates(
                subset=['imageType', 'imageRegExp'], keep='first')

            uniqueTypes = uniqueEntries['imageType']
            uniqueIndexes = uniqueEntries.index.values.tolist()

            fileNames = self._dataSet.get_image_file_names()
            if len(fileNames) == 0:
                raise dataset.DataFormatException(
                    'No image files found at %s.' % self._dataSet.rawDataPath)
            fileData = []
            for currentType, currentIndex in zip(uniqueTypes, uniqueIndexes):
                matchRE = re.compile(
                        self.data.imageRegExp[currentIndex])

                matchingFiles = False
                for currentFile in fileNames:
                    matchedName = matchRE.match(os.path.split(currentFile)[-1])
                    if matchedName is not None:
                        transformedName = matchedName.groupdict()
                        if transformedName['imageType'] == currentType:
                            if 'imagingRound' not in transformedName:
                                transformedName['imagingRound'] = -1
                            transformedName['imagePath'] = currentFile
                            matchingFiles = True
                            fileData.append(transformedName)

                if not matchingFiles:
                    raise dataset.DataFormatException(
                        'Unable to identify image files matching regular '
                        + 'expression %s for image type %s.'
                        % (self.data.imageRegExp[currentIndex],
                           currentType))

            self.fileMap = pandas.DataFrame(fileData)
            self.fileMap[['imagingRound', 'fov']] = \
                self.fileMap[['imagingRound', 'fov']].astype(int)
            self.fileMap['imagePath'] = self.fileMap['imagePath'].apply(
                self._truncate_file_path)

            self._validate_file_map()

            self._dataSet.save_dataframe_to_csv(
                    self.fileMap, 'filemap', index=False)

    def _validate_file_map(self) -> None:
        """
        This function ensures that all the files specified in the file map
        of the raw images are present.

        Raises:
            InputDataError: If the set of raw data is incomplete or the
                    format of the raw data deviates from expectations.
        """

        expectedImageSize = None
        for dataChannel in self.get_data_channels():
            for fov in self.get_fovs():
                channelInfo = self.data.iloc[dataChannel]
                try:
                    imagePath = self._get_image_path(
                        channelInfo['imageType'], fov,
                        channelInfo['imagingRound'])
                except IndexError:
                    raise FileNotFoundError(
                        'Unable to find image path for %s, fov=%i, round=%i' %
                        (channelInfo['imageType'], fov,
                         channelInfo['imagingRound']))

                if not self._dataSet.rawDataPortal.open_file(
                        imagePath).exists():
                    raise InputDataError(
                        ('Image data for channel {0} and fov {1} not found. '
                         'Expected at {2}')
                        .format(dataChannel, fov, imagePath))

                try:
                    imageSize = self._dataSet.image_stack_size(imagePath)
                except Exception as e:
                    raise InputDataError(
                        ('Unable to determine image stack size for fov {0} from'
                         ' data channel {1} at {2}')
                        .format(dataChannel, fov, imagePath))

                frames = channelInfo['frame']

                # this assumes fiducials are stored in the same image file
                requiredFrames = max(np.max(frames),
                                     channelInfo['fiducialFrame'])
                if requiredFrames >= imageSize[2]:
                    raise InputDataError(
                        ('Insufficient frames in data for channel {0} and '
                         'fov {1}. Expected {2} frames '
                         'but only found {3} in file {4}')
                        .format(dataChannel, fov, requiredFrames, imageSize[2],
                                imagePath))

                if expectedImageSize is None:
                    expectedImageSize = [imageSize[0], imageSize[1]]
                else:
                    if expectedImageSize[0] != imageSize[0] \
                            or expectedImageSize[1] != imageSize[1]:
                        raise InputDataError(
                            ('Image data for channel {0} and fov {1} has '
                             'unexpected dimensions. Expected {1}x{2} but '
                             'found {3}x{4} in image file {5}')
                            .format(dataChannel, fov, expectedImageSize[0],
                                    expectedImageSize[1], imageSize[0],
                                    imageSize[1], imagePath))
