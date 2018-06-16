import matplotlib

import storm_analysis.sa_library.parameters as parameters
import storm_analysis.daostorm_3d.mufit_analysis as mfit
import storm_analysis.sa_library.sa_h5py as saH5Py
from storm_analysis.sa_library import datareader
from storm_analysis.sa_library import datawriter
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os


import matplotlib.pyplot as plt
plt.style.use('./util/default.mplstyle')

def getDefaultParameters(startFrame = -1, endFrame = -1):
    params = parameters.ParametersDAO()

    params.setAttr("max_frame", "int", endFrame)    
    params.setAttr("start_frame", "int", startFrame)
    
    params.setAttr("background_sigma", "float", 20.0)
    params.setAttr("camera_gain", "float", 1)
    params.setAttr("camera_offset", "float", 100)
    params.setAttr("find_max_radius", "int", 5)
    params.setAttr("foreground_sigma", "float", 2.0)
    params.setAttr("iterations", "int", 1)
    params.setAttr("model", "string", '2dfixed')
    params.setAttr("pixel_size", "float", 106)
    params.setAttr("roi_size", "int", 9)
    params.setAttr("sigma", "float", 2)
    params.setAttr("threshold", "float", 6.0)

    # Do tracking.
    params.setAttr("descriptor", "string", "1")
    params.setAttr("radius", "float", 0)

    # Do drift-correction.
    params.setAttr("d_scale", "int", 2)
    params.setAttr("drift_correction", "int", 0)
    params.setAttr("frame_step", "int", 500)
    params.setAttr("z_correction", "int", 0)

    return params
    
dataPath = '//10.245.74.90/data/flow_differentiation/180531_U2OS_smFISH_FLNA_Surface_Treatment/Sample1/';


destPath = '//10.245.74.90/data/htseq_fun/180209_MERFISH_Test/data'
destPath2 = '//10.245.74.90/data/flow_differentiation/180430_MERFISH_L26_U2OS_Sample2/data'
#dataPath = '//10.245.74.90/data/flow_differentiation/180531_ChromaticAbberationTest_750_650/data'
dataPath = '//10.245.74.90/data/htseq_fun/180604_PostEmbed_FLNA_Attempt2/Sample1_zstack'
os.chdir(dataPath)
print(os.getcwd())

testFile = 'Conventional_750_650_561_405_035.dax';

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def fit_spots(fileName, analysisDirectory='analysis', 
        startFrame=-1, endFrame=-1):
    ensure_directory(analysisDirectory)
    baseName = str.split(fileName, '.')[0]
    parametersName = analysisDirectory + os.path.sep + \
            baseName + '_parameters.xml'
    outputName = analysisDirectory + os.path.sep + \
            baseName + '_spots.hdf5'

    params = getDefaultParameters(startFrame, endFrame)
    params.toXMLFile(parametersName)

    if not os.path.exists(outputName):
        mfit.analyze(fileName, outputName, parametersName)
    else:
        print(outputName + ' already exists')

def fit_all_dax_in_current_dir(startFrame=-1, endFrame=-1):
    for currentFile in os.listdir():
        if currentFile.endswith('.dax'):
            fit_spots(currentFile, startFrame=startFrame, endFrame=endFrame)

def load_all_localizations(directory='.'):

    localizationDictionary = {}
    for currentFile in os.listdir(directory):
        if currentFile.endswith('.hdf5'):
            fileName = currentFile.split('.')[0]
            localizationDictionary[fileName] = \
                    saH5Py.SAH5Py(directory + os.path.sep + currentFile)
    
    return localizationDictionary

def extract_coordinates(localizationData, frameIndex):
    localizationSet = localizationData.getLocalizationsInFrame(frameIndex)
    return [[x, y] for x,y in zip(localizationSet['x'], localizationSet['y'])]

def pair_points(localizationData, referenceFrame, movingFrame,
        distanceThreshold = 5):
    referencePoints = extract_coordinates(localizationData, referenceFrame)
    movingPoints = extract_coordinates(localizationData, movingFrame)

    neighbors = NearestNeighbors(n_neighbors=1, radius=10)

    neighbors.fit(referencePoints)
    distances, indexes = neighbors.kneighbors(
            movingPoints, return_distance=True)

    controlPoints = [(referencePoints[indexes[i][0]], movingPoints[i]) \
            for i in range(len(movingPoints)) \
            if distances[i][0] < distanceThreshold]

    return controlPoints


def load_and_pair(framesToPair):
    localizationDictionary = load_all_localizations('analysis')
    pairedDictionaries = [{k: pair_points(v, f[0], f[1]) for k,v in \
            localizationDictionary.items()} for f in framesToPair]
    pairedList = [[p for x in pairedDictionary.values() for p in x] \
            for  pairedDictionary in pairedDictionaries]
    return pairedList

def bin_pairs(pairedList, binSize=16, extents=2048):
    binCount = int(extents/binSize)
    binnedPoints = [[[] for i in range(binCount)] for j in range(binCount)]

    for currentPair in pairedList:
        binX = int(currentPair[0][0]/binSize)
        binY = int(currentPair[0][1]/binSize)
        binnedPoints[binX][binY].append(currentPair)

    return binnedPoints

def calculate_displacement_for_pair(pair):
    return [pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]]

def plot_binned_displacements(
        pairedPoints, magnification=50, binSize=128, extents=2048,
        saveName=None):
    meanDisplacements = [[np.mean(
            [calculate_displacement_for_pair(x) for x in y], axis=0) \
        for y in z] for z in bin_pairs(pairedPoints, binSize, extents)]
    binCount = int(extents/binSize)

    plt.figure(figsize=(5,5))
    for i in range(binCount-1):
        for j in range(binCount-1):
            x = (i+0.5)*binSize
            y = (j+0.5)*binSize
            plt.plot([x, x+magnification*meanDisplacements[i][j][0]],
                    [y, y+magnification*meanDisplacements[i][j][1]], 
                    color='#55AA55')
            plt.scatter([x], [y], s=40, facecolors='none', 
                edgecolors='#407F7F') 

    plt.title('Mean displacement by position\n(extended ' + \
            str(magnification) + 'x)')
    plt.ylabel('Y (pixels)')
    plt.xlabel('X (pixels)')

    if saveName is None:
        save_figure('mean_displacements_by_position')
    else:
        save_figure(saveName)

    plt.show()


'''
Returns a list where the first element in each entry is the distance of the
first point from the center and the second element is the displacement 
between the two points.
'''
def calculate_radial_displacements(pairedPoints, centerX=1024, centerY=1024):
    radialDisplacements = [[np.linalg.norm(
                    [p[0][0]-centerX, p[0][1]-centerY]), \
                np.linalg.norm([p[0][0]-p[1][0], p[0][1]-p[1][1]])] \
                for p in pairedPoints]
    return np.array(radialDisplacements)


def plot_radial_displacements(pairedPoints, binCount=500, saveName=None):
    radialDisplacements = calculate_radial_displacements(pairedPoints)
    
    binWidth = np.ceil(1.001*np.max(radialDisplacements[:,0])/binCount)
    binnedDisplacements = [[] for i in range(binCount)]

    for currentDisplacement in radialDisplacements:
        binnedDisplacements[int(currentDisplacement[0]/binWidth)].append(
                currentDisplacement[1])

    plt.figure(figsize=(5,4))
    plt.plot(np.arange(0, binWidth*binCount, binWidth), 
            [np.mean(x) for x in binnedDisplacements])

    plt.xlabel('Radial distance (pixels)')
    plt.ylabel('Mean displacement (pixels)')
    plt.title('Displacement vs radial distance')

    if saveName is None:
        save_figure('radial_displacement')
    else:
        save_figure(saveName)

    return binnedDisplacements





def save_figure(saveName, savePath = 'figures/'):
    ensure_directory(savePath)
    if saveName is not None:
        fullPath = os.sep.join([savePath, saveName])
        plt.savefig(fullPath + '.png', pad_inches=0)
        plt.savefig(fullPath + '.pdf', transparent=True, pad_inches=0)

from storm_analysis.sa_library import datareader
import cv2


def plot_image_overlay(image1, image2, vmax=15000):
    width = image1.shape[0]
    height = image1.shape[1]
    colorImage = np.zeros((width, height, 3))
    colorImage[:,:,0] = image1/vmax
    colorImage[:,:,1] = image2/vmax
    plt.imshow(colorImage)

def transform_all():
    ensure_directory('preprocessed')
    for currentFile in os.listdir():
        if currentFile.endswith('.dax'):
            transform_750(currentFile, h, [0])


def transform_750(inputName, transform, indexesToTransform): 
    reader = datareader.inferReader(inputName)
    writer = datawriter.inferWriter(os.sep.join(['preprocessed', inputName]))
    
    filmSize = reader.filmSize()

    for i in range(filmSize[2]):
        if i in indexesToTransform:
            writer.addFrame(cv2.warpPerspective(
                reader.loadAFrame(i), transform, (filmSize[0], filmSize[1])))
        else:
            writer.addFrame(reader.loadAFrame(i))

    reader.close()
    writer.close()


