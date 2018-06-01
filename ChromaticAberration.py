import storm_analysis.sa_library.parameters as parameters
import storm_analysis.daostorm_3d.mufit_analysis as mfit
import os


import matplotlib.pyplot as plt

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
    
dataPath = 'C:/users/george/documents/test_data/';
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

for currentFile in os.listdir():
    if currentFile.endswith('.dax'):
        fit_spots(currentFile, endFrame=2)

'''
params = getDefaultParameters()

params.toXMLFile('parameters.xml')

if not os.path.exists('final.hdf5'):
    mfit.analyze(testFile, 'final.hdf5', 'parameters.xml')

import storm_analysis.sa_library.sa_h5py as saH5Py
infile = saH5Py.SAH5Py('final.hdf5')    
locs0 = infile.getLocalizationsInFrame(0)
locs1 = infile.getLocalizationsInFrame(1)


from sklearn.neighbors import NearestNeighbors

referencePoints = [[x, y] for x,y in zip(locs1['x'], locs1['y'])]
movingPoints = [[x,y] for x,y in zip(locs0['x'], locs0['y'])]

neighbors = NearestNeighbors(n_neighbors=1, radius=10)

neighbors.fit(referencePoints)
distances, indexes = neighbors.kneighbors(movingPoints, return_distance=True)

distanceThreshold = 5

controlPoints = [(referencePoints[indexes[i][0]], movingPoints[i]) \
        for i in range(len(movingPoints)) \
        if distances[i][0] < distanceThreshold]



'''
