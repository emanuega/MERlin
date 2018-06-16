import os

from merfish_code.analysis import analysistask

import storm_analysis.sa_library.parameters as parameters
import storm_analysis.daostorm_3d.mufit_analysis as mfit


def generate_default_parameters(startFrame = -1, endFrame = -1):
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


class SpotFitTask(analysistask.ParallelAnalysisTask):
    
    def __init__(self, dataSet, analysisName=None, startFrame=-1, endFrame=-1):
        super().__init__(dataSet, analysisName)

        self.fitParameters = generate_default_parameters(startFrame, endFrame)

    def fragment_count(self):
        return len(self.dataSet.get_image_files())

    def run_for_fragment(self, fragmentIndex):
        imageFile = self.dataSet.get_image_files()[fragmentIndex]

        baseName = str.split(os.path.basename(imageFile), '.')[0]
        parametersName = self.get_savepath(baseName + '_parameters.xml') 
        outputName = self.get_savepath(baseName + '_spots.hdf5')

        self.fitParameters.toXMLFile(parametersName)

        if not os.path.exists(outputName):
            mfit.analyze(imageFile, outputName, parametersName)

        

