import pandas 
from shapely import geometry
from scipy import ndimage
import rtree
import networkx
import numpy as np
import cv2
from skimage.measure import regionprops

from merlin.core import analysistask
from merlin.core import dataset
from merlin.util import spatialfeature
from merlin.analysis import segment
from merlin.data import dataorganization

class SequentialSignal(analysistask.ParallelAnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.zplane = int(self.parameters['zplane'])
        dataorg = dataorganization.DataOrganization(self.dataSet)
        sequentialInfo = dataorg.get_sequential_rounds()

        self.channels = sequentialInfo[0]
        self.geneNames = sequentialInfo[1]
        self.alignTask = self.dataSet.load_analysis_task('SimpleGlobalAlignment')

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048 

    def get_estimated_time(self):
        return 1 

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['segment_task']]

    def realCoordsToPixels(self,fov,cell,zplane):
        regions = cell.get_boundaries()[zplane]
        if len(regions) == 0:
            return []
        else:
            pixels = []
            tForm = np.linalg.inv(self.alignTask.fov_to_global_transform(fov))
            for region in regions:
                coords = region.exterior.coords.xy
                xyZip = list(zip(coords[0].tolist(),coords[1].tolist()))
                pixels.append(np.array([(np.array([x[0],x[1],1])*tForm).sum(1).round(0)[:2].astype(int) for x in xyZip]))
        return pixels
        
    def extractSignal(self,cells,img,zplane):
        cellCoords = [self.realCoordsToPixels(x.get_fov(),x,zplane) for x in cells]
        #keptCells and keptCellIDs prevent cells with no area from getting through
        #Isn't strictly neccessary if get_intersection_graph is run with an area threshold
        keptCells = [cellCoords[x] for x in range(len(cells)) if len(cellCoords[x])>0]
        keptCellIDs = [str(cells[x].get_feature_id()) for x in range(len(cells)) if len(cellCoords[x])>0]
        mask = np.zeros(img.shape,np.uint8)
        i = 1
        for cell in keptCells:
            cv2.drawContours(mask,cell,-1,i,-1)
            i += 1
        props = regionprops(mask,img)
        propsOut = pandas.DataFrame(data = [(x.intensity_image.sum(),x.filled_area) for x in props],index = keptCellIDs,columns = ['Intensity','Pixels'])
        return(propsOut)

#Not really clear to me if high pass filter should or shouldn't be used to clean the image prior to getting intensity
    def highPassFilter(self,img):
        img = img.astype(np.int16)
        lowpass = ndimage.gaussian_filter(img, 3)
        gauss_highpass = img - lowpass
        gauss_highpass[gauss_highpass<0] = 0
        return gauss_highpass

#This is only currently necessary to eliminate problematic cell overlaps. If cell boundaries have been cleaned \
#prior to running sum signal this isn't necessary    
    def get_intersection_graph(self,polygonList, areaThreshold=250):
        polygonIndex = rtree.index.Index()
        intersectGraphEdges = [[i,i] for i in range(len(polygonList))]
        for i,cell in enumerate(polygonList):
            if len(cell.get_bounding_box()) == 4:
                putativeIntersects = list(polygonIndex.intersection(cell.get_bounding_box()))
            
                if len(putativeIntersects) > 0:
                    try:
                        intersectGraphEdges += \
                                [[i, j] for j in putativeIntersects \
                                if cell.intersection(
                                    polygonList[j])>areaThreshold]
                    except Exception as e:
                        print(i)

                polygonIndex.insert(i, cell.get_bounding_box())

        intersectionGraph = networkx.Graph()
        intersectionGraph.add_edges_from(intersectGraphEdges)
        
        return intersectionGraph

    def getSumSignal(self,fov,channels,zplane):

        fTask = self.dataSet.load_analysis_task('FiducialCorrelationWarp')
        sTask = segment.WatershedSegment(self.dataSet, {'warp_task': 'FiducialCorrelationWarp', 'global_align_task': 'SimpleGlobalAlignment'})
        sDB = spatialfeature.HDF5SpatialFeatureDB(sTask.dataSet, sTask)
        cells = sDB.read_features(fov)  
    
        #If cell boundaries are going to becleaned prior to this we should remove the part enclosed by pound symbols
        ig = self.get_intersection_graph(cells, areaThreshold=0)

        cellIndex  = [x for x in ig.nodes() if len(ig.edges(nbunch=x))==1 and cells[x].get_volume() > 0.0]
        cells = [cells[x] for x in range(len(cells)) if x in cellIndex]
        #####
        for ch in channels:
            img = fTask.get_aligned_image(fov,ch,zplane)
            hpImg = self.highPassFilter(img)
            signal = self.extractSignal(cells,hpImg,zplane)
            if ch == channels[0]:
                compiledSignal = signal.iloc[:,[0]].copy(deep=True)
            else:
                compiledSignal = pandas.concat([compiledSignal,signal.iloc[:,[0]]],1)
            if ch == channels[-1]:
                compiledSignal.columns = channels
                compiledSignal = pandas.concat([compiledSignal,signal.iloc[:,[1]]],1)
        return compiledSignal


    def _run_analysis(self, fragmentIndex):

        fovSignal = self.getSumSignal(fragmentIndex,self.channels,self.zplane)
        normSignal = fovSignal.iloc[:,:-1].div(fovSignal.loc[:,'Pixels'],0)

        normSignal.columns = self.geneNames

        self.dataSet.save_dataframe_to_csv(
                normSignal, 'sequential_signal',
                self.get_analysis_name(),
                fragmentIndex, 'signals')

