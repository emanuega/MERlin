import pandas 
import rtree
import networkx
import numpy as np
import cv2
from skimage.measure import regionprops

from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import filter

class SequentialSignal(analysistask.ParallelAnalysisTask):

    """
    An analysis task that calculates the signal intensity within the boundaries
    of a cell for all rounds not used in the codebook, useful for measuring 
    RNA species that were stained individually.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.zplane = int(self.parameters['zplane'])
        self.highpass = self.parameters['apply_highpass'].upper() == 'TRUE'
        dataorg = self.dataSet.get_data_organization()
        sequentialInfo = dataorg.get_sequential_rounds()

        self.channels = sequentialInfo[0]
        self.geneNames = sequentialInfo[1]
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048 

    def get_estimated_time(self):
        return 1 

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def real_coords_to_pixels(self,fov,cell,zplane):
        regions = cell.get_boundaries()[zplane]
        if len(regions) == 0:
            return []
        else:
            pixels = []
            tForm = np.linalg.inv(self.alignTask.fov_to_global_transform(fov))
            for region in regions:
                coords = region.exterior.coords.xy
                xyZip = list(zip(coords[0].tolist(),coords[1].tolist()))
                pixels.append(np.array(
                    [(np.array([x[0],x[1],1])*tForm)
                    .sum(1).round(0)[:2].astype(int) for x in xyZip]))
        return pixels
        
    def extract_signal(self,cells,img,zplane):
        cellCoords = []
        for cell in cells:
            regions = cell.get_boundaries()[zplane]
            if len(regions) == 0:
                cellCoords.append([])
            else:
                pixels = []
                for region in regions: 
                    coords = region.exterior.coords.xy
                    xyZip = list(zip(coords[0].tolist(),coords[1].tolist()))
                    pixels.append(np.array(
                                self.alignTask.global_coordinates_to_fov(
                                    cell.get_fov(), xyZip)))
                cellCoords.append(pixels)
        # keptCells and keptCellIDs prevent cells with no area from getting 
        # through, isn't strictly neccessary if get_intersection_graph 
        # is run with an area threshold
        keptCells = [cellCoords[x] for x in range(len(cells))
                     if len(cellCoords[x])>0]
        keptCellIDs = [str(cells[x].get_feature_id()) for x in range(len(cells))
                       if len(cellCoords[x])>0]
        mask = np.zeros(img.shape,np.uint8)
        i = 1
        for cell in keptCells:
            cv2.drawContours(mask,cell,-1,i,-1)
            i += 1
        props = regionprops(mask,img)
        propsOut = pandas.DataFrame(
            data = [(x.intensity_image.sum(),x.filled_area) for x in props],
            index = keptCellIDs,
            columns = ['Intensity','Pixels'])
        return(propsOut)

    def get_intersection_graph(self,polygonList, areaThreshold=250):
        # This is only currently necessary to eliminate 
        # problematic cell overlaps. If cell boundaries have been cleaned
        # prior to running sum signal this isn't necessary    

        polygonIndex = rtree.index.Index()
        intersectGraphEdges = [[i,i] for i in range(len(polygonList))]
        for i,cell in enumerate(polygonList):
            if len(cell.get_bounding_box()) == 4:
                putativeIntersects = list(polygonIndex.intersection(
                                          cell.get_bounding_box()))
            
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

    def get_sum_signal(self,fov,channels,zplane):

        fTask = self.dataSet.load_analysis_task(self.parameters['warp_task'])
        sTask = self.dataSet.load_analysis_task(self.parameters['segment_task'])

        sDB = sTask.get_feature_database()
        cells = sDB.read_features(fov)  
    
        # If cell boundaries are going to becleaned prior to this we should 
        # remove the part enclosed by pound symbols
        ig = self.get_intersection_graph(cells, areaThreshold=0)

        cellIndex  = [x for x in ig.nodes() if len(ig.edges(nbunch=x))==1 and 
                      cells[x].get_volume() > 0.0]
        cells = [cells[x] for x in range(len(cells)) if x in cellIndex]
        #####
        for ch in channels:
            img = fTask.get_aligned_image(fov,ch,zplane)
            if self.highpass:
                img = filter.high_pass_filter(img,
                    self.parameters['highpass_kernel'])
            signal = self.extract_signal(cells,img,zplane)
            if ch == channels[0]:
                compiledSignal = signal.iloc[:,[0]].copy(deep=True)
            else:
                compiledSignal = pandas.concat(
                                    [compiledSignal,signal.iloc[:,[0]]],1)
            if ch == channels[-1]:
                compiledSignal.columns = channels
                compiledSignal = pandas.concat(
                                    [compiledSignal,signal.iloc[:,[1]]],1)
        return compiledSignal

    def retrieve_fov_signal(self,fragmentIndex):
        return(self.dataSet.load_dataframe_from_csv(
            'sequential_signal', self.get_analysis_name(),
            fragmentIndex, 'signals'))

    def _run_analysis(self, fragmentIndex):

        fovSignal = self.get_sum_signal(fragmentIndex,self.channels,self.zplane)
        normSignal = fovSignal.iloc[:,:-1].div(fovSignal.loc[:,'Pixels'],0)

        normSignal.columns = self.geneNames

        self.dataSet.save_dataframe_to_csv(
                normSignal, 'sequential_signal',
                self.get_analysis_name(),
                fragmentIndex, 'signals')


class CombineSequential(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048 

    def get_estimated_time(self):
        return 5 

    def get_dependencies(self):
        return [self.parameters['sequential_task']]

    def _run_analysis(self):
        fovs =self.dataSet.get_fovs()
        sTask = self.dataSet.load_analysis_task(
                    self.parameters['sequential_task'])
        signals = pandas.concat([sTask.retrieve_fov_signal(fov)
                                 for fov in fovs]).reset_index(drop=True)
        
        self.dataSet.save_dataframe_to_csv(
                    signals, 'sequential_signal_compiled',
                    self.get_analysis_name())


