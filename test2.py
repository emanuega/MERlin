dataName = '180710_HAECs_NoFlow_HAEC2\sample1'

dataOrganization = 'Culture_16bits'
codebook = 'HAEC1E1'

from merfish_code.core import dataset
from merfish_code.core import executor
from merfish_code.analysis import warp
from merfish_code.analysis import preprocess
from merfish_code.analysis import optimize
from merfish_code.analysis import decode
from merfish_code.analysis import globalalign
from merfish_code.analysis import generatemosaic
from merfish_code.analysis import plotperformance
from merfish_code.analysis import filterbarcodes

import json
from merfish_code.core import scheduler

dSet = dataset.MERFISHDataSet(
        dataName, 
        dataOrganizationName=dataOrganization,
        codebookName=codebook)


with open('test_analysis_parameters.json', 'r') as f:
    s = scheduler.Scheduler(dSet, json.load(f))

'''
wTask = warp.FiducialCorrelationWarp(dSet)
d = {'warp_task': 'FiducialCorrelationWarp'}
pTask = preprocess.DeconvolutionPreprocess(dSet, d)
d2 = {'preprocess_task': 'DeconvolutionPreprocess'}
oTask = optimize.Optimize(dSet, d2, analysisName='o2')
oTask2 = optimize.Optimize(dSet, d2)



d3 = {'preprocess_task': 'DeconvolutionPreprocess',
        'optimize_task': 'Optimize',
        'global_align_task': 'SimpleGlobalAlignment'}
dTask = decode.Decode(dSet, d3)

gTask = globalalign.SimpleGlobalAlignment(dSet)
cgTask = globalalign.CorrelationGlobalAlignment(dSet)

d4 = {'global_align_task': 'SimpleGlobalAlignment',
        'warp_task': 'FiducialCorrelationWarp'}
mTask = generatemosaic.GenerateMosaic(dSet, d4)

d5 = {'global_align_task': 'SimpleGlobalAlignment',
        'warp_task': 'FiducialCorrelationWarp',
        'decode_task': 'Decode'}

ppTask = plotperformance.PlotPerformance(dSet, parameters=d5)

fTask = filterbarcodes.FilterBarcodes(dSet, parameters=d5)

e = executor.LocalExecutor(dTask)

dataName2 = '180425_MERFISH_B4_6/data'
dataOrganization2 = 'Tissue_16bits'
dataHome2 = '//10.245.74.90/data/htseq_fun'
codebook2 = 'M22E1'

dSet2 = dataset.MERFISHDataSet(
        dataName2, dataOrganizationName=dataOrganization2,
        dataHome=dataHome2,
        codebookName=codebook2)


wTask2 = warp.FiducialCorrelationWarp(dSet2)
d = {'warp_task': 'FiducialCorrelationWarp'}
pTask2 = preprocess.DeconvolutionPreprocess(dSet2, d)

gTask2 = globalalign.SimpleGlobalAlignment(dSet2)
d4 = {'align_task': 'SimpleGlobalAlignment'}
mTask2 = generatemosaic.GenerateMosaic(dSet2, d4)
'''
