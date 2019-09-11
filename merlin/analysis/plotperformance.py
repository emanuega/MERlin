import os
from matplotlib import pyplot as plt
import pandas
import merlin
import seaborn
import numpy as np
from typing import List
from merlin.core import analysistask
from merlin.analysis import filterbarcodes
from random import sample
import time

from merlin import plots
plt.style.use(
        os.sep.join([os.path.dirname(merlin.__file__),
                     'ext', 'default.mplstyle']))


class PlotPerformance(analysistask.AnalysisTask):

    """
    An analysis task that generates plots depicting metrics of the MERFISH
    decoding.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'exclude_plots' in self.parameters:
            self.parameters['exclude_plots'] = []

        self.taskTypes = ['decode_task', 'filter_task', 'optimize_task',
                          'segment_task', 'sum_task', 'partition_task',
                          'global_align_task']

    def get_estimated_memory(self):
        return 30000

    def get_estimated_time(self):
        return 180

    def get_dependencies(self):
        return []

    def _run_analysis(self):
        taskDict = {t: self.dataSet.load_analysis_task(self.parameters[t])
                    for t in self.taskTypes if t in self.parameters}
        plotEngine = plots.PlotEngine(self, taskDict)
        while not plotEngine.take_step():
            pass

