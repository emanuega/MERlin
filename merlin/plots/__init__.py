import inspect
import pkgutil
import importlib
from typing import Set

import merlin
from merlin.plots._base import AbstractPlot
from merlin.plots._base import PlotMetadata


def get_available_plots() -> Set:
    """ Get all plots defined within any submodule of merlin.plots

    Returns: a set of references to the plots
    """
    plotSet = set()
    for importer, modname, ispkg in pkgutil.iter_modules(merlin.plots.__path__):
        currentModule = importlib.import_module(
            merlin.plots.__name__ + '.' + modname)
        for name, obj in inspect.getmembers(currentModule):
            if inspect.isclass(obj)\
                    and issubclass(obj, AbstractPlot)\
                    and obj != AbstractPlot:
                plotSet.add(obj)
    return plotSet
