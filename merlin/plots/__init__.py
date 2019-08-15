import inspect
import merlin
import pkgutil


def get_available_plots():
    plotModules = [obj for name, obj in inspect.getmembers(merlin.plots)
                   if inspect.ismodule(obj)]

    for importer, modname, ispkg in pkgutil.iter_modules(merlin.plots.__path__):
        for name, obj in inspect.getmembers(modname):
            if inspect.isclass(obj):
                print(name)
                print(obj)
