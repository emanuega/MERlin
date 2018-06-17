from merfish_code.data import dataset
d = dataset.DataSet('180616_TestData')
from merfish_code.analysis import analysistask
s = analysistask.SimpleTask(d)
from merfish_code.schedule import executor
e = executor.LocalExecutor(s)



s2 = analysistask.SimpleParallelTask(d)
s2.fragment_count()
e2 = executor.LocalExecutor(s2)
