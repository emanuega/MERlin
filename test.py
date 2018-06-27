
class SimpleTask(AnalysisTask):
    def run(self):
        for i in range(100):
            print('running')
            time.sleep(5)
        super().run()

class SimpleParallelTask(ParallelAnalysisTask):
    def fragment_count(self):
        return 50

    def run_for_fragment(self, fragmentIndex):
        for i in range(10):
            print('running ' + str(fragmentIndex))
            time.sleep(1)


from merfish_code.data import dataset
d = dataset.DataSet('180616_TestData')
from merfish_code.analysis import analysistask
s = SimpleTask(d)
from merfish_code.schedule import executor
e = executor.LocalExecutor(s)



s2 = SimpleParallelTask(d)
s2.fragment_count()
e2 = executor.LocalExecutor(s2)
e2.run()
