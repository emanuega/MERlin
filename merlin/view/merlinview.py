import sys
import argparse

from PyQt5 import QtWidgets

from merlin.core import dataset
from merlin.util import binary
from merlin.view.widgets import regionview

temp = '180710_HAECs_NoFlow_HAEC2\Sample1'

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data-set', required=True)

    return parser

def merlin_view():
    print('MERlinView - MERFISH data exploration software')
    parser = build_parser()
    args, argv = parser.parse_known_args()

    data = dataset.MERFISHDataSet(args.data_set)
    wTask = data.load_analysis_task('FiducialCorrelationWarp')
    dTask = data.load_analysis_task('DeconvolutionPreprocess')
    fTask = data.load_analysis_task('StrictFilterBarcodes')

    app = QtWidgets.QApplication([])

    frame = QtWidgets.QFrame()
    window = QtWidgets.QMainWindow()
    window.setCentralWidget(frame)
    window.resize(1000,1000)
    layout = QtWidgets.QGridLayout(frame)
    layout.addWidget(regionview.RegionViewWidget(
        wTask, fTask.get_barcode_database(), data))


    window.show()
    sys.exit(app.exec_())
