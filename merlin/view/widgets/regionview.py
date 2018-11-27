import numpy as np
import time

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from merlin.core import dataset
from merlin.util import binary
from merlin.view.widgets import regionview


class RegionViewWidget(QWidget):

    def __init__(self, warpTask, barcodeDB, dataSet):
        super().__init__()

        self.fov = 0
        self.zIndex = 0
        self.warpTask = warpTask
        self.dataSet = dataSet
        self.barcodeDB = barcodeDB

        vSynchronize = ImageViewSynchronizer()
        imageData =[self.warpTask.get_aligned_image(
            self.fov, dc, self.zIndex) \
                    for dc in self.dataSet.get_data_channels()]
        imageCount = len(imageData)
        barcodes = self.barcodeDB.get_barcodes(fov=self.fov)

        self.imageViews = [RegionImageViewWidget(imageData[i],
            vSynchronize, bitIndex=i, barcodes=barcodes,
            title=self.dataSet.get_data_channel_name(i)) \
                    for i in range(imageCount)]

        self._initialize_layout()

    def _initialize_layout(self):
        fovLabel = QLabel('Field of view')
        self.fovScrollBar = QScrollBar(Qt.Horizontal)
        self.fovScrollBar.setMaximum(np.max(self.dataSet.get_fovs()))
        self.fovScrollBar.sliderReleased.connect(self.fov_scroll_update)

        self.zScrollBar = QScrollBar(Qt.Horizontal)
        self.zScrollBar.setMaximum(len(self.dataSet.get_z_positions())-1)
        self.zScrollBar.sliderReleased.connect(self.z_scroll_update)

        self.controlForm = QGroupBox()
        controlFormLayout = QFormLayout()
        controlFormLayout.addRow(QLabel('Field of view'), self.fovScrollBar)
        controlFormLayout.addRow(QLabel('Z index'), self.zScrollBar)
        self.controlForm.setLayout(controlFormLayout)
        self.controlForm.setMaximumHeight(50)

        mainLayout = QVBoxLayout(self)
        mainLayout.addWidget(self.controlForm)
        imageLayout = QGridLayout()
        mainLayout.addLayout(imageLayout)
        columnCount = 6
        imageCount = len(self.imageViews)
        rowCount = int(np.ceil(imageCount/columnCount))

        imageIndex = 0
        for i in range(1,rowCount+1):
            for j in range(1,columnCount+1):
                imageLayout.addWidget(self.imageViews[imageIndex], i, j)
                imageIndex += 1

    def fov_scroll_update(self):
        self.set_fov(self.fovScrollBar.value())

    def z_scroll_update(self):
        self.set_z_index(self.zScrollBar.value())

    def _update_fov_data(self):
        imageData = [self.warpTask.get_aligned_image(
            self.fov, dc, self.zIndex) \
                    for dc in self.dataSet.get_data_channels()]
        barcodes = self.barcodeDB.get_barcodes(fov=self.fov)
        for i, iView in enumerate(self.imageViews):
            iView.set_data(imageData[i], barcodes)

    def set_fov(self, fov):
        if fov == self.fov:
            return
        self.fov = fov
        self._update_fov_data()

    def set_z_index(self, zIndex):
        self.zIndex = zIndex
        self._update_fov_data()


class RegionImageViewWidget(QWidget):
    def __init__(self, imageData, vSynchronize, bitIndex=None, barcodes=None,
            title=''):
        super().__init__()
        self._synchronizer = vSynchronize
        self._synchronizer.updateViewSignal.connect(self.update)

        self._mousePressed = False
        self._pressPosition = None

        self.bitIndex = bitIndex
        self.set_data(imageData, barcodes)
        self.title = title

        self.setMouseTracking(True)

    def set_data(self, image, barcodes=None):
        self.imageData = self.scale_image(image)
        self.barcodes = barcodes

        if barcodes is not None:
            self.barcodePositions = np.array(
                    [[bc['x'], bc['y'], i, binary.k_bit_set(
                        int(bc['barcode']), self.bitIndex)] for i,bc in 
                        self.barcodes.iterrows()])
            self.barcodePositions = self.barcodePositions[\
                    self.barcodePositions[:,0].argsort()]
        else:
            self.barcodePositions = None


        self.update()


    def scale_image(self, inputImage):
        imageMax = np.max(inputImage)
        imageMin = np.min(inputImage)

        lookupTable = np.arange(2**16, dtype='uint16')
        lookupTable.clip(imageMin, imageMax, out=lookupTable)
        lookupTable -= imageMin
        np.floor_divide(lookupTable, (imageMax-imageMin+1)/256,
                out=lookupTable, casting='unsafe')
        lookupTable = lookupTable.astype('uint8')

        return np.take(lookupTable, inputImage)

    def paintEvent(self, event):

        start = time.time()
        painter = QPainter(self)


        transform = self._synchronizer.get_transform()
        inverseTransform = transform.inverted()[0]

        painter.setTransform(transform)
        painter.setRenderHint(QPainter.Antialiasing, True)

        windowSize = self.size()
        clippingBounds = inverseTransform.map(
                QPolygon(QRect(0, 0, self.width(), self.height())))\
                        .boundingRect()

        self._paint_image(painter)
        self._paint_barcodes(painter, clippingBounds)

        rawCursorPosition = self._synchronizer.get_cursor_position()
        if rawCursorPosition:
            cursorPosition = inverseTransform.map(
                    self._synchronizer.get_cursor_position())

            cursorPen = QPen(QColor(20, 255, 20, 20))
            cursorPen.setWidthF(0.01*clippingBounds.width())
            painter.setPen(cursorPen)
            painter.drawLine(clippingBounds.left(), cursorPosition.y(),
                    clippingBounds.right(), cursorPosition.y())
            painter.drawLine(cursorPosition.x(), clippingBounds.top(),
                    cursorPosition.x(), clippingBounds.bottom())

        painter.resetTransform()
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(10, 20, self.title) 

        print(time.time()-start)

    def _paint_image(self, painter):
        height, width = self.imageData.shape
        qimage = QImage(
                self.imageData, height, width, QImage.Format_Grayscale8)
        painter.drawImage(0, 0, qimage)

    def _paint_barcodes(self, painter, clippingBounds):
        regionTooLarge = clippingBounds.width() > 500 \
                or clippingBounds.height() > 500 

        if self.barcodePositions is not None and not regionTooLarge:
            indexBounds = np.searchsorted(self.barcodePositions[:,0],
                    [clippingBounds.left(), clippingBounds.right()])
            for currentBarcode in \
                    self.barcodePositions[indexBounds[0]:indexBounds[1]]:
                if clippingBounds.contains(
                        currentBarcode[0], currentBarcode[1]):

                    if currentBarcode[3]:
                        painter.setPen(QColor(191, 0, 0, 150))
                    else:
                        painter.setPen(QColor(3, 129, 255, 50))

                    painter.drawEllipse(
                            currentBarcode[0]-2, currentBarcode[1]-2,
                            4, 4)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._mousePressed = True
            self._pressPosition = event.globalPos()

    def mouseReleaseEvent(self, event):
        self._mousePressed = False

    def mouseMoveEvent(self, event):
        if self._mousePressed:
            positionDifference = event.globalPos() - self._pressPosition
            transform = self._synchronizer.get_transform()
            transform.translate(
                    positionDifference.x()/transform.m11(),
                    positionDifference.y()/transform.m22())
            self._pressPosition = event.globalPos()

        self._synchronizer.set_cursor_position(event.localPos())
        self._synchronizer.updateViewSignal.emit()

    def wheelEvent(self, event):
        angleDelta = event.angleDelta()

        if angleDelta:
            transform = self._synchronizer.get_transform()

            scaleFactor = np.exp(angleDelta.y()/(10*8*15))
            oldCenter = transform.inverted()[0].map(
                    0.5*self.width(), 0.5*self.height())
            transform.scale(scaleFactor,  scaleFactor)
            newCenter = transform.inverted()[0].map(
                    0.5*self.width(), 0.5*self.height())
            transform.translate(newCenter[0] - oldCenter[0],
                    newCenter[1] - oldCenter[1])
            self._synchronizer.updateViewSignal.emit()


class ImageViewSynchronizer(QObject):

    updateViewSignal = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.transform = QTransform()
        self.cursorPosition = None
        self.selectedBarcode = None

    def get_transform(self):
        return self.transform

    def get_cursor_position(self):
        return self.cursorPosition

    def set_cursor_position(self, newPosition):
        self.cursorPosition = newPosition

    def get_selected_barcode(self):
        return self.selectedBarcode

    def set_selected_barcode(self, newSelection):
        self.selectedBarcode = newSelection


