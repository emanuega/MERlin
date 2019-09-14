import hashlib
import numpy as np
import os
import re
import tifffile
import boto3
from urllib import parse
from typing import List

# The following  code is adopted from github.com/ZhuangLab/storm-analysis and
# is subject to the following license:
#
# The MIT License
#
# Copyright (c) 2013 Zhuang Lab, Harvard University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


def infer_reader(filename: str, verbose: bool=False):
    """
    Given a file name this will try to return the appropriate
    reader based on the file extension.
    """
    ext = os.path.splitext(filename)[1]
    s3 = filename.startswith('s3://')
    if s3:
        if ext == ".dax":
            return S3DaxReader(filename, verbose=verbose)
    else:
        if ext == ".dax":
            return DaxReader(filename, verbose=verbose)
        elif ext == ".tif" or ext == ".tiff":
            return TifReader(filename, verbose=verbose)
    print(ext, "is not a recognized file type")
    raise IOError(
        "only .dax, .spe and .tif are supported (case sensitive..)")


class Reader(object):
    """
    The superclass containing those functions that
    are common to reading a STORM movie file.
    Subclasses should implement:
     1. __init__(self, filename, verbose = False)
        This function should open the file and extract the
        various key bits of meta-data such as the size in XY
        and the length of the movie.
     2. loadAFrame(self, frame_number)
        Load the requested frame and return it as np array.
    """

    def __init__(self, filename, verbose=False):
        super(Reader, self).__init__()
        self.image_height = 0
        self.image_width = 0
        self.number_frames = 0
        self.stage_x = 0
        self.stage_y = 0
        self.filename = filename
        self.fileptr = None
        self.verbose = verbose

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.close()

    def average_frames(self, start=None, end=None):
        """
        Average multiple frames in a movie.
        """
        length = 0
        average = np.zeros((self.image_height, self.image_width),
                           np.float)
        for [i, frame] in self.frame_iterator(start, end):
            if self.verbose and ((i % 10) == 0):
                print(" processing frame:", i, " of", self.number_frames)
            length += 1
            average += frame

        if length > 0:
            average = average / float(length)

        return average

    def close(self):
        if self.fileptr is not None:
            self.fileptr.close()
            self.fileptr = None

    def film_filename(self):
        """
        Returns the film name.
        """
        return self.filename

    def film_size(self):
        """
        Returns the film size.
        """
        return [self.image_width, self.image_height, self.number_frames]

    def film_location(self):
        """
        Returns the picture x,y location, if available.
        """
        if hasattr(self, "stage_x"):
            return [self.stage_x, self.stage_y]
        else:
            return [0.0, 0.0]

    def film_scale(self):
        """
        Returns the scale used to display the film when
        the picture was taken.
        """
        if hasattr(self, "scalemin") and hasattr(self, "scalemax"):
            return [self.scalemin, self.scalemax]
        else:
            return [100, 2000]

    def frame_iterator(self, start=None, end=None):
        """
        Iterator for going through the frames of a movie.
        """
        if start is None:
            start = 0
        if end is None:
            end = self.number_frames

        for i in range(start, end):
            yield [i, self.load_frame(i)]

    def hash_ID(self):
        """
        A (hopefully) unique string that identifies this movie.
        """
        return hashlib.md5(self.load_frame(0).tostring()).hexdigest()

    def load_frame(self, frame_number):
        assert frame_number >= 0, \
            "Frame_number must be greater than or equal to 0, it is "\
            + str(frame_number)
        assert frame_number < self.number_frames, \
            "Frame number must be less than " + str(self.number_frames)

    def lock_target(self):
        """
        Returns the film focus lock target.
        """
        if hasattr(self, "lock_target"):
            return self.lock_target
        else:
            return 0.0


class DaxReader(Reader):
    """
    Dax reader class. This is a Zhuang lab custom format.
    """

    def __init__(self, filename, verbose=False):
        super(DaxReader, self).__init__(filename, verbose=verbose)

        # save the filenames
        dirname = os.path.dirname(filename)
        if len(dirname) > 0:
            dirname = dirname + "/"
        self.inf_filename = dirname + os.path.splitext(
            os.path.basename(filename))[0] + ".inf"

        # defaults
        self.image_height = None
        self.image_width = None

        with open(self.inf_filename, 'r') as inf_file:
            self._parse_inf(inf_file.read().splitlines())

        # set defaults, probably correct, but warn the user
        # that they couldn't be determined from the inf file.
        if not self.image_height:
            print("Could not determine image size, assuming 256x256.")
            self.image_height = 256
            self.image_width = 256

        # open the dax file
        if os.path.exists(filename):
            self.fileptr = open(filename, "rb")
        else:
            if self.verbose:
                print("dax data not found", filename)

    def _parse_inf(self, inf_lines: List[str]) -> None:
        size_re = re.compile(r'frame dimensions = ([\d]+) x ([\d]+)')
        length_re = re.compile(r'number of frames = ([\d]+)')
        endian_re = re.compile(r' (big|little) endian')
        stagex_re = re.compile(r'Stage X = ([\d.\-]+)')
        stagey_re = re.compile(r'Stage Y = ([\d.\-]+)')
        lock_target_re = re.compile(r'Lock Target = ([\d.\-]+)')
        scalemax_re = re.compile(r'scalemax = ([\d.\-]+)')
        scalemin_re = re.compile(r'scalemin = ([\d.\-]+)')

        # defaults
        self.image_height = None
        self.image_width = None

        for line in inf_lines:
            m = size_re.match(line)
            if m:
                self.image_height = int(m.group(2))
                self.image_width = int(m.group(1))
            m = length_re.match(line)
            if m:
                self.number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    self.bigendian = 1
                else:
                    self.bigendian = 0
            m = stagex_re.match(line)
            if m:
                self.stage_x = float(m.group(1))
            m = stagey_re.match(line)
            if m:
                self.stage_y = float(m.group(1))
            m = lock_target_re.match(line)
            if m:
                self.lock_target = float(m.group(1))
            m = scalemax_re.match(line)
            if m:
                self.scalemax = int(m.group(1))
            m = scalemin_re.match(line)
            if m:
                self.scalemin = int(m.group(1))

        # set defaults, probably correct, but warn the user
        # that they couldn't be determined from the inf file.
        if not self.image_height:
            print("Could not determine image size, assuming 256x256.")
            self.image_height = 256
            self.image_width = 256

    def load_frame(self, frame_number):
        """
        Load a frame & return it as a np array.
        """
        super(DaxReader, self).load_frame(frame_number)

        self.fileptr.seek(
            frame_number * self.image_height * self.image_width * 2)
        image_data = np.fromfile(self.fileptr, dtype='uint16',
                                 count=self.image_height * self.image_width)
        image_data = np.reshape(image_data,
                                [self.image_height, self.image_width])
        if self.bigendian:
            image_data.byteswap(True)
        return image_data


class S3DaxReader(DaxReader):
    """
    Dax reader class for dax files stored on AWS S3.
    """

    def __init__(self, filename, verbose=False):
        parsedPath = parse.urlparse(filename)
        path = parsedPath.path

        dirname = os.path.dirname(path)
        if len(dirname) > 0:
            dirname = dirname + "/"
        self.inf_filename = dirname + os.path.splitext(
            os.path.basename(path))[0] + ".inf"

        self._parse_inf(boto3.resource('s3').Object(
            parsedPath.netloc, self.inf_filename.strip('/')
            ).get()['Body'].read().decode('utf-8').splitlines())

        # open the dax file
        self.fileptr = boto3.resource('s3').Object(
            parsedPath.netloc, parsedPath.path.strip('/'))

    def load_frame(self, frame_number):
        """
        Load a frame & return it as a np array.
        """
        super(DaxReader, self).load_frame(frame_number)

        startByte = frame_number * self.image_height * self.image_width * 2
        endByte = startByte + 2*(self.image_height * self.image_width) - 1
        image_data = np.frombuffer(self.fileptr.get(
            Range='bytes=%i-%i' % (startByte, endByte))['Body'].read(),
            dtype='uint16')
        image_data = np.reshape(image_data,
                                [self.image_height, self.image_width])
        if self.bigendian:
            image_data.byteswap(True)
        return image_data


class TifReader(Reader):
    """
    TIF reader class.

    This is supposed to handle the following:
    1. A normal Tiff file with one frame/image per page.
    2. Tiff files with multiple frames on a single page.
    3. Tiff files with multiple frames on multiple pages.
    """

    def __init__(self, filename, verbose=False):
        super(TifReader, self).__init__(filename, verbose)

        self.page_data = None
        self.page_number = -1

        # Save the filename
        self.fileptr = tifffile.TiffFile(filename)
        number_pages = len(self.fileptr.pages)

        # Single page Tiff file, which might be a "ImageJ Tiff"
        # with many frames on a page.
        if number_pages == 1:

            # Determines the size without loading the entire file.
            isize = self.fileptr.series[0].shape

            # Check if this is actually just a single frame tiff, if
            # it is we'll just load it into memory.
            if len(isize) == 2:
                self.frames_per_page = 1
                self.number_frames = 1
                self.image_height = isize[0]
                self.image_width = isize[1]
                self.page_data = self.fileptr.asarray()

            # Otherwise we'll memmap it in case it is really large.
            else:
                self.frames_per_page = isize[0]
                self.number_frames = isize[0]
                self.image_height = isize[1]
                self.image_width = isize[2]
                self.page_data = self.fileptr.asarray(out='memmap')

        # Multiple page Tiff file.
        #
        else:
            isize = self.fileptr.asarray(key=0).shape

            # Check for one frame per page.
            if len(isize) == 2:
                self.frames_per_page = 1
                self.number_frames = number_pages
                self.image_height = isize[0]
                self.image_width = isize[1]

            # Multiple frames per page.
            #
            # FIXME: No unit test for this kind of file.
            #
            else:
                self.frames_per_page = isize[0]
                self.number_frames = number_pages * isize[0]
                self.image_height = isize[1]
                self.image_width = isize[2]

        if self.verbose:
            print("{0:0d} frames per page, {1:0d} pages".format(
                self.frames_per_page, number_pages))

    def load_frame(self, frame_number, cast_to_int16=True):
        super(TifReader, self).load_frame(frame_number)

        # All the data is on a single page.
        if self.number_frames == self.frames_per_page:
            if self.number_frames == 1:
                image_data = self.page_data
            else:
                image_data = self.page_data[frame_number, :, :]

        # Multiple frames of data on multiple pages.
        elif self.frames_per_page > 1:
            page = int(frame_number / self.frames_per_page)
            frame = frame_number % self.frames_per_page

            # This is an optimization for files with a large number of frames
            # per page. In this case tifffile will keep loading the entire
            # page over and over again, which really slows everything down.
            # Ideally tifffile would let us specify which frame on the page
            # we wanted.
            #
            # Since it was going to load the whole thing anyway we'll have
            # memory overflow either way, so not much we can do about that
            # except hope for small file sizes.
            #
            if page != self.page_number:
                self.page_data = self.fileptr.asarray(key=page)
                self.page_number = page
            image_data = self.page_data[frame, :, :]

        # One frame on each page.
        else:
            image_data = self.fileptr.asarray(key=frame_number)

        assert (len(
            image_data.shape) == 2), "Not a monochrome tif image! " + str(
            image_data.shape)

        if cast_to_int16:
            image_data = image_data.astype(np.uint16)

        return image_data
