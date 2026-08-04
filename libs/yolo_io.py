#!/usr/bin/env python
# -*- coding: utf8 -*-
import codecs
import os

from libs.constants import DEFAULT_ENCODING

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING

class YOLOWriter:

    def __init__(self, folder_name, filename, img_size, database_src='Unknown', local_img_path=None):
        self.folder_name = folder_name
        self.filename = filename
        self.database_src = database_src
        self.img_size = img_size
        self.box_list = []
        self.local_img_path = local_img_path
        self.verified = False

    def add_bnd_box(self, x_min, y_min, x_max, y_max, name, difficult):
        bnd_box = {'xmin': x_min, 'ymin': y_min, 'xmax': x_max, 'ymax': y_max}
        bnd_box['name'] = name
        bnd_box['difficult'] = difficult
        self.box_list.append(bnd_box)

    def bnd_box_to_yolo_line(self, box, class_list=[]):
        x_min = box['xmin']
        x_max = box['xmax']
        y_min = box['ymin']
        y_max = box['ymax']

        x_center = float((x_min + x_max)) / 2 / self.img_size[1]
        y_center = float((y_min + y_max)) / 2 / self.img_size[0]

        w = float((x_max - x_min)) / self.img_size[1]
        h = float((y_max - y_min)) / self.img_size[0]

        # PR387
        box_name = box['name']
        if box_name not in class_list:
            class_list.append(box_name)

        class_index = class_list.index(box_name)

        return class_index, x_center, y_center, w, h

    def save(self, class_list=[], target_file=None):
        try:
            out_file = None  # Update yolo .txt
            out_class_file = None   # Update class list .txt

            if target_file is None:
                out_file = open(
                self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)
                classes_file = os.path.join(os.path.dirname(os.path.abspath(self.filename)), "classes.txt")
                out_class_file = open(classes_file, 'w')

            else:
                out_file = codecs.open(target_file, 'w', encoding=ENCODE_METHOD)
                classes_file = os.path.join(os.path.dirname(os.path.abspath(target_file)), "classes.txt")
                out_class_file = open(classes_file, 'w')


            for box in self.box_list:
                class_index, x_center, y_center, w, h = self.bnd_box_to_yolo_line(box, class_list)
                # print (classIndex, x_center, y_center, w, h)
                out_file.write("%d %.6f %.6f %.6f %.6f\n" % (class_index, x_center, y_center, w, h))

            # print (classList)
            # print (out_class_file)
            for c in class_list:
                out_class_file.write(c+'\n')

            out_class_file.close()
            out_file.close()
        except Exception as e:
            print('Error saving YOLO format: %s' % e)
            import traceback
            traceback.print_exc()
            raise



class YoloReader:

    def __init__(self, file_path, image, class_list_path=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.file_path = file_path

        if class_list_path is None:
            dir_path = os.path.dirname(os.path.realpath(self.file_path))
            self.class_list_path = os.path.join(dir_path, "classes.txt")
        else:
            self.class_list_path = class_list_path

        # Safely load classes file — fall back to empty classes if missing
        try:
            with open(self.class_list_path, 'r') as classes_file:
                self.classes = classes_file.read().strip('\n').split('\n')
        except (IOError, OSError) as e:
            print('Warning: Could not read classes.txt at %s: %s. '
                  'Trying to infer classes from the label file.' % (self.class_list_path, e))
            # Try to infer class indices from the label file directly
            self.classes = self._infer_classes_from_file(file_path)

        img_size = [image.height(), image.width(),
                    1 if image.isGrayscale() else 3]

        self.img_size = img_size

        self.verified = False
        try:
            self.parse_yolo_format()
        except Exception as e:
            print('Error: Failed to parse YOLO format from %s: %s' % (file_path, e))
            import traceback
            traceback.print_exc()

    def _infer_classes_from_file(self, file_path):
        """Fallback: collect unique class indices from the label file."""
        class_ids = set()
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ')
                    if len(parts) >= 5:
                        class_ids.add(int(parts[0]))
        except Exception as e:
            print('Warning: Could not infer classes from %s: %s' % (file_path, e))
        # Create synthetic class names for any found indices
        return ['class_%d' % i for i in sorted(class_ids)]

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, x_min, y_min, x_max, y_max, difficult):

        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.shapes.append((label, points, None, None, difficult))

    def yolo_line_to_shape(self, class_index, x_center, y_center, w, h):
        try:
            class_idx = int(float(class_index))
            if class_idx < len(self.classes):
                label = self.classes[class_idx]
            else:
                label = 'class_%d' % class_idx
        except (ValueError, IndexError):
            label = str(class_index)

        x_min = max(float(x_center) - float(w) / 2, 0)
        x_max = min(float(x_center) + float(w) / 2, 1)
        y_min = max(float(y_center) - float(h) / 2, 0)
        y_max = min(float(y_center) + float(h) / 2, 1)

        # Guard against zero-size image
        if self.img_size[1] <= 0 or self.img_size[0] <= 0:
            return label, int(x_min), int(y_min), int(x_max), int(y_max)

        x_min = round(self.img_size[1] * x_min)
        x_max = round(self.img_size[1] * x_max)
        y_min = round(self.img_size[0] * y_min)
        y_max = round(self.img_size[0] * y_max)

        return label, x_min, y_min, x_max, y_max

    def parse_yolo_format(self):
        try:
            with open(self.file_path, 'r') as bnd_box_file:
                for bndBox in bnd_box_file:
                    bndBox = bndBox.strip()
                    if not bndBox:
                        continue
                    parts = bndBox.split(' ')
                    if len(parts) < 5:
                        print('Warning: Skipping malformed YOLO line in %s: %s' % (self.file_path, bndBox))
                        continue
                    class_index, x_center, y_center, w, h = parts[:5]
                    label, x_min, y_min, x_max, y_max = self.yolo_line_to_shape(class_index, x_center, y_center, w, h)

                    # Caveat: difficult flag is discarded when saved as yolo format.
                    self.add_shape(label, x_min, y_min, x_max, y_max, False)
        except Exception as e:
            print('Error parsing YOLO format from %s: %s' % (self.file_path, e))
            import traceback
            traceback.print_exc()
