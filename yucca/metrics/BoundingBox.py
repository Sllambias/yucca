#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:25:09 2022

@author: zcr545
"""
from yucca.functional.array_operations.bounding_boxes import get_bbox_for_label


class Box(object):
    def __init__(self, array, label, padding):
        # Create bounding box for the object.
        self.array = array
        self.label = label
        self.padding = padding
        self.set_coordinates()

    def set_coordinates(self):
        box = get_bbox_for_label(self.array, self.label, self.padding)

        assert len(box) in [4, 6], "invalid box dimensions. Should be " f"4 (2D) or 6 (3D) but is {len(box)}"
        if len(box) == 6:
            self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = box
        if len(box) == 4:
            self.xmin, self.xmax, self.ymin, self.ymax = box

    @property
    def area(self):
        """
        Calculates the surface area. useful for IOU!
        """
        width = abs(self.xmax - self.xmin)
        height = abs(self.ymax - self.ymin)
        depth = abs(self.zmax - self.zmin)
        return width * height * depth

    def intersect(self, bbox):
        # Checks that exclude intersection
        if self.xmin > bbox.xmax or bbox.xmin > self.xmax:
            return False

        if self.ymin > bbox.ymax or bbox.ymin > self.ymax:
            return False

        if self.zmin > bbox.zmax or bbox.zmin > self.zmax:
            return False

        x1 = min(self.xmax, bbox.xmax)
        x2 = max(self.xmin, bbox.xmin)

        y1 = min(self.ymax, bbox.ymax)
        y2 = max(self.ymin, bbox.ymin)

        z1 = min(self.zmax, bbox.zmax)
        z2 = max(self.zmin, bbox.zmin)

        intersection = max(x1 - x2, 0) * max(y1 - y2, 0) * max(z1 - z2, 0)
        return intersection

    def iou(self, bbox):
        intersection = self.intersect(bbox)
        if not intersection:
            return False

        iou = intersection / float(self.area + bbox.area - intersection)
        # return the intersection over union value
        return iou
