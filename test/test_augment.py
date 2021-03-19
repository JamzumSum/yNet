from random import randint
from unittest import TestCase

import cv2 as cv
import torch
from common.utils import BoundingBoxCrop, morph_close
from data.augment import elastic, affine, getGaussianFilter
from data.dataset.cacheset import CachedDatasetGroup

gray2numpy = lambda tensor: (
    tensor[0] if tensor.dim() == 3 else tensor if tensor.dim() == 2 else None
).numpy()
show_gray_tensor = lambda title, tensor: cv.imshow(title, gray2numpy(tensor))


class TestAugment(TestCase):
    def setUp(self):
        self.ds = CachedDatasetGroup("data/set2/")

    def randomItem(self):
        rint = randint(0, len(self.ds))
        print(rint)
        return self.ds[rint]

    def item(self, i=0):
        return self.ds[i]

    def testElastic(self):
        org = self.item()["X"]
        img = elastic(org, *getGaussianFilter(4), alpha=34)
        show_gray_tensor("origin", org)
        show_gray_tensor("elastic", img)
        cv.waitKey()

    def testMorphClose(self):
        org = self.item()["X"]
        k = 15
        img = morph_close(org, k)

        org_np = (gray2numpy(org) * 255).astype("uint8")
        img_np = cv.morphologyEx(
            org_np, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (k, k))
        )

        cv.imshow("origin", org_np)
        show_gray_tensor("close", img)
        cv.imshow("close_cv", img_np)
        cv.waitKey()

    def testRotateScale(self):
        org = self.randomItem()["X"]
        img = affine(org, scale=0.5, angle=30)
        show_gray_tensor("org", org)
        show_gray_tensor("scale", img)
        cv.waitKey()

    def testTranslate(self):
        org = self.randomItem()["X"]
        img = affine(org, 0.1, -0.1)
        show_gray_tensor("org", org)
        show_gray_tensor("scale", img)
        cv.waitKey()
