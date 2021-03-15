from random import randint
from unittest import TestCase

import cv2 as cv
import torch
from common.utils import BoundingBoxCrop, morph_close
from data.augment import elastic, scale, translate, getGaussianFilter
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

    def testBoundingBox(self):
        org = self.randomItem()
        mask = org["mask"]
        p = torch.randn_like(mask).abs()
        p /= p.max() / 0.4
        mask = mask + p
        u = 32
        crop = BoundingBoxCrop(0.5, u)

        roi = crop(mask.unsqueeze(0))[0][0]

        self.assertTrue(any(i > 0 for i in roi.shape))
        self.assertTrue(roi.size(0) % u == 0)
        self.assertTrue(roi.size(1) % u == 0)

        show_gray_tensor("mask", mask)
        show_gray_tensor("roi", roi)
        cv.waitKey()

    def testScale(self):
        org = self.randomItem()["X"]
        img = scale(org, 0.5)
        show_gray_tensor("org", org)
        show_gray_tensor("scale", img)
        cv.waitKey()

    def testTranslate(self):
        org = self.randomItem()["X"]
        img = translate(org, 0.1, -0.1)
        show_gray_tensor("org", org)
        show_gray_tensor("scale", img)
        cv.waitKey()
