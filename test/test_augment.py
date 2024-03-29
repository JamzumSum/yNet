from random import randint
from unittest import TestCase

import cv2 as cv
import torch
from common.utils import CropUpsample, morph_close
from data.augment import elastic, affine, getGaussianFilter
from data.dataset.cacheset import CachedDatasetGroup
from data.augment.online import RandomAffine

gray2numpy = lambda tensor: (
    tensor[0, 0] if tensor.dim() == 4 else tensor[0] if tensor.dim() == 3 else tensor
    if tensor.dim() == 2 else None
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


class TestOnline(TestCase):
    def setUp(self):
        self.ds = CachedDatasetGroup("data/BUSI/")

    def randomItem(self):
        rint = randint(0, len(self.ds))
        print(rint)
        return self.ds[rint]

    def item(self, i=0):
        return self.ds[i]

    def testRandomAffine(self):
        ra = RandomAffine(0, .1, 1.5)
        for _ in range(20):
            d = self.randomItem()
            X, mask = ra(d['X'], d['mask'])
            show_gray_tensor('org', d['X'])
            show_gray_tensor('org_mask', d['mask'])
            show_gray_tensor('affine', X)
            show_gray_tensor('mask_affine', mask)
            cv.waitKey()


class TestCrop(TestCase):
    def setUp(self):
        self.ds = CachedDatasetGroup("data/BUSI/")

    def randomItem(self):
        rint = randint(0, len(self.ds))
        print(rint)
        return self.ds[rint]

    def item(self, i=0):
        return self.ds[i]

    def testCrop(self):
        o = self.randomItem()
        X = o['X'].unsqueeze(0)
        mask = o['mask'].unsqueeze(0)

        crop = CropUpsample(512)
        croped = crop.forward(mask, mask)

        self.assertTrue(croped.shape == (1, 1, 512, 512))

        show_gray_tensor('croped', croped)
        show_gray_tensor('X', X)
        show_gray_tensor('mask', mask)
        cv.waitKey()

    def testEmptyCrop(self):
        o = self.randomItem()
        X = o['X'].unsqueeze(0)
        mask = torch.zeros_like(X)

        crop = CropUpsample(512)
        croped = crop.forward(X, mask)

        self.assertTrue(croped.shape == (1, 1, 512, 512))

        show_gray_tensor('croped', croped)
        show_gray_tensor('X', X)
        cv.waitKey()