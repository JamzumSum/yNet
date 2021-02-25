from unittest import TestCase

import cv2 as cv
from data.augment import elastic, ElasticAugmentSet
from data.dataset import CachedDatasetGroup
from common.utils import morph_close


class TestAugment(TestCase):
    def testElastic(self):
        o = CachedDatasetGroup('data/BIRADs/ourset.pt')
        org = o.__getitem__(0)['X']
        img = elastic(org, *ElasticAugmentSet.getFilter(4), alpha=34)
        cv.imshow('origin', org.numpy()[0])
        cv.imshow('elastic', img.numpy()[0])
        cv.waitKey()

    def testMorphClose(self):
        o = CachedDatasetGroup('data/BIRADs/ourset.pt')
        org = o.__getitem__(0)['X']
        k = 15
        img = morph_close(org, k)

        org_np = (org[0].numpy() * 255).astype('uint8')
        img_np = cv.morphologyEx(org_np, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (k, k)))

        cv.imshow('origin', org_np)
        cv.imshow('close', img.numpy()[0])
        cv.imshow('close_cv', img_np)
        cv.waitKey()
