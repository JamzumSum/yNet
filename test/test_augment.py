from unittest import TestCase

import cv2 as cv
from data.augment import ElasticAugmentSet
from data.dataset import CachedDatasetGroup


class TestAugment(TestCase):
    def testElastic(self):
        o = CachedDatasetGroup('data/BIRADs/ourset.pt')
        org = o.__getitem__(0)['X']
        img = ElasticAugmentSet.elastic(org, ElasticAugmentSet.getFilter(4))
        cv.imshow('origin', org.numpy()[0])
        cv.imshow('elastic', img.numpy()[0])
        cv.waitKey()