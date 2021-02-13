from unittest import TestCase

import cv2 as cv
from data.augment import ElasticAugmentSet
from data.dataset import CachedDatasetGroup


class TestAugment(TestCase):
    def testElastic(self):
        o = CachedDatasetGroup('data/BIRADs/ourset.pt')
        a = ElasticAugmentSet(o, 'Ym', kernel=45, alpha=300, sigma=25)
        org = o.__getitem__(0)['X']
        img = a.elastic(org)
        cv.imshow('origin', org.numpy()[0])
        cv.imshow('elastic', img.numpy()[0])
        cv.waitKey()