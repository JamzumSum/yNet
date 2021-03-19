from unittest import TestCase
from common.decorators import autoProperty

class Foo:
    ko: int; co: int; da: int; yo: int
    @autoProperty
    def __init__(self, ko, co, da, yo=4):
        print('init:', locals())

class TestAutoProperty(TestCase):
    def testKW(self):
        kls = Foo(ko=1, co=2, da=3, yo=4)
        self.assertEqual(kls.ko, 1)
        self.assertEqual(kls.co, 2)
        self.assertEqual(kls.da, 3)
        self.assertEqual(kls.yo, 4)

    def testPW(self):
        kls = Foo(1, 2, 3, 4)
        self.assertEqual(kls.ko, 1)
        self.assertEqual(kls.co, 2)
        self.assertEqual(kls.da, 3)
        self.assertEqual(kls.yo, 4)