from unittest import TestCase
from misc.indexserial import indexDumpAll, IndexLoader
from random import randint

class TestSerialize(TestCase):
    def testLoad(self):
        data = [randint(0, 10) for i in range(100)]
        idx = indexDumpAll(data, 'tmp/test/serialize.pt')
        
        f = IndexLoader('tmp/test/serialize.pt', idx)
        ld = f.loadAll()
        self.assertEquals(data, ld)

    def testRandomRead(self):
        data = [randint(0, 10) for i in range(100)]
        idx = indexDumpAll(data, 'tmp/test/serialize.pt')
        f = IndexLoader('tmp/test/serialize.pt', idx)

        i = randint(0, 100)
        self.assertEqual(data[i], f.load(i))