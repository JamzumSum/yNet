import os
from io import BytesIO

from torch import load, save

class IndexDumper:
    def __init__(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._f = open(filepath, 'wb')

    def dump(self, item)-> int:
        i = self._f.tell()
        save(item, self._f)
        return i

    def dumpAll(self, ls: list)-> list:
        return [self.dump(i) for i in ls]

    def __del__(self):
        if not self._f.closed: self._f.close()

class IndexLoader:
    def __init__(self, filepath: str, index: list):
        self._f = self._f = open(filepath, 'rb')
        self._index = index

    def load(self, i):
        self._f.seek(self._index[i])
        if i + 1 < len(self._index): 
            buf = BytesIO(self._f.read(self._index[i + 1] - self._index[i]))
        else:
            buf = BytesIO(self._f.read())
        return load(buf)

    def loadAll(self):
        return [self.load(i) for i in self._index]
        
    def __del__(self):
        if not self._f.closed: self._f.close()

def indexDumpAll(ls, filepath)->list:
    return IndexDumper(filepath).dumpAll(ls)

def indexLoadAll(filepath, index)-> list:
    return IndexLoader(filepath, index).loadAll()