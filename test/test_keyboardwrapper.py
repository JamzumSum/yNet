from unittest import TestCase
from common.utils import KeyboardInterruptWrapper

class KeyboardInterruptWrapperTest(TestCase):

    @property
    def react(self): 
        print('Successfully react.')
        return 1

    @KeyboardInterruptWrapper(lambda self, *args, **argv: self.react)
    def testSelf(self):
        raise KeyboardInterrupt