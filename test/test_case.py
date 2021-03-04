from unittest import TestCase
from misc import switch

class SwitchTest(TestCase):
    def testCase(self):
        i = 4
        with switch(i) as s:
            @s.case(1)
            def f():
                print(i)
                self.assertEqual(i, 1)
            
            @s.case(3)
            def f():
                print(i)
                self.assertEqual(i, 3)

            @s.default
            def f():
                print(i)
