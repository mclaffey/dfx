#!/usr/bin/env python

import sys
import unittest
import io
import tempfile

import dfx
import dfx.datasets

class CommandLineTest(unittest.TestCase):
    """Test output of dfx command line
    """
    def setUp(self):
        """Create a variable to capture output
        """
        self._output = io.StringIO()
        
    def print(self, s, end='\n'):
        """Instead of printing to screen, save to out
        output variable.
        """
        self._output.write(str(s)+end)

    @property
    def output(self):
        self._output.seek(0)
        return self._output.read()
        
    def test_usage(self):
        """Without arguments, we get a bunch of usage text
        """
        dfx.main(['dfx'], print_func=self.print)
        arbitrary_long_length = 200
        self.assertTrue(len(self.output) > arbitrary_long_length)

    def test_value_patterns(self):
        """Create a csv, call dfx on it, check the value
        pattern output
        """
        f = tempfile.NamedTemporaryFile(delete=False)
        dfx.datasets.employees.to_csv(f.name, index=False)
        dfx.main(['dfx', f.name], print_func=self.print)
        expected="""employee_id         : id, num_normal, num long tail
region              : categorical, flag
state               : categorical
salary              : num_normal, num long tail
company             : categorical
manager_id          : categorical, num_normal
       """
        # ignore first line of output
        actual="\n".join(self.output.split('\n')[1:])
        self.assertEqual(actual, expected)
        
        

if __name__ == '__main__':
    unittest.main()
    #sys.exit(unittest_dfx.main(__file__))
        
