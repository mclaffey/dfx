import sys
import unittest

"""
Command line and TestCase helpers for dfx

Add this to a test file:

  if __name__ == '__main__':
      sys.exit(unittest_dfx.main(__file__))


"""


class AbstractDfTestCase(unittest.TestCase):
    """Provide assert methods related to pandas
    """
    
    def assertEqualDf(self, act, exp):
        """add an .assert() method to test if two Pandas dataframes are equal

        The work is done by Pandas's assert_frame_equal function, but
        this method helps by displaying the dataframes if they aren't
        equal.

        """
        act = act.reset_index(drop=True).copy()
        exp = exp.reset_index(drop=True).copy()
        #if not act._indexed_same(exp):
        #    self.fail(act.index, exp.index)
        msg = [
            "--- Unequal dataframes: actual ---------------------"
            ,str(act)
            ,"--- Unequal dataframes: expected--------------------"
            ,str(exp)
            ]
        msg = "\n".join(msg)
        #print(msg)
        from pandas.testing import assert_frame_equal
        assert_frame_equal(act, exp)#, msg=msg)

    def assertEqualSeries(self, act, exp):
        self.assertEqualDf(act.to_frame(), exp.to_frame())

def report_tests(test_suite,
                 label="",
                 filter=None,
                 filter_i=None,
                 first_only=False,
                 spacer_lines=10):
    """Report tests of a test suite using either:
      single line per fail/error, or
      stack trace for first error

    verbose
      0 = single line, only print fails/errors
      1 = single line, all tests (fails/errors/success)
      2 = print stack
    """
    
    # print spacer lines
    [print("") for _ in range(spacer_lines)]

    # run tests
    res = unittest.TestResult()
    test_suite.run(res)

    # if successful, report and be done
    if res.wasSuccessful():
        print("SUCCESS " + label)
        return

    i = 0
    # errors
    if filter in [None, 'errors']:
        for test_case, stack_trace in res.errors:
            i += 1
            if not filter_i or filter_i == i:
                print(f"ERROR   {i:2d}  " + test_case.id())
                if first_only or filter_i == i:
                    print("")
                    print(stack_trace)
                if first_only:
                    return

    # failures
    if filter in [None, 'failures']:
        for test_case, stack_trace in res.failures:
            i += 1
            if not filter_i or filter_i == i:            
                print(f"FAIL    {i:2d}  " + test_case.id())
                if first_only or filter_i == i:
                    print("")
                    print(stack_trace)
            if first_only:
                return

def main(module_file_name):
    """

    > test.py
    # all tests, one line per test

    > test.py full
    # unittest.main(), full stacks for all tests

    > test.py list
    # lists tests without testing, one per line

    > test.py <other_args> errors
    # only show errors

    > test.py <other_args> failures
    # only show failures

    > test.py <other_args> first
    # show stack trace for the first error/failure and quit

    > test.py 1 (or any integer)
    # show stack trace for test #1 in previous listing

    > test.py SomeTestClass
    # show results for all tests for given class, one per line

    > test.py SomeTestClass.test_something
    # show stack trace for specific test    

    """

    # load all tests in the module
    module_name = module_file_name[2:-3]
    suite = unittest.defaultTestLoader.loadTestsFromName(module_name)    

    args = sys.argv[1:]
    
    # if no args, use brief on all tests
    if not args:
        report_tests(suite)
        return

    # if first arg = 'full', use unittest.main()
    if args[0] == 'full':
        unittest.main()
        return

    # if first arg is 'list', list tests and quit
    if args[0] == 'list':
        for kls in suite:
            for t in kls:
                print(f"{t.__class__.__name__}.{t._testMethodName}")
        return
                    
    # parse arguments
    first_only=False
    filter=None
    if 'first' in args:
        first_only = True
        args.remove('first')
    if 'errors' in args:
        filter='errors'
        args.remove('errors')
    if 'failures' in args:
        filter='failures'
        args.remove('failures')

    # see if there's an arg for a test number
    filter_i=None
    if args:
        try:
            filter_i = int(args[0])
        except ValueError as e:
            pass
        if filter_i:
            args.pop(0)            

    # see if remaining arguments are the name of a specific test
    # class or test
    if args:
        break_now = False
        for kls in suite:
            if break_now:
                break
            for test in kls:
                if break_now:
                    break
                class_name = test.__class__.__name__
                test_name = test._testMethodName
                if args[0] == test_name or \
                   args[0] == f"{class_name}.{test_name}":            
                    suite = test
                    first_only=True
                    args.pop(0)
                    break_now = True
                    break
                if args[0] == class_name:
                    suite = kls
                    args.pop(0)
                    break_now = True
                    break

    # if still unprocessed arguments, complain
    if args:
        return f"Unrecognized arguments: {args}"                    
                    
    # finally, run tests
    report_tests(suite,
                 filter=filter,
                 filter_i=filter_i,
                 first_only=first_only)
