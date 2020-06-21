#!/usr/bin/env python

import sys
import pickle
import os
import curses
import curses.textpad
import random
import textwrap
import pickle
import threading

import dfx.ops
import dfx.grain

TIMEOUT=300

pd = None # import on demand when needed
def import_pandas():
    global pd
    if pd:
        return
    import pandas
    pd = pandas
    pd.options.display.max_colwidth=12


class Environment(object):
    """Holds multiple DfViews and moves between them
    """
    def __init__(self):
        # dict of all DfViews, with keys file names or derived
        # names
        self.dfvs={}        
        # key of current Dfview
        self.current_dfv_name=None
        # directory to look for other csvs
        self.currend_dir=None

    @property
    def dfv(self):
        return self.dfvs[self.current_dfv_name]

    def next(self):
        """Cycle to next dfv by alphabetical name
        """
        names = sorted(self.dfvs.keys())
        i = names.index(self.current_dfv_name)
        if i == len(names)-1:
            i = 0
        else:
            i+=1
        self.current_dfv_name = names[i]

    def next_file(self):
        """Load the next csv in current directory that isn't in dfvs
        yet

        Returns the loaded file name. If no files were found, or all
        files have already been loaded, returns None
        """
        for f in sorted([os.path.join(self.current_dir, f) for f in
                         os.listdir(self.current_dir)
                         if f.endswith('.csv')]):
            if f not in self.dfvs:
                self.load_file(f)
                return f
        return None

    def load_file(self, path):
        """Load a specific file
        """
        if path in self.dfvs:
            raise KeyError('File already loaded')
        import_pandas()
        df=pd.read_csv(path)
        if 'now_number' not in df.columns:
            df=df.reset_index().rename(
                columns={'index': 'row_number'})
            df['row_number'] += 1
        self.dfvs[path]=DfView(df)
        self.current_dfv_name=path
        self.current_dir=os.path.dirname(path)

    def close(self):
        """Shut down DfView threads and save to disk
        """
        for dfv in self.dfvs.values():
            dfv.close()
        with open('env.pickle', 'wb') as f:
            pickle.dump(self, f)
        

class IncrementalDfComp(object):
    """Compute a function on the first 10, 100, 1000 rows
    of a dataframe
    """
    def __init__(self, df, func):
        self.df=df
        self.func=func
        self.results=[]
        self.message='Not started'
        self.done=False
        self._set_nrows()
        self._thread=None

    @property
    def result(self):
        if self.results:
            return self.results[0]
        else:
            return None

    def _set_nrows(self):
        i = 1
        self.nrows=[]
        while True:
            i *= 10
            if i < (0.8*self.df.shape[0]):
                self.nrows.append(i)
            else:
                self.nrows.append(self.df.shape[0])
                return
        
    def still_needed(self):
        """Tell the instance to calculate the next longest
        increment if one is available
        """
        if self.done:
            return
        if self._thread and self._thread.isAlive():
            return
        self._thread=threading.Thread(target=self.calc) 
        self._thread.start()
        
    def calc(self):
        if not self.nrows:
            raise ValueError('No more nrows')
        nrow=self.nrows[0]
        self.message='Running for {} rows'.format(nrow)
        self.results.insert(0, self.func(self.df[:nrow]))
        self.message='Done with {} rows'.format(nrow)
        self.nrows.remove(nrow)
        if not self.nrows:
            self.done=True
            self._thread=None

    def cancel(self):
        """TODO I only need this because I'm trying to pickle a
        DfView, and that can't pickle a thread. So for now I'm just
        clearing the reference and not worrying about the thread
        continuing to run in the background.
        """
        self._thread=None
        
class DfView(object):
    def __init__(self, df):
        self.reduced = dfx.ops.ReducedDf(df)
        self.df=self.reduced.df

        """
        a subset of rows (Pandas dataframe) to be displayed to the
        user. Updated by generate_preview()        
        """
        self.row_preview = None

        ## column settings
        
        self.col_selected=0
        self.cols_hidden = []
        self.cols_keep_left = []

        ## row settings
        """
        _preview_head_n/_preview_tail_n
          count of rows from begininning and end of dataframe
          to be displayed in preview
        _preview_mid_n
          count of rows besides head/tails to show
          
        """
        self._preview_head_n = 5
        self._preview_tail_n = 5
        self._preview_mid_n = 10
        self._preview_center_i = int(self.df.shape[0]/2)

        """
        Which columns to sort by, and ascending/descending

        This is a list of tuples, where each tuple is a column name
        and boolean, with the boolean True for ascending and False
        for descending. Set by .sort()
        """
        self._sort_fields = []

        """
        Not sure about this design, namely where to store the
        preference to randomize the row_preview. Rather than storing
        with DfView, it might be a property of the user environment
        across all DfViews.

        Regardless, this isn't currently linked to any behavior of
        DfView; it is up to the calling code to use:
          if dfv.randomize:
            dfx.set_preview(random=True)
        
        """
        self.randomize=True

        """
        integer offset of row that user has selected with cursor
        """
        self.row_selected=-1

        self.set_preview(randomize=True)

    @property
    def grain(self):
        if not hasattr(self, '_grain_comp'):
            self._grain_comp=IncrementalDfComp(
                self.df, dfx.grain.GrainDf)
        self._grain_comp.still_needed()
        return (
            self._grain_comp.result,
            self._grain_comp.done,
            self._grain_comp.message,
            )

    @property
    def value_patterns(self):
        if not hasattr(self, '_vp_comp'):
            self._vp_comp=IncrementalDfComp(
                self.df, dfx.ops.ValuePatternsDf)
        self._vp_comp.still_needed()
        return (
            self._vp_comp.result,
            self._vp_comp.done,
            self._vp_comp.message,
        )

    def close(self):
        """Stop any threads on IncrementalDfComps
        """
        if hasattr(self, '_grain_comp'):
            self._grain_comp.cancel()
        if hasattr(self, '_vp_comp'):
            self._vp_comp.cancel()

    @property
    def col_widths_orig(self):
        """Calculate the maximum string length of values in each
        column, using the first and last 5 rows
        """
        if not hasattr(self, '_col_width_orig'):
            preview_orig = self.row_preview
            mid_orig = self._preview_mid_n
            self._preview_mid_n=0
            df=self.set_preview()
            self.row_preview=preview_orig
            self._preview_mid_n=mid_orig
            
            self._col_widths_orig = {
                col_name: max([len(str(_)) for _ in col])
                for col_name, col in df.iteritems()}
        return self._col_widths_orig

    def sort(self, field, ascending=True, add=False):
        """Sort the dataframe for display in row_preview

        This method is called for one field at a time. Use add=True
        to add a second, third, etc sort field.

        :field - string. Name of column to sort by. If None,
          existing fields are discarded and sort is set to
          'row_number' ascending.        
        :ascending - boolean, default True. If True, sort
          ascending. If False, sort descending.
        :add - boolean, default False. If False, any existing
          sort fields will be discarded and new sort is only based
          on 'field'. If True, 'field' is added as a sort field
          after any existing fields.
        """
        self.randomize=False

        # no sort means sort by row number
        if field is None:
            self._sort_fields=[]
            self.df.sort_values('row_number', inplace=True)
            self.set_preview()
            return

        # set self._sort_fields
        if not add:
            self._sort_fields = [(field, ascending)]
        else:
            # remove if already in it
            for sort_field in self._sort_fields:
                if sort_field[0]==field:
                    self._sort_fields.remove(sort_field)
                    break
            self._sort_fields.append((field, ascending))

        # perform sort
        fields, ascendings = zip(*self._sort_fields)
        self.df.sort_values(list(fields),
                            ascending=ascendings,
                            inplace=True)

        # update preview
        self.set_preview()
    
    def set_preview(self,
                    center_i=None,
                    offset=None,
                    randomize=False,
    ):
        """Get a set of rows from the dataframe

        Typically this will be the first few rows, the last few rows,
        and sample of rows in between. The middle sample can either be
        set by specifying a row to center on (center_i=), by
        specifying a number of rows to move up or down (offset=), or
        by specifying self.randomize=True and a random sample will be
        selected.
        
        If center_i or offset are provided, the relevant rows are
        set. If these are not provided, the mid rows are selected
        by random sample.

        Only one of the following can be specified at a time:
          - center_i
          - offset
          - randomize
        
        :center_i - int. Optional, a row number. If provided, the
           mid_rows are selected so that they center on the specified
           row. First row is 1, not 0.
        :offset - int, default None. If not None and positive int,
           shift the middle rows down to show later row numbers. If
           negative, shift up.
        :randomize - bool. Default False. If True, middle rows
           will be randomly sampled from dataframe, but in order.

        """

        # check arguments
        arg_sum = sum([1 if _ else 0 for _ in
                       [randomize, center_i, offset]])
        if arg_sum > 1:        
            raise ValueError(('Can not specify more than one: '+\
                              'randomize={} '.format(randomize)+\
                              'center_i={} '.format(center_i)+\
                              'offset={}'.format(offset)))

        # if offset, call set_preview() again
        if offset is not None:
            return self.set_preview(
                center_i=self._preview_center_i+offset)
        
        # calculate some stuff
        nrow=self.df.shape[0]        # count of df rows
        head_n=self._preview_head_n
        tail_n=self._preview_tail_n
        head_i = list(range(head_n)) # indices of first rows to show
        tail_i = sorted([nrow - i - 1 for i in range(0, tail_n)])
        # get count for middle, unless dataframe is smaller than that
        mid_n = min(self._preview_mid_n, nrow-head_n-tail_n)
        
        # if small dataset, return as is
        if nrow <= (head_n+tail_n):
            self.row_preview = self.df
            return self.row_preview
           
        # if not asking for middle rows, return head+tail
        if mid_n <= 0:
            self.row_preview=self.df.iloc[head_i+tail_i]
            return self.row_preview

        # if no arguments, just refreshing
        if arg_sum == 0:
            if self.randomize:
                randomize=True
            else:
                center_i = self._preview_center_i

        # if center_i is being specified, grab rows before/after
        # it and updated row_selected
        if center_i is not None:
            # if more than number of rows, cap to last row
            if center_i >= nrow:
                center_i = nrow-1
            self._preview_center_i=center_i
            if center_i in head_i:
                self.row_selected = head_i.index(center_i)
                return self.row_preview
            elif center_i in tail_i:
                self.row_selected = head_n+mid_n+tail_i.index(center_i)-1
                return self.row_preview
            elif center_i >= (nrow - mid_n - tail_n):
                mid_i = list(range(nrow-mid_n-tail_n, nrow))
                self.row_selected = head_n + mid_i.index(center_i)
            elif center_i < (head_n + mid_n):
                mid_i = list(range(head_n, head_n+mid_n))
                self.row_selected = head_n + mid_i.index(center_i)
                
            else:
                start_i = center_i - int(mid_n/2)
                mid_i = list(range(
                    start_i,
                    start_i + mid_n,
                ))
                self.row_selected = head_n + int(mid_n/2)

        # random, grab that many random rows
        if randomize:
            mid_i=random.sample(
                range(head_n, nrow-tail_n),
                mid_n)
            rows_i = sorted(mid_i)
            # row_selected isn't updated. the rows on the screen will
            # be changing, but the cursor will stay in the same place

        # use mid_i to set row_preview
        rows_i = sorted(set(head_i + mid_i + tail_i)) # dedup
        #try:
        #    rows_i = set(head_i + mid_i + tail_i) # dedup
        #except UnboundLocalError as e:
        #    raise ValueError(center_i, head_i, tail_i, center_i in head_i)
        rows_i = list(filter(lambda x: 0 <= x < nrow, rows_i)) # cap
        self.row_preview=self.df.iloc[rows_i]
        return self.row_preview

    @property
    def reduced_short_desc(self):
        """Provide a brief (one line?) summary of reduced
        """
        if not self.reduced.reduced:
            return 'Not reduced'
        s = ""

        if self.reduced.constants:            
            s += 'Constants: ' + ",".join([
                '{}={:30.30s}'.format(k,v)
                for k,v in self.reduced.constants.items()])
        if self.reduced.zeros:
            s += " Zeros: " + ",".join(self.reduced.zeros)
        if self.reduced.nulls:
            s += " Nulls: " + ",".join(self.reduced.nulls)
        s = s.strip()

        return s

def rect(s):
    return "\n".join([_[:120] for _ in s.split("\n")[:40]])
        
def curses_loop(scr, env):
    key = ''
    scr.timeout(TIMEOUT)
    find_mode=False
    find_column=None
    find_value=None
    jump_mode=False
    jump_value=''

    while key != 'q':

        dfv=env.dfv
        df=dfv.df

        # row preview
        if find_column:
            col = dfv.df[find_column].apply(str)
            row_i = col.str.contains(find_value, case=False)
            dfv.set_preview(center_i=row_i)
        if dfv.randomize:
            dfv.set_preview(randomize=True)
        dfi=dfv.row_preview
        if dfi is None:
            raise RuntimeError()
        
        # columns to be displayed
        col_names = dfv.cols_keep_left +\
                    [_ for _ in df.columns
                     if _ not in dfv.cols_hidden 
                     and _ not in dfv.cols_keep_left]
            
        ### calculate column widths
        x_max = 140  # width of screen
        y_max = 45
        col_spacer=3 # spaces between columns
        col_min=3    # show atleast this many chars per column
        # initial widths based on longest values
        col_widths = {k:v+col_spacer for k,v in
                      dfv.col_widths_orig.items()                      
                      if k in col_names}        
        # keep reducing the longest column until they all fit or
        # are at the minimum width
        while sum(col_widths.values()) > x_max:
            col_max_name = ''
            col_max_width = 0
            for col_name, col_width in col_widths.items():
                if col_width > col_max_width:
                    col_max_width = col_width
                    col_max_name = col_name
            col_widths[col_max_name] -= 1
            if col_max_width==col_min+col_spacer:
                break
        col_widths = {k:v-col_spacer for k,v in col_widths.items()}

        # column headers
        col_headers = {}
        col_head_rows=3
        for col_name in col_names:
            col_head_strs = textwrap.wrap(col_name,
                                          width=col_widths[col_name])
            while len(col_head_strs) < col_head_rows:
                col_head_strs.insert(0, '')
            col_head_strs = col_head_strs[:col_head_rows]
            col_headers[col_name] = col_head_strs

        # column footers (value patterns)
        col_footers = {}
        col_foot_rows=3
        vp_result, vp_done, vp_message = dfv.value_patterns
        if not vp_result:
            col_footers = {_: ['']*col_foot_rows for _ in col_names}
        else:
            for col_name in col_names:
                vp = vp_result.value_patterns[col_name][0]
                vp_str = ", ".join(vp.keys())
                col_foot_strs = textwrap.wrap(vp_str,
                                              width=col_widths[col_name])
                col_foot_strs = col_foot_strs[:col_foot_rows]
                col_footers[col_name] = col_foot_strs
                
        # print header to screen
        scr.erase()
        y=0

        # screen line - which dfv
        line='{shape:15s} {dfv_name:} > {col_name:}({col_i:})'.format(
            shape=str(df.shape),
            dfv_name=env.current_dfv_name,
            col_name=col_names[dfv.col_selected],
            col_i=dfv.col_selected)
        scr.addstr(y, 0, line)
        y+=1

        # screen line - key, randomize on, find
        line='{:15s} {} {} {}'.format(
            key,
            'RANDOMIZE' if dfv.randomize else 'not randomize',
            'Find: {}={}'.format(find_column, find_value),
            'Jump {}={}'.format(jump_mode, jump_value),
            )
        scr.addstr(y, 0, line)
        y+=1

        # screen line - sort
        scr.addstr(y, 0, 'Sort: {}'.format(dfv._sort_fields))
        y+=1

        # screen line - reduced
        scr.addstr(y, 0, "Reduced: {}".format(dfv.reduced_short_desc)[:x_max])
        y+=1

        # screen line - value patterns
        scr.addstr(y, 0, "Value pattern: {}".format(vp_message))
        y+=1

        # screen line - grain
        grain_result, grain_done, grain_message = dfv.grain
        if not grain_result:
            grain_str = grain_message
        else:
            if grain_done:
                grain_message = ''
            else:
                grain_message = ' ({})'.format(grain_message)
            grain_str = 'Grain{}: {}'.format(
                grain_message,
                ". ".join(_.strip() for _ in
                          grain_result.summary.split('\n')).replace('  ', ' '))
        scr.addstr(y, 0, grain_str[:x_max])
        y+=1

        # print dataframe to screen
        x = 0
        data_top_y=y
        for col_i, col_name in enumerate(col_names):
            y = data_top_y
            col_w = col_widths[col_name]
            col_w = min(col_w, x_max-x) # not past right edge
            
            # column heading
            addstr_args=[]
            if col_i==dfv.col_selected:
                addstr_args.append(curses.A_REVERSE)
            for i, s in enumerate(col_headers[col_name]):
                scr.addstr(y+i, x, s[:col_w], *addstr_args)
            y+=col_head_rows

            # heading underling
            scr.addstr(y, x, '-'*col_w)
            y+=1

            # column values
            col = dfi[col_name]
            for i, val in enumerate(col):
                if y+i > y_max:
                    raise ValueError(y+i, dfi.shape)
                addstr_args=[]
                if i==dfv.row_selected and col_i==dfv.col_selected:
                    addstr_args.append(curses.A_REVERSE)
                scr.addstr(y+i,
                           x,
                           str(val)[:col_w],
                           *addstr_args)
            y+=col.size
                
            # footer underling
            scr.addstr(y, x, '-'*col_w)
            y+=1

            # column footers (value patterns)
            for i, s in enumerate(col_footers[col_name]):
                addstr_args=[]
                if col_i==dfv.col_selected:
                    addstr_args.append(curses.A_REVERSE)
                scr.addstr(y+i, x, s[:col_w], *addstr_args)
            y+=col_foot_rows
                       
            # increment for next column, but stop if already wide enough
            x+=col_w + col_spacer
            if x > x_max:
                break
            
        # get key and navigate
        try:
            key = scr.getkey()
        except curses.error:
            key = ''

        # find mode
        if find_mode and key != '':
            if key=='\n':
                find_mode=False
            else:
                find_value+=key
            key=''

        # jump mode
        if jump_mode and key != '':
            if key=='\n':
                jump_mode=False
                jump_value=''
            if key=='KEY_BACKSPACE':
                if jump_value:
                    jump_value=jump_value[:-1]                
            if key in '0123456789':
                jump_value+=key
                
            key=''                
            jump_int=None
            try:
                jump_int = int(jump_value)
            except ValueError:
                pass
            if jump_int is not None and jump_int != 0:
                jump_int -= 1
                dfv.set_preview(center_i=jump_int)

        # move around
        if key=='KEY_LEFT':
            if dfv.col_selected > 0:
                dfv.col_selected -= 1
        if key=='KEY_RIGHT':
            if dfv.col_selected < len(col_names)-1:
                dfv.col_selected += 1
        if key=='KEY_DOWN':
            dfv.row_selected+=1
        if key=='KEY_UP':
            dfv.row_selected-=1

        # a A - sort ascending
        if key=='a':
            col_name=col_names[dfv.col_selected]
            dfv.sort(col_name, add=True)
        if key=='A':
            col_name=col_names[dfv.col_selected]
            dfv.sort(col_name, add=False)
            
        # c - value counts
        if key=='c':
            vc=df[col_names[dfv.col_selected]].value_counts().head(40)
            s = rect(str(vc))
            scr.erase()
            scr.addstr(3,0,s)
            scr.timeout(0)
            scr.getkey()
            scr.timeout(TIMEOUT)

        # d - scroll down
        if key=='d':
            dfv.set_preview(offset=1)

        # f F - find
        if key=='f':
            find_mode=True
            find_value=''
        if key=='F':
            find_column=None
            find_value=None
            
        # g - show GrainDf details
        if key=='g':
            scr.erase()
            scr.addstr(0,0,dfv.grain.summary)
            scr.getkey()

        # h - unhide columns
        if key=='h':
            dfv.cols_hidden=[]

        # j - jump to row
        if key=='j':
            dfv.randomize=False
            jump_mode=True

        # l - load file
        if key=='l':
            env.next_file()

        # m - toggle randomization
        if key=='m':
            dfv.randomize=True
        if key=='M':
            dfv.randomize=False

        # n - next DfView
        if key=='n':
            env.next()

        # r- show ReducedDf details
        if key=='r' and dfv.reduced.reduced:
            scr.erase()
            y=0
            scr.addstr(y, 0, 'Reduced on {}'.format(env.current_dfv_name))
            y+=2
            for k,v in dfv.reduced.constants.items():
                scr.addstr(y, 0, '{:>20s} = {}'.format(k,v)[:x_max])
                y+=1
            if dfv.reduced.zeros:
                y+=1
                scr.addstr(y, 0, 'Zeros:')
                y+=1
                for col in dfv.reduced.zeros:
                    scr.addstr(y, 2, col)
            if dfv.reduced.nulls:
                y+=1
                scr.addstr(y, 0, 'Nulls:')
                y+=1
                for col in dfv.reduced.nulls:
                    scr.addstr(y, 2, col)
            scr.timeout(0)
            scr.getkey()
            scr.timeout(TIMEOUT)

        # u - scroll up
        if key=='u':
            dfv.set_preview(offset=-1)

        # v - show ValuePatternsDf details
        if key=='v':
            scr.erase()
            scr.addstr(0,0, str(dfv.value_patterns))
            scr.timeout(0)
            scr.getkey()
            scr.timeout(TIMEOUT)

        # y - remove sort
        if key=='y':
            dfv.sort(None)

        # z Z - sort descending
        if key=='z':
            col_name=col_names[dfv.col_selected]
            dfv.sort(col_name, ascending=False, add=True)
        if key=='Z':
            col_name=col_names[dfv.col_selected]
            dfv.sort(col_name, ascending=False, add=False)

        # ENTER, filter to value
        if key=='\n':
            col_name=col_names[dfv.col_selected]
            filter_val=dfi[col_name].values[dfv.row_selected]
            df_filt=df[df[col_name]==filter_val]
            dfv=DfView(df_filt)
            env.current_dfv_name=env.current_dfv_name +\
                ' > {}={}'.format(col_name, filter_val)
            env.dfvs[env.current_dfv_name]=dfv

        # hide column
        if key==' ':
            dfv.cols_hidden.append(col_names[dfv.col_selected])
            if dfv.col_selected == len(col_names)-1:
                dfv.col_selected -= 1

        # freeze column to left
        if key=='<':
            dfv.cols_keep_left.append(col_names[dfv.col_selected])

                
    # save state
    env.close()
                


def main():

    # if environment exists
    env_path = 'env.pickle'
    if os.path.exists(env_path):
        with open(env_path, 'rb') as f:
            env = pickle.load(f)
        res = curses.wrapper(curses_loop, env)
        print(res)
        sys.exit()

    # otherwise build environment from command line args, if any
    env = Environment()

    # determine path to start loading files from    
    if sys.argv[1:]:
        start_path = sys.argv[1]
    else:
        start_path = os.getcwd()
    if os.path.isfile(start_path):
        env.load_file(start_path)
    else:
        env.current_dir = start_path
        if not env.next_file():
            return 'No csvs found in {}'.format(
                os.path.abspath(env.current_dir))

    # run main loop    
    res = curses.wrapper(curses_loop, env)
    print(res)

if __name__ == '__main__':
    sys.exit(main())
    
