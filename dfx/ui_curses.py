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
        self.dfvs[path]=DfView(df)
        self.current_dfv_name=path

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
        self.col_selected=0
        self.row_selected=-1
        self.cols_hidden = []
        self.cols_keep_left = []

        # settings
        self._randomize=False
        self.mid_i=[]

    @property
    def randomize(self):
        return self._randomize

    @randomize.setter
    def randomize(self, val):
        self._randomize=val
        
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
            df=self.row_preview(mid_max=0)
            self._col_widths_orig = {
                col_name: max([len(str(_)) for _ in col])
                for col_name, col in df.iteritems()}
        return self._col_widths_orig

    def row_preview(self, mid_max=10):
        """Return head+tail, and optionally randomize middle rows

        :mid_max - int. If greater than zero, up to this many rows
           will be randomly selected to be included. If self.randomize
           is True, or mid_max is greater than past mid_max, these
           rows will be newly randomly selected.

        """
        head_n=5
        head_i = list(range(head_n))
        tail_n=5
        tail_i = list(range(-tail_n, 0))
        nrow=self.df.shape[0]

        # if small dataset or not asking for mid, return
        if nrow <= (head_n+tail_n):
            return self.df        
        elif not mid_max:
            return self.df.iloc[head_i+tail_i]

        # if self.randomize, generate a new sample for mid
        if self.randomize or mid_max > len(self.mid_i):
            random_n=min(mid_max, nrow-head_n-tail_n)
            self.mid_i=random.sample(
                range(head_n, nrow-tail_n),
                random_n)

        mid_i = self.mid_i[:mid_max]        
        return self.df.iloc[head_i + mid_i + tail_i]

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
    scr.timeout(300)
    find_mode=False
    find_column=None
    find_value=None

    while key != 'q':

        dfv=env.dfv
        df=dfv.df
        if find_column:
            col = dfv.df[find_column].apply(str)
            row_i = col.str.contains(find_value, case=False)
            dfi=dfv.df[row_i]
            dfi=dfi[:10]
        else:
            dfi=dfv.row_preview()
        
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
        line='{:10s} {} {}'.format(
            key,
            'RANDOMIZE' if dfv.randomize else 'not randomize',
            'Find: {}={}'.format(find_column, find_value),
            )
        scr.addstr(y, 0, line)
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

        if find_mode and key != '':
            if key=='\n':
                find_mode=False
            else:
                find_value+=key
            key=''
            

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

        # find
        if key=='a':
            find_mode=True
            find_value=''
        if key=='A':
            find_column=None
            find_value=None
            
        # value counts
        if key=='c':
            vc=df[col_names[dfv.col_selected]].value_counts().head(40)
            s = rect(str(vc))
            scr.erase()
            scr.addstr(3,0,s)
            scr.getkey()

        # load file
        if key=='f':
            env.next_file()

        # show GrainDf details
        if key=='g':
            scr.erase()
            scr.addstr(0,0,dfv.grain.summary)
            scr.getkey()

        # next DfView
        if key=='n':
            env.next()

        # show ReducedDf details
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
                    scr.getkey()

        # unhide columns
        if key=='u':
            dfv.cols_hidden=[]

        # show ValuePatternsDf details
        if key=='v':
            scr.erase()
            scr.addstr(0,0, str(dfv.value_patterns))
            scr.getkey()

        # toggle randomization
        if key=='z':
            dfv.randomize=~dfv.randomize

        # ENTER, filter to value
        if key=='\n':
            col_name=col_names[dfv.col_selected]
            filter_val=dfi[col_name].values[dfv.row_selected]
            df_filt=df[df[col_name]==filter_val]
            dfv=DfView(df_filt)
            env.current_dfv_name=env.current_dfv_name + ' > {}={}'.format(col_name, filter_val)
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
    
