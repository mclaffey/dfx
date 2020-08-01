User Interface
**************

ui_curses addresses the problem of needing to quickly explore a new
data, using keyboard commands for common navigation and manipulations.

Call dfx on a file you want to explore::

  $ dfx my/data/obesity.csv

This produces a screen like this::

  (999, 5)        data/obesity.csv > row_number(0)

		  not randomize Jump False=

  Sort: []
  Reduced: Not reduced
  Value pattern: Done with 10 rows
  Grain (Done with 10 rows): Unique 100%. 47% Year. 53% Obesity. Year         Obesity       many:many


  row_number   Country       Year   Obesity            Sex
  ----------   -----------   ----   ----------------   ----------
  1            Afghanistan   1975   0.5 [0.2-1.1]      Both sexes
  2            Afghanistan   1975   0.2 [0.0-0.6]      Male
  3            Afghanistan   1975   0.8 [0.2-2.0]      Female
  4            Afghanistan   1976   0.5 [0.2-1.1]      Both sexes
  5            Afghanistan   1976   0.2 [0.0-0.7]      Male
  23           Afghanistan   1982   0.3 [0.1-0.9]      Male
  71           Afghanistan   1998   1.0 [0.4-2.2]      Male
  176          Albania       1991   8.2 [4.6-12.9]     Male
  406          Andorra       1984   17.4 [12.9-22.3]   Both sexes
  566          Angola        1995   1.0 [0.3-2.3]      Male
  681          Antigua and   1991   14.5 [9.1-20.9]    Female
  713          Antigua and   2002   7.3 [4.0-11.7]     Male
  744          Antigua and   2012   23.7 [16.4-31.8]   Female
  872          Argentina     2013   25.7 [19.7-32.3]   Male
  952          Armenia       1998   13.6 [10.3-17.4]   Both sexes
  995          Armenia       2012   15.2 [10.0-21.5]   Male
  996          Armenia       2012   21.2 [15.7-27.2]   Female
  997          Armenia       2013   18.8 [14.8-23.0]   Both sexes
  998          Armenia       2013   15.7 [10.3-22.1]   Male
  999          Armenia       2013   21.7 [16.0-27.8]   Female
  ----------   -----------   ----   ----------------   ----------
  id,          categorical   cate                      categorica
  normal,                    gori                      l
  num long                   cal,

   a A       sort ascending   l         load file                 m         melt                  s         selected on/off
   z Z       sort descending  g G       add to grain (G=remove)   n         next DfView           t         filter join to parent
   c         value counts     ^g        grain details             p         add column to pivot   u         scroll up
   d         scroll down      h         unhide columns            r         reduced detail        v         value patterns
   f         find             j         jump to row number        ^R        randomize on/off      y         unsort (row_number)
   <arrows>  move cursor      <enter>   filter to value           <space>   hide column           <         freeze column to left


The help at the bottom of the above screens indicates which commands are available.

Start by using the `left, right, up, down` arrows to move the cursor around the data set.

To hide a column that isn't important for now, hit the `<space>` bar.


.. note::

  Still need to document the following

  * column left
  * sort
  * find
  * jump
  * value counts
  * grain
  * simple filter
  * multi-select filter
  * pivot
  * melt
  * random rows
