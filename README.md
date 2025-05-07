# gradesheetmaker
A simple python script that helps automate the gradesheet creation task for given class late/grading policies. Optimized for use with Gradescope at Caltech.

### How to use

1) Download relevant documents. Primarily this is the grades from gradescope (under "assignments" tab) in csv form. Optionally, if you want to distinguish between seniors and non-seniors, you will have to also download the class roll sheet from REGIS.
2) Download and import the main file `gradesheetmaker.py`.
3) The basic loop is to construct a `Gradesheet` class and then add a series of class late or grading policies in sequential order.
4) There are also a couple functions to view the grade distributions or to export results in excel sheet or csv form.

See the Jupyter Notebook `example.ipynb` for example usage.

### Contributing

Pull requests are welcome. If you want to make your own class policies that are not included by default, inherit from the `Policy` abstract class.

