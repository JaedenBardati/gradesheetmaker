"""
The purpose of this code is to streamline and automate the making of gradesheets for large classes of students.
It primarily takes in Gradescope grades csv file (downloaded from the "assignments" tab) along with a series of 
class policies (including e.g. extension/late policy) that you can use to customize this to your particular class. 
Optionally it also takes in the class roll sheet downloadable from REGIS to distinguish between seniors or not.


There are two methods of use:

1) INTERACTIVE MODE (recommended) -- Here you are lead through the process interactively and can select from a list of predefined class policies. To access this, launch the python file main code (e.g. call main() or run python directly on this file). Something like:

        python gradesheet.py

Follow the onscreen instructions to contruct the gradesheet.


2) MANUAL MODE -- Here you create a Gradesheet class yourself and can create custom class policies. Use this method if you need to make new features to the code (e.g. new policies). See example code for common usages. The basic usage is:

        import gradesheet as gspy 
        gs = gspy.Gradesheet('Waves_Quantum_Mechanics_and_Statistical_Physics_Spring_2025_grades.csv', 'ClassRollSheet.csv')
        print('Inferred evaluations:', gs.evaluations)
        
        late_policy = gspy.SleepDayLatePolicy()
        gs.apply_policy(late_policy, ['PS1', 'PS2', 'PS3'])
        
        grade_policy = gspy.WeightedFinalGrade([0.2, 0.2, 0.2, 0.4])
        gs.apply_policy(grade_policy, ['PS1', 'PS2', 'PS3', 'Final Exam'])
        
        gs.to_excel()

Note that the policies will run in the order that they are applied. Thus, you should generally leave any grading policies to the end.


Requires numpy, pandas, matplotlib.

I made this while Head TAing for Ph2c 2025 at Caltech. I hope others can find it useful too :)
Please contact me if you have any questions or want to suggest adding support for a certain policy.
Note that the current state of the code is very messy and I could definitely be using better dev practices.

Disclaimer: Please note that I (or any other contributor) do not assume responsibility for grading mistakes as 
a consequence of this code. It is your responsibility to double check the results. In using this code, you also 
acknowlege that it is your responsibility to maintain FERPA compliance and you assume full responsibility for
any breach of student privacy. 

Jaeden Bardati 2025
"""


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings
from abc import ABC, abstractmethod
from collections import Counter


_NON_INTERACTIVE = True  # if set to true, removes all interactive features


class Gradesheet:
    """
    A python class that automates creation of the gradesheet using Gradescope grades export (and optionally REGIS export). 
    """
    
    # Gradescope export structure (change this if ever this structure is updated):
    _GRADESCOPE_LAST_NAME = 'Last Name'
    _GRADESCOPE_FIRST_NAME = 'First Name'
    _GRADESCOPE_SECTION = 'Sections'
    _GRADESCOPE_SID = 'SID'
    _GRADESCOPE_EMAIL = 'Email'
    _GRADESCOPE_EVAL_MATCH_MAX_POINTS = ' - Max Points'
    _GRADESCOPE_EVAL_MATCH_SUBTIME = ' - Submission Time'
    _GRADESCOPE_EVAL_MATCH_LATENESS = ' - Lateness (H:M:S)'
    _GRADESCOPE_TOTAL_LATENESS = 'Total Lateness (H:M:S)'
    
    # REGIS Roll sheet export structure (change this if ever this structure is updated):
    _ROLL_SHEET_INSTRUCTOR_NAME = 'Instructor Name'
    _ROLL_SHEET_COURSE_NAME = 'Course name'
    _ROLL_SHEET_SECTION_NUMBER = 'Section Number'
    _ROLL_SHEET_FULL_NAME = 'Name'
    _ROLL_SHEET_SID = 'Caltech UID'
    _ROLL_SHEET_EMAIL = 'Email'
    _ROLL_SHEET_ADVISER = 'Adviser'
    _ROLL_SHEET_CLASS = 'Class'
    _ROLL_SHEET_OPTION = 'Option'
    _ROLL_SHEET_UNITS = 'Units'
    _ROLL_SHEET_GRADING_SCHEME = 'Grading Scheme'
    _ROLL_SHEET_TYPE = 'Type'

    _ROLL_SHEET_CLASS_SENIOR = 'Senior'
    
    
    def __init__(self, gradescope_csv_filename, class_roll_sheet_filename=None, evaluations='infer', load_files=True):
        """
        Initializes the gradesheet object.
        It uses the grades csv exported from Gradescope and (optionally) the class roll sheet obtained from REGIS.
        You need the class roll sheet only if you want to destinguish between classes (e.g. seniors).
        You can manually specify all the evaluation names in a list form or just let it be infered from the gradescope csv.
        """
        self._gradescope = pd.read_csv(gradescope_csv_filename)
        self._roll_sheet = None if class_roll_sheet_filename is None else pd.read_csv(class_roll_sheet_filename)
        self._warn_if_potential_conflict_gradescope_roll_sheet()
        
        if evaluations == 'infer':
            self.evaluations = self._infer_evaluations()
        else:
            self.evaluations = evaluations
        
        self.master = self._basic_mastersheet()
        self.policies = []

    
    def _warn_if_potential_conflict_gradescope_roll_sheet(self):
        if self._GRADESCOPE_LAST_NAME not in self._gradescope:
            raise Exception('Gradescope Last Name column not found. Format has possibly changed.')
        if self._GRADESCOPE_FIRST_NAME not in self._gradescope:
            raise Exception('Gradescope First Name column not found. Format has possibly changed.')
        if self._GRADESCOPE_SECTION not in self._gradescope:
            raise Exception('Gradescope Sections column not found. Format has possibly changed.')
        if self._GRADESCOPE_SID not in self._gradescope:
            raise Exception('Gradescope SID column not found. Format has possibly changed.')
        if self._GRADESCOPE_EMAIL not in self._gradescope:
            raise Exception('Gradescope Email column not found. Format has possibly changed.')
        if self._GRADESCOPE_TOTAL_LATENESS not in self._gradescope:
            warnings.warn('Gradescope Total Lateness column not found. Format has possibly changed.')
        if len(self._gradescope) == 0:
                raise Exception('There are no students in the gradescope csv file. Something may be wrong with the file.')

        if self._roll_sheet is not None:
            if self._ROLL_SHEET_INSTRUCTOR_NAME not in self._roll_sheet:
                warnings.warn('Roll Sheet Intructor Name column not found. Format has possibly changed.')
            if self._ROLL_SHEET_COURSE_NAME not in self._roll_sheet:
                warnings.warn('Roll Sheet Course Name column not found. Format has possibly changed.')
            if self._ROLL_SHEET_SECTION_NUMBER not in self._roll_sheet:
                warnings.warn('Roll Sheet Section Number column not found. Format has possibly changed.')
            if self._ROLL_SHEET_FULL_NAME not in self._roll_sheet:
                warnings.warn('Roll Sheet Full Name column not found. Format has possibly changed.')
            if self._ROLL_SHEET_SID not in self._roll_sheet:
                raise Exception('Roll Sheet SID column not found. Format has possibly changed.')
            if self._ROLL_SHEET_EMAIL not in self._roll_sheet:
                raise Exception('Roll Sheet Email column not found. Format has possibly changed.')
            if self._ROLL_SHEET_ADVISER not in self._roll_sheet:
                warnings.warn('Roll Sheet Adviser column not found. Format has possibly changed.')
            if self._ROLL_SHEET_CLASS not in self._roll_sheet:
                raise Exception('Roll Sheet Class column not found. Format has possibly changed.')
            if self._ROLL_SHEET_OPTION not in self._roll_sheet:
                warnings.warn('Roll Sheet Option column not found. Format has possibly changed.')
            if self._ROLL_SHEET_UNITS not in self._roll_sheet:
                warnings.warn('Roll Sheet Units column not found. Format has possibly changed.')
            if self._ROLL_SHEET_GRADING_SCHEME not in self._roll_sheet:
                warnings.warn('Roll Sheet Grading Scheme column not found. Format has possibly changed.')
            if self._ROLL_SHEET_TYPE not in self._roll_sheet:
                raise Exception('Roll Sheet Type column not found. Format has possibly changed.')
            
            _students = self._roll_sheet[self._roll_sheet[self._ROLL_SHEET_TYPE] == 'Student']
            if len(_students) == 0:
                raise Exception('There are no students in the class roll sheet. Something may be wrong with the file or format.')

            _students_enrolled_not_ingradescope = _students[~_students[self._ROLL_SHEET_SID].isin(self._gradescope[self._GRADESCOPE_SID])]
            if len(_students_enrolled_not_ingradescope) != 0:
                warnings.warn('Some students are enrolled in the roll sheet but not in the Gradescope. You may need to sync the Gradescope roster to Canvas.')
                print('Emails of students enrolled, but not in Gradescope:', list(_students_enrolled_not_ingradescope[self._ROLL_SHEET_EMAIL]))
                self._roll_sheet = self._roll_sheet.drop(_students_enrolled_not_ingradescope.index, axis=0)
                if _NON_INTERACTIVE or input("Remove problematic students from this analysis? (y/n)") == 'y':
                    _students = self._roll_sheet[self._roll_sheet[self._ROLL_SHEET_TYPE] == 'Student']
                    print("Removed problematic students.")
            
            _students_ingradescope_not_enrolled = self._gradescope[~self._gradescope[self._GRADESCOPE_SID].isin(_students[self._ROLL_SHEET_SID])]
            if len(_students_ingradescope_not_enrolled) != 0:
                warnings.warn("Some students are in gradescope but not enrolled in the roll sheet. You may need to sync the Gradescope roster to Canvas, but it's also possible these are just auditors or peer tutors of the course, so check Canvas before removing them.")
                print('Emails of students in gradescope, but not in roll sheet:', list(_students_ingradescope_not_enrolled[self._GRADESCOPE_EMAIL]))
                if _NON_INTERACTIVE or input("Remove problematic students from this analysis? (y/n)") == 'y':
                    self._gradescope = self._gradescope.drop(_students_ingradescope_not_enrolled.index, axis=0)
                    print("Removed problematic students.")
    
    def _infer_evaluations(self):
        _match = self._GRADESCOPE_EVAL_MATCH_SUBTIME  # somewhat arbituary, but less likely to have conflict 
        _evals = [c[:-len(_match)] for c in self._gradescope.columns if c[-len(_match):] == _match]
        
        if len(_evals) == 0:
            raise Exception('Inferred zero evaluations in gradescope csv according to match filter. Gradescope format has possibly changed.')
        
        return _evals
    
    def _basic_mastersheet(self):
        # make a new dataframe to hold all relevant stuff for each student
        master = pd.DataFrame()
        master['SID'] = self._gradescope[self._GRADESCOPE_SID].copy()
        master['Last Name'] = self._gradescope[self._GRADESCOPE_LAST_NAME].copy()
        master['First Name'] = self._gradescope[self._GRADESCOPE_FIRST_NAME].copy()
        master['Section'] = self._gradescope[self._GRADESCOPE_SECTION].copy()
        master['Email'] = self._gradescope[self._GRADESCOPE_EMAIL].copy()
        
        # add columns for assignments/exams grades and lateness
        for a in self.evaluations:
            # assignment grade - giving NaNs a grade of 0
            master[a + ' grade (%)'] = np.nan_to_num(100*self._gradescope[a]/self._gradescope[a + self._GRADESCOPE_EVAL_MATCH_MAX_POINTS]) 
            
            # lateness in seconds, gradescope defaults lateness to 0 if there is no submission
            master[a + ' lateness (seconds)'] = self._gradescope[a + self._GRADESCOPE_EVAL_MATCH_LATENESS].apply(
                lambda t: int(t.split(':')[0])*3600 + int(t.split(':')[1])*60 + int(t.split(':')[2])
            )
        
        # define who are seniors
        if self._roll_sheet is not None:
            senior_uids = self._roll_sheet[self._roll_sheet[self._ROLL_SHEET_CLASS]==self._ROLL_SHEET_CLASS_SENIOR][self._ROLL_SHEET_SID]
            master['is senior'] = master.apply(lambda row: any(row['SID'] == suid for suid in senior_uids), axis=1)
        
        # sort in a way that is easy to read off in REGIS when entering grades
        master = master.sort_values(by=['Section', 'Last Name', 'First Name'])
        master = master.set_index('SID', drop=True)

        return master

    
    def apply_policy(self, policy, selected_evaluations=None, **kwargs):
        if not isinstance(policy, Policy):
            raise TypeError('The attemped applied policy of type {} does not inherit from the Policy abstract class.'.format(type(policy)))
        policy(self, selected_evaluations, **kwargs)
        self.policies.append((policy, selected_evaluations))

    def apply_policies(self, policy_list, selected_evaluations_list=None, **kwargs):
        if selected_evaluations_list is None:
            selected_evaluations_list = [None,]*len(policy_list)
        elif len(selected_evaluations_list) != len(policy_list):
            is_single_list = False
            try:
                len(selected_evaluations_list[0])
                if type(selected_evaluations_list[0]) == str:
                    is_single_list = True
            except TypeError:
                is_single_list = True
            if not is_single_list:
                raise Exception('Selected evaluations list length did not match the policy list length.')
            else:
                selected_evaluations_list = [selected_evaluations_list,]*len(policy_list)
        for policy, selected_evaluations in zip(policy_list, selected_evaluations_list):
            self.apply_policy(policy, selected_evaluations, **kwargs)

    
    def to_excel(self, filename='gradesheet.xlsx'):
        self.master.to_excel(filename)

    def to_csv(self, filename='gradesheet.csv'):
        self.master.to_csv(filename)

    def print_stats(self, keys='final grade (%)'):
        if type(keys) is str:
            keys = [keys,]
        for key in keys:
            if key not in self.master.columns:
                raise ValueError('There is no key "{}" in the master gradesheet dataframe.'.format(key))
            print('----', key, '----')
            name_functions = [
                ("Mean", np.mean),
                ("Median", np.median),
                ("Stdev", np.std),
                ("Min", np.min),
                ("Max", np.max),
            ]
            for name, function in name_functions:
                try:
                    print(name + ':', function(self.master[key]))
                except:
                    print(name + ':', 'Failed compute')
                    
    def display_dist(self, keys='final grade (%)', bins=20, order=None):
        if type(keys) is str:
            keys = [keys,]
        for key in keys:
            if key not in self.master.columns:
                raise ValueError('There is no key "{}" in the master gradesheet dataframe.'.format(key))
            fig, ax = plt.subplots()
            if order is None:
                plt.hist(self.master[key], bins=bins)
            else:
                _actual_order = [o for o in order if o in list(self.master[key])]
                if len(_actual_order) == 0:
                    raise ValueError('No element in order shows up in the provided key "{}"'.format(key))
                self.master[key].value_counts().loc[_actual_order].plot.bar()
                plt.gca().yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            plt.xlabel(key)
            plt.show()
    

class Policy(ABC):
    """
    Abstract method for a general policy. If you construct your own policy, it should inherit from this class.
    """
    MASTER_REQUIRES = [] # Abstract property for a list of master df column requirements
    MASTER_CREATES = [] # Abstract property for a list of master df columns that the application of this policy creates
    
    @abstractmethod
    def _apply(self, gs, selected_evaluations):
        pass

    def __call__(self, gs, selected_evaluations=None):
        # Apply the policy to the passed gradesheet
        self.resolve_conflicts(gs, selected_evaluations)
        self._apply(gs, selected_evaluations)

    def resolve_conflicts(self, gs, selected_evaluations):
        # Resolve any conflicts with the requirements of the file
        for r in self.MASTER_REQUIRES:
            if r not in gs.master.columns:
                raise Exception('Policy "{}" requires the column {} in the gradesheet prior to application.'.format(type(self).__name__, r))
        # Resolve any conflicts with the selected evaluations
        if selected_evaluations is not None:
            for evaluation in selected_evaluations:
                if evaluation not in gs.evaluations:
                    raise Exception("Evaluation '{}' is not in the list of evaluations in this gradesheet.".format(evaluation))


class GracePeriodLatePolicy(Policy):
    """
    Gives a grace period of a given amount of time. Functionally this means subtracting a certain time from all late students.
    """
    def __init__(self, grace_period_min, only_initially=False):
        self.grace_period_min = grace_period_min
        self.only_initially = only_initially
    
    def _apply(self, gs, selected_evaluations):
        for evaluation in selected_evaluations:
            if self.only_initially:
                gs.master[evaluation + ' lateness (seconds)'] = gs.master[evaluation + ' lateness (seconds)'].apply(lambda l: l if l > self.grace_period_min*60 else 0)
            else:
                gs.master[evaluation + ' lateness (seconds)'] = np.maximum(gs.master[evaluation + ' lateness (seconds)'] - self.grace_period_min*60, 0)



class SleepDayLatePolicy(Policy):
    """
    Applies a late policy based on a number of allowed "sleep days."
    Under this policy, students are given a certain amount of sleep days. 
    For each day that they are late, they consume a sleep day. For every
    3 days that they are late, their grade reduces by a factor of 0.7  
    """
    MASTER_REQUIRES = [] # note, can optionally require is senior
    MASTER_CREATES = ['sleep days left', '{} sleep days consumed']
    
    def __init__(self, sleep_day_factor=0.7, sleep_day_factor_period_days=3, default_number_of_sleep_days=5, default_number_of_sleep_days_for_seniors=None, extra_sleep_days_for_sids=dict()):
        if default_number_of_sleep_days_for_seniors is not None:
            self.MASTER_REQUIRES.append('is senior')
        self.sleep_day_factor = sleep_day_factor
        self.sleep_day_factor_period_days = sleep_day_factor_period_days
        self.default_number_of_sleep_days = default_number_of_sleep_days
        self.default_number_of_sleep_days_for_seniors = default_number_of_sleep_days_for_seniors
        self.extra_sleep_days_for_sids = extra_sleep_days_for_sids
        self._initialized = False

    def _apply(self, gs, selected_evaluations, reset_initial=False):
        # If first is True, sets the number of initial sleep days. 
        if reset_initial or not self._initialized:
            if self.default_number_of_sleep_days_for_seniors is not None:
                gs.master['sleep days left'] = self.default_number_of_sleep_days*~gs.master['is senior'] + self.default_number_of_sleep_days_for_seniors*gs.master['is senior']
            else:
                gs.master['sleep days left'] = self.default_number_of_sleep_days
    
            # add extra sleep days for exceptional cases
            for sid, ed in self.extra_sleep_days_for_sids.items():
                gs.master.loc[sid, 'sleep days left'] += ed
            
            for a in selected_evaluations:
                gs.master[a + ' sleep days consumed'] = 0

            self._initialized = True
        
        # reduce down sleep days for each assignment, scaling grade whenever neccessary
        def apply_sleep_days(row):
            sdl = row['sleep days left']
            for a in selected_evaluations: # for each assignment
                secs_late = row[a + ' lateness (seconds)']
                if secs_late > 0: # if late on assignment
                    days_late = secs_late/86400
                    if sdl < days_late: # if done with sleep days
                        periods_over = int(np.ceil((days_late - sdl)/self.sleep_day_factor_period_days))
                        row[a + ' grade (%)'] *= self.sleep_day_factor**periods_over
                        row[a + ' sleep days consumed'] = sdl
                        sdl = 0
                    else: # if still has sleep days left
                        sd_consumed = int(np.ceil(days_late))
                        row[a + ' sleep days consumed'] = sd_consumed
                        sdl -= sd_consumed # consume some sleep days
            row['sleep days left'] = sdl
            return row
        
        gs.master = gs.master.apply(apply_sleep_days, axis=1)


def _for_seniors_wrapper(apply_func, for_seniors):
    if for_seniors is None:
        return apply_func
    if for_seniors:
        def _wrapper(row, *args, **kwargs):
            if row['is senior']:
                return apply_func(row, *args, **kwargs)
            return row
    else:
        def _wrapper(row, *args, **kwargs):
            if not row['is senior']:
                return apply_func(row, *args, **kwargs)
            return row
    return _wrapper


class WeightedFinalGrade(Policy):
    """
    Applies a policy for the final grade computation based on a weighted sum of certain assignments.
    """
    MASTER_REQUIRES = [] # changes based on input provided
    MASTER_CREATES = [] # changes based on output name provided

    def __init__(self, weighting_list, output_name='final', for_seniors=None, cap_grade=True):
        self.weighting_list = weighting_list
        self.output_field = '{} grade (%)'.format(output_name)
        self.for_seniors = for_seniors
        self.cap_grade = cap_grade

        if self.for_seniors is not None:
            self.MASTER_REQUIRES.append('is senior')
        self.MASTER_CREATES.append(self.output_field)
    
    def _apply(self, gs, selected_evaluations):
        if len(selected_evaluations) != len(self.weighting_list):
            raise ValueError('The length of the list of applied evaluations ({}) should be the same as the length of the weightings list ({}).'.format(len(selected_evaluations), len(self.weighting_list)))
        
        def _enforce_policy(row):
            _grade = 0
            for i, a in enumerate(selected_evaluations):
                _grade += row[a + ' grade (%)'] * self.weighting_list[i]
            if self.cap_grade:
                _grade = min(_grade, 100)
            row[self.output_field] = _grade
            return row

        gs.master = gs.master.apply(_for_seniors_wrapper(_enforce_policy, self.for_seniors), axis=1)


class LetterGradePercentageCap(Policy):
    """
    Applies a policy for the final letter grade based on a precentage threshold.
    """
    MASTER_CREATES = [] # changes based on output name provided

    _DEFAULT_SELECTED_EVALUATIONS = ['final grade (%)',]

    def __init__(self, grade_cutoffs=[90, 80, 70, 60], letters=['A', 'B', 'C', 'D'], output_fields=['final letter',], fail_letter='F', for_seniors=None):
        # letter cutoff is a minimum grade, below which is no longer the corresponding letter. If less than last number, it's an F.
        if len(grade_cutoffs) != len(letters):
            raise ValueError('The number of grade cutoffs ({}) should be the same as the number of letters ({}).'.format(len(grade_cutoffs), len(letters)))
        sargs = np.argsort(grade_cutoffs)[::-1] # sort grades in descending order
        self.grade_cutoffs = np.array(grade_cutoffs)[sargs]
        self.letters = np.array(letters)[sargs]
        self.output_fields = output_fields
        self.fail_letter = fail_letter
        self.for_seniors = for_seniors

        if any(count > 1 for item, count in Counter(self.grade_cutoffs).items()):
            raise ValueError('You have repeated elements in the grade cutoffs.')
        
        self.MASTER_CREATES.extend(self.output_fields)
    
    def _apply(self, gs, selected_evaluations=None):
        if selected_evaluations is None:
            selected_evaluations = self._DEFAULT_SELECTED_EVALUATIONS
        if len(selected_evaluations) != len(self.output_fields):
            raise ValueError('The number of selected evaluations ({}) must match the number of output fields ({}).'.format(len(selected_evaluations), len(self.output_fields)))
        for evaluation in selected_evaluations:
            if evaluation not in gs.master.columns:
                raise Exception("'{}' is not a column in this gradesheet.".format(evaluation))
        
        def _enforce_policy(row):
            for i in range(len(selected_evaluations)):
                row[self.output_fields[i]] = self.fail_letter
                for grade_cutoff, letter in zip(self.grade_cutoffs, self.letters):
                    if row[selected_evaluations[i]] >= grade_cutoff:
                        row[self.output_fields[i]] = letter
                        break
            return row

        gs.master = gs.master.apply(_for_seniors_wrapper(_enforce_policy, self.for_seniors), axis=1)


class RequireEvaluationMinimumToPass(Policy):
    """
    Applies a policy where it is required to pass a certain grade threshold on an evaluation to pass the course.
    """
    MASTER_REQUIRES = [] # changes based on input provided 

    _DEFAULT_SELECTED_EVALUATIONS = ['final letter',]
    
    def __init__(self, evaluation_to_pass, evaluation_minimum_grade=50, fail_letter='F', for_seniors=None):  
        self.evaluation_to_pass = evaluation_to_pass
        self.evaluation_minimum_grade = evaluation_minimum_grade
        self.fail_letter = fail_letter
        self.for_seniors = for_seniors

    def _apply(self, gs, selected_evaluations=None):
        if selected_evaluations is None:
            selected_evaluations = LetterGradePercentageCap._DEFAULT_SELECTED_EVALUATIONS
        for evaluation in selected_evaluations:
            if evaluation not in gs.master.columns:
                raise Exception("'{}' is not a column in this gradesheet.".format(evaluation))
        
        def _enforce_policy(row):
            if row[self.evaluation_to_pass + ' grade (%)'] < evaluation_minimum_grade:
                row['final letter'] = self.fail_letter
            return row

        gs.master = gs.master.apply(_for_seniors_wrapper(_enforce_policy, self.for_seniors), axis=1)


def main():
    pass


def interactive_main():
    global _NON_INTERACTIVE
    _NON_INTERACTIVE = True
    
    raise NotImplementedError('Interactive main not yet created.')
    # call main


if __name__ == "__main__":
    interactive_main()


