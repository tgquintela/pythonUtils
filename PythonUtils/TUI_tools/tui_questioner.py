
"""
Module implemented to group functions usable to perform a correct terminal
user interface in a correct, readable and re-usable way.
"""

import numpy as np


def general_questioner(qtype, question_spec):
    """Switcher function to select which question type use.

    Parameters
    ----------
    qtype: str, optional
        type of question you want to do.
    question_spec: dict
        dictionary of variables and values needed for each selected function.

    Returns
    -------
    selections: optional
        returns the selected values.

    """

    if qtype == 'simple_input':
        selection = simple_input(**question_spec)
    elif qtype == 'confirmation_question':
        selection = confirmation_question(**question_spec)
    elif qtype == 'selection_options':
        selection = selection_options(**question_spec)
    elif qtype == 'selection_list_options':
        selection = selection_list_options(**question_spec)

    return selection


def simple_input(question, transf):
    """Function prepared for doing a question a simple question.

    Parameters
    ----------
    question: str
        string with a question you want to ask in order to get the selection.
    transf: function
        function you want to apply for the string input in the terminal as a
        answer of the question.

    Returns
    -------
    output: optional
        returns the input value after apply the transformation specified.

    """
    output = raw_input(question+'\n')
    output = transf(output)
    return output


def confirmation_question(question=None, options=['Y', 'N']):
    """Function prepared for doing a confirmation question. This kind of
    question outputs a boolean function depending of the selection of the user.

    Parameters
    ----------
    question: str
        string with a question you want to ask in order to get the selection.
    options: list
        list of possible options in which we want to select. The first one will
        be related with the True.

    Returns
    -------
    output: boolean
        returns a boolean in which the selection of the first one returns True.

    """

    # Creation of the question and asking
    if question is None:
        question = "Is it correct your option? (Y/N)\n"
        options = ['Y', 'N']
    else:
        question = question + " (" + options[0] + "/" + options[1] + ")" + "\n"
    answer = raw_input(question)

    # Trying to normalize the input
    try:
        answer = answer.lower()
    except:
        return False

    # Return
    if answer == options[0].lower():
        return True
    else:
        return False


def selection_options(question, name, options, representation='list',
                      transf=lambda x: x):
    """Function prepared for asking a question of selection options. The user
    has the possibility to select one of the possible options in the list.

    Parameters
    ----------
    question: str
        string with a question you want to ask in order to get the selection.
    name: str
        name of the variable for which we want to ask.
    options: list
        list of possible options in which we want to select.
    representation: str, optional
        the way we want to show the options: list, numerical
    transf: function
        function you want to apply for the string input in the terminal as a
        answer of the question.

    Returns
    -------
    output: optional
        returns the input value after apply the transformation specified.

    """

    # Prepare question
    if representation == 'list':
        options_str = str(options)
    elif representation == 'numerical':
        options_str = "\n"
        for i in range(len(options)):
            options_str = options_str+"("+str(i)+")"+options[i]+"\n"
    question_f = question+"\n"+"Select "+name+" from: "+options_str+"\n"

    # Asking up to correctness
    correctness = False
    while not correctness:
        selection = simple_input(question_f, transf)
        correctness = selection in options
        if correctness is False:
            print 'Incorrect option. Try another time.'
    return selection


def selection_list_options(question, name, options, representation='list',
                           transf=lambda x: x):
    """Function prepared for asking a question of selection options. The user
    has the possibility to select more than one of the possible options in the
    list.

    Parameters
    ----------
    question: str
        string with a question you want to ask in order to get the selection.
    name: str
        name of the variable for which we want to ask.
    options: list
        list of possible options in which we want to select.
    representation: str, optional
        the way we want to show the options: list, numerical
    transf: function
        function you want to apply for the string input in the terminal as a
        answer of the question.

    Returns
    -------
    output: optional
        returns the input value after apply the transformation specified.

    """

    # Prepare question
    if representation == 'list':
        options_str = str(options)
    elif representation == 'numerical':
        options_str = "\n"
        for i in range(len(options)):
            options_str = options_str+"("+str(i)+")"+options[i]+"\n"
    question_f = question+"\n"+"Select "+name+" from: "+options_str+"\n"

    # Asking up to correctness
    correctness = False
    while not correctness:
        selections = simple_input(question_f, transf)
        selections = selections.strip().split(',')
        correctness = np.all([s in options for s in selections])
        if correctness is False:
            print 'Incorrect option. Try another time.'
    return selections
