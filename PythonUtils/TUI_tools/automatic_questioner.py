
'''
Module which serves as a interactor between the possible database with the 
described structure and which contains information about functions and
variables of other packages.

    Scheme of the db
    ----------------
# {'function_name':
#   {'variables':
#       {'variable_name':
#           {'question_info':
#               {'qtype': ['simple_input', 'confirmation_question',
#                          'selection_options', 'selection_list_options'],
#                'question_spec': 'question_spec'},
#            'default': default}},
########
#    'descendants': [{'agg_description':
#                            {variable_name:
#                                {'variable_value': 'function_name'}
#                             },
#                     'agg_name': 'aggregated_parameter_name'}]
# }}
######## OR
#    'descendants': [{'agg_description': 'function_name'
#                     'agg_name': 'aggregated_parameter_name'}]
# }}

#TODO: checker 1 function with list of functions and dicts of dicts
'''

from tui_questioner import general_questioner


def check_quest_info(db):
    """Function which carry out the automatic checking of the database of
    function and variables.

    Parameters
    ----------
    db: dict
        the dictionary of all the information about the system with all its
        functions and dependencies between them in order to ask for their
        variables authomatically.

    Returns
    -------
    check: boolean
        returns the correctness of the database.
    path: list
        path of the possible error.
    message: str
        message of the error if it exists.

    """
    ## 0. Initial preset variables needed
    # Function to compare lists
    def equality_elements_list(a, b):
        a = a.keys() if type(a) == dict else a
        b = b.keys() if type(b) == dict else b
        c = a[-1::-1]
        return a == b or c == b
    # List of elements available in some dicts at some levels
    first_level = ['descendants', 'variables']
    desc_2_level = ['agg_description', 'agg_name']
    vars_2_level = ['question_info', 'default']
    vars_3_level = ['qtype', 'question_spec']
    # Messages of errors
    m0 = "The given database of functions is not a dictionary."
    m1 = "The function '%s' does not have "+str(first_level)+" as keys."
    m2 = "The variables of function '%s' is not a dict."
    m3 = "Incorrect keys "+str(vars_2_level)+" in function %s and variable %s."
    m4 = "Incorrect question_info format for function %s and variable %s."
    m5 = "Not a string the 'qtype' of function %s and variable %s."
    m6 = "Incorrect 'question_spec' format for function %s and variable %s."
    m7 = "Descendants of the function %s is not a list."
    m8 = "Elements of the list of descendants not a dict for function %s."
    m9 = "Incorrect structure of a dict in descendants for function %s."
    m10 = "Incorrect type of agg_description for function %s and variable %s."
    m11 = "Incorrect type of agg_description for function %s."

    ## Check db is a dict
    if type(db) != dict:
        return False, [], m0

    ## Loop for check each function in db
    for funct in db.keys():
        ## Check main keys:
        first_bl = equality_elements_list(db[funct], first_level)
        if not first_bl:
            return False, [funct], m1 % funct

        ## Check variables
        if not type(db[funct]['variables']) == dict:
            check = False
            path = [funct, 'variables']
            message = m2 % funct
            return check, path, message
        for var in db[funct]['variables']:
            varsbles = db[funct]['variables']
            v2_bl = equality_elements_list(varsbles[var], vars_2_level)
            v3_bl = equality_elements_list(varsbles[var]['question_info'],
                                           vars_3_level)
            qtype_bl = db[funct]['variables'][var]['question_info']['qtype']
            qtype_bl = type(qtype_bl) != str
            qspec_bl = db[funct]['variables'][var]['question_info']
            qspec_bl = type(qspec_bl['question_spec']) != dict

            if not v2_bl:
                check = False
                path = [funct, 'variables', var]
                message = m3 % (funct, var)
                return check, path, message
            ### Check question_info

            if not v3_bl:
                check = False
                path = [funct, 'variables', 'question_info']
                message = m4 % (funct, var)
                return check, path, message

            if qtype_bl:
                check = False
                path = [funct, 'variables', 'question_info', 'qtype']
                message = m5 % (funct, var)
                return check, path, message

            if qspec_bl:
                check = False
                path = [funct, 'variables', 'question_info', 'question_spec']
                message = m6 % (funct, var)
                return check, path, message

        ## Check descendants
        if not type(db[funct]['descendants']) == list:
            check = False
            path = [funct, 'descendants']
            message = m7 % funct
            return check, path, message

        for var_desc in db[funct]['descendants']:
            if not type(var_desc) == dict:
                check = False
                path = [funct, 'descendants']
                message = m8 % funct
                return check, path, message
            d2_bl = equality_elements_list(var_desc.keys(), desc_2_level)
            if not d2_bl:
                check = False
                path = [funct, 'descendants']
                message = m9 % funct
                return check, path, message
            if type(var_desc['agg_description']) == str:
                pass
            elif type(var_desc['agg_description']) == dict:
                for varname in var_desc['agg_description']:
                    if not type(var_desc['agg_description'][varname]) == dict:
                        check = False
                        path = [funct, 'descendants', 'agg_description']
                        message = m10 % (funct, varname)
                        return check, path, message
            else:
                check = False
                path = [funct, 'descendants', 'agg_description']
                message = m11 % funct
                return check, path, message

    return True, [], ''


def automatic_questioner(function_name, db, choosen={}):
    """Function which carry out the automatic questioning task.

    Parameters
    ----------
    function_name: str
        the function for which we are interested in their params in order to
        call it.
    db: dict
        the dictionary of all the information about the system with all its
        functions and dependencies between them in order to ask for their
        variables authomatically.
    choosen: dict
        previous choosen parameters. The function will avoid to ask for the
        pre-set parameters.

    Returns
    -------
    choosen_values: dict
        the selected values which are disposed to input in the function we want
        to call.

    """

    ## Initialize variables needed
    m1 = "Not value for a variables in order to create aggregate variables."
    choosen_values = choosen
    if function_name in db.keys():
        data_f = db[function_name]
    else:
        # Better raise error?
        return choosen_values

    # Put the variables
    for var in data_f['variables'].keys():
        # Put the variables if there are still not selected
        if var not in choosen_values.keys():
            question = data_f['variables'][var]['question_info']
            choosen_values[var] = general_questioner(**question)

    # Put aggregated variables (descendants)
    for var_desc in data_f['descendants']:
        # Possible variables and aggregated parameter name
        agg_description = var_desc['agg_description']
        agg_param = var_desc['agg_name']
        # prepare possible input for existant aggregated value in choosen
        ifaggvar = agg_param in choosen_values
        aggvarval = choosen_values[agg_param] if ifaggvar else {}

        ## Without dependant variable
        if type(agg_description) == str:
            # Obtain function name
            fn = choosen_values[agg_param]
            # Recurrent call
            aux = automatic_questioner(fn, db, aggvarval)
            # Aggregate to our values
            choosen_values[agg_param] = aux
        ## With dependant variable
        elif type(agg_description) == dict:
            for var in var_desc['agg_description']:
                if not var in choosen_values:
                    raise Exception(m1)
                ## Give a list and return a dict in the aggparam variable
                elif type(choosen_values[var]) == str:
                    # Obtain function name
                    fn = var_desc['agg_description'][var][choosen_values[var]]
                    # Recurrent call
                    aux = automatic_questioner(fn, db, aggvarval)
                    # Aggregate to our values
                    choosen_values[agg_param] = aux
                ## Give a list and return a list in the aggparam variable
                elif type(choosen_values[var]) == list:
                    choosen_values[agg_param] = []
                    aggvarval = [] if type(aggvarval) != list else aggvarval
                    for i in range(len(choosen_values[var])):
                        val = choosen_values[var][i]
                        fn = var_desc['agg_description'][var][val]
                        aux = automatic_questioner(fn, db, aggvarval[i])
                        choosen_values.append(aux)

    return choosen_values


def get_default(function_name, db, choosen={}):
    """Function which returns a dictionary of choosen values by default.

    Parameters
    ----------
    function_name: str
        the function for which we are interested in their params in order to
        call it.
    db: dict
        the dictionary of all the information about the system with all its
        functions and dependencies between them in order to ask for their
        variables authomatically.
    choosen: dict
        previous choosen parameters. The function will avoid to ask for the
        pre-set parameters.

    Returns
    -------
    choosen_values: dict
        the selected values which are disposed to input in the function we want
        to call.

    -----
    TODO: Possibility of being integrated with authomatic_questioner after
    testing.
    """

    ## Initialize variables needed
    m1 = "Not value for a variables in order to create aggregate variables."
    choosen_values = choosen
    if function_name in db.keys():
        data_f = db[function_name]
    else:
        # Better raise error?
        return choosen_values

    # Get the variables
    for var in data_f['variables'].keys():
        # Put the variables if there are still not selected
        if var not in choosen_values.keys():
            default = data_f['variables'][var]['default']
            choosen_values[var] = default

    # Get aggregated variables (descendants)
    for var_desc in data_f['descendants']:
        # Possible variables and aggregated parameter name
        agg_description = var_desc['agg_description']
        agg_param = var_desc['agg_name']
        # prepare possible input for existant aggregated value in choosen
        ifaggvar = agg_param in choosen_values
        aggvarval = choosen_values[agg_param] if ifaggvar else {}

        ## Without dependant variable
        if type(agg_description) == str:
            # Obtain function name
            fn = choosen_values[agg_param]
            # Recurrent call
            aux = get_default(fn, db, aggvarval)
            # Aggregate to our values
            choosen_values[agg_param] = aux
        ## With dependant variable
        elif type(agg_description) == dict:
            for var in var_desc['agg_description']:
                if not var in choosen_values:
                    raise Exception(m1)
                ## Give a list and return a dict in the aggparam variable
                elif type(choosen_values[var]) == str:
                    # Obtain function name
                    fn = var_desc['agg_description'][var][choosen_values[var]]
                    # Recurrent call
                    aux = get_default(fn, db, aggvarval)
                    # Aggregate to our values
                    choosen_values[agg_param] = aux
                ## Give a list and return a list in the aggparam variable
                elif type(choosen_values[var]) == list:
                    choosen_values[agg_param] = []
                    aggvarval = [] if type(aggvarval) != list else aggvarval
                    for i in range(len(choosen_values[var])):
                        val = choosen_values[var][i]
                        fn = var_desc['agg_description'][var][val]
                        aux = get_default(fn, db, aggvarval[i])
                        choosen_values.append(aux)

    return choosen_values


###############################################################################
###############################################################################
###############################################################################
def get_default3(function_name, db, choosen={}):
    """Function which returns a dictionary of choosen values by default.

    Parameters
    ----------
    function_name: str
        the function for which we are interested in their params in order to
        call it.
    db: dict
        the dictionary of all the information about the system with all its
        functions and dependencies between them in order to ask for their
        variables authomatically.
    choosen: dict
        previous choosen parameters. The function will avoid to ask for the
        pre-set parameters.

    Returns
    -------
    choosen_values: dict
        the selected values which are disposed to input in the function we want
        to call.

    -----
    TODO: Possibility of being integrated with authomatic_questioner after
    testing.
    """

    choosen_values = choosen
    if function_name in db.keys():
        data_f = db[function_name]
    else:
        # Better raise error?
        return choosen_values

    # Get the variables
    for var in data_f['variables'].keys():
        # Put the variables if there are still not selected
        if var not in choosen_values.keys():
            default = data_f['variables'][var]['default']
            choosen_values[var] = default

    # Get the aggregated variables (descendants)
    for i in range(len(data_f['descendants'])):
        # Possible variables and aggregated parameter name
        vars_values = data_f['descendants'][i]['variable_values']
        agg_param = data_f['descendants'][i]['parameters']
        variables = vars_values.keys()
        # prepare possible input for existant aggregated value in choosen
        ifaggvar = agg_param in choosen_values
        aggvarval = choosen_values[agg_param] if ifaggvar else {}

        for var in variables:
            # boolean variables
            value = choosen_values[var]
            iflist = type(value) == list
            ifvars = var in choosen_values.keys()

            # if we have to return a list
            if ifvars and iflist:
                # Initialization values
                n = len(value)
                aggvarval = aggvarval if ifaggvar else [{} for i in range(n)]
                results = []
                i = 0
                for val in value:
                    # Obtain function_name
                    f_name = vars_values[var][value]
                    # Recurrent call
                    aux = get_default(f_name, db, aggvarval[i])
                    # Insert in the correspondent list
                    results.append(aux)
                    i += 1

            # if we have to return a dict
            elif ifvars and not iflist:
                # Obtain function_name
                f_name = vars_values[var][value]
                # Recurrent call
                choosen_values[agg_param] = get_default(f_name, db, aggvarval)

    return choosen_values


def automatic_questioner3(function_name, db, choosen={}):
    """Function which carry out the automatic questioning task.

    Parameters
    ----------
    function_name: str
        the function for which we are interested in their params in order to
        call it.
    db: dict
        the dictionary of all the information about the system with all its
        functions and dependencies between them in order to ask for their
        variables authomatically.
    choosen: dict
        previous choosen parameters. The function will avoid to ask for the
        pre-set parameters.

    Returns
    -------
    choosen_values: dict
        the selected values which are disposed to input in the function we want
        to call.

    """

    ## Initialize variables needed
    m1 = "Not value for a variables in order to create aggregate variables."
    choosen_values = choosen
    if function_name in db.keys():
        data_f = db[function_name]
    else:
        # Better raise error?
        return choosen_values

    # Put the variables
    for var in data_f['variables'].keys():
        # Put the variables if there are still not selected
        if var not in choosen_values.keys():
            question = data_f['variables'][var]['question_info']
            choosen_values[var] = general_questioner(**question)

    # Put aggregated variables (descendants)
    for i in range(len(data_f['descendants'])):
        # Possible variables and aggregated parameter name
        vars_values = data_f['descendants'][i]['variable_values']
        agg_param = data_f['descendants'][i]['parameters']
        variables = vars_values.keys()
        # prepare possible input for existant aggregated value in choosen
        ifaggvar = agg_param in choosen_values
        aggvarval = choosen_values[agg_param] if ifaggvar else {}

        for var in variables:
            # boolean variables
            value = choosen_values[var]
            iflist = type(value) == list
            ifvars = var in choosen_values.keys()

            # if we have to return a list
            if ifvars and iflist:
                # Initialization values
                n = len(value)
                aggvarval = aggvarval if ifaggvar else [{} for i in range(n)]
                results = []
                i = 0
                for val in value:
                    # Obtain function_name
                    f_name = vars_values[var][value]
                    # Recurrent call
                    aux = authomatic_questioner(f_name, db, aggvarval[i])
                    # Insert in the correspondent list
                    results.append(aux)
                    i += 1

            # if we have to return a dict
            elif ifvars and not iflist:
                # Obtain function_name
                f_name = vars_values[var][value]
                # Recurrent call
                choosen_values[agg_param] = authomatic_questioner(f_name, db,
                                                                  aggvarval)

    return choosen_values

''' TO DEPRECATE

def authomatic_questioner(function_name, db):
    """Function which carry out the authomatic questioning task.
    """

    choosen_values = {}
    if function_name in db.keys():
        data_f = db[function_name]
    else:
        return choosen_values

    # Put the variables
    for var in data_f['variables'].keys():
        question = data_f['variables'][var]['question_info']
        choosen_values[var] = general_questioner(**question)

    # Put aggregated variables (descendants)
    for i in range(len(data_f['descendants'])):
        vars_values = data_f['descendants'][i]['variable_values']
        agg_param = data_f['descendants'][i]['parameters']
        variables = vars_values.keys()
        for var in variables:
            if var in choosen_values.keys():
                # Obtain function_name
                f_name = vars_values[var][choosen_values[var]]
                # Recurrent call
                choosen_values[agg_param] = authomatic_questioner(f_name, db)

    return choosen_values


def authomatic_questioner(function_name, db):
    """Function which carry out the authomatic questioning task.
    """

    choosen_values = {}
    if function_name in db.keys():
        data_f = db[function_name]
    else:
        return choosen_values

    # Put the variables
    for var in data_f['variables'].keys():
        question = data_f['variables'][var]['question_info']
        choosen_values[var] = general_questioner(**question)

    # Put aggregated variables (descendants)
    for i in range(len(data_f['descendants'])):
        vars_values = data_f['descendants'][i]['variable_values']
        agg_param = data_f['descendants'][i]['parameters']
        variables = vars_values.keys()
        for var in variables:
            if var in choosen_values.keys():
                # Obtain function_name
                f_name = vars_values[var][choosen_values[var]]
                # Recurrent call
                choosen_values[agg_param] = authomatic_questioner(f_name, db)

    return choosen_values


def get_default(function_name, db):
    """Function which returns a dictionary of choosen values by default.

    -----
    TODO: Possibility of being integrated with authomatic_questioner after
    testing.
    """

    data_f = db[function_name]
    choosen_values = {}

    # Put the variables
    for var in data_f['variables'].keys():
        default = data_f['variables'][var]['default']
        choosen_values[var] = default

    # Put aggregated variables (descendants)
    for i in range(len(data_f['descendants'])):
        vars_values = data_f['descendants'][i]['variable_values']
        agg_param = data_f['descendants'][i]['parameters']
        variables = vars_values.keys()
        for var in variables:
            if var in choosen_values.keys():
                # Obtain function_name
                f_name = vars_values[var][choosen_values[var]]
                # Recurrent call
                choosen_values[agg_param] = authomatic_questioner(f_name, db)

    return choosen_values

'''
