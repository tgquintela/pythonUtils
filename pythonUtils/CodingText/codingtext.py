
"""
Codingtext
----------
coding text funtions.

"""

import unicodedata


################################ Functions to  ################################
###############################################################################
def change_char_list(lista, rep):
    """The strings of a list by replacling some parts.

    Parameters
    ----------
    lista: list
        the list of strings that we want to transform.
    rep: dict
        the replace dictionary.

    Returns
    -------
    new_lista: list
        the list of transformed strings.

    """
    new_lista = []
    for e in lista:
        for r in rep:
            new_lista.append(e.replace(r, rep[r]))
    return new_lista


def encode_list(lista):
    """Encode a list of strings.

    Parameters
    ----------
    lista: list
        the list of strings that we want to transform.
    rep: dict
        the replace dictionary.

    Returns
    -------
    new_lista: list
        the list of transformed strings.

    """
    new_lista = []
    for e in lista:
        try:
            new_lista.append(e.encode('utf-8').strip())
        except:
            new_lista.append(e)
    return new_lista


def encode_dictlist(dictlist):
    """Encode the keys and the values in form of list of strings

    Parameters
    ----------
    dictlist: dict
        the dictionary of list of strings.

    Returns
    -------
    new_dictlist: dict
        the transformed dictionary of list of strings.

    """
    new_dictlist = {}
    for e in dictlist:
        try:
            key_d = e.encode('utf-8').strip()
        except:
            key_d = e
        new_dictlist[key_d] = encode_list(dictlist[e])
    return new_dictlist


def strip_accents(s):
    """It removes accents of the strings.

    Parameters
    ----------
    s: str
        the string to be formatted or transformed.

    Returns
    -------
    s_t: str
        the transformed string.

    """
    try:
        s = unicode(s)
    except:
        s = s.decode("utf-8")
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def strip_lower(s):
    """It removes the useless spaces and transform all the letters to
    lower-case letters.

    Parameters
    ----------
    s: str
        the string to be formatted or transformed.

    Returns
    -------
    s_t: str
        the transformed string.

    """
    return s.strip().lower()


def strong_formatter(s):
    """Strong formatter. It uses others formatters and make a collapse of
    strings only to their meanings result, ignoring other not meaningful
    properties.

    Parameters
    ----------
    s: str
        the string to be formatted or transformed.

    Returns
    -------
    s_t: str
        the transformed string.

    """
    s_split = s.strip().split(" ")
    return " ".join([strip_accents(strip_lower(e)) for e in s_split])


def mapper_unique_str(s):
    """Mapping function of a string to a number using hash.

    Parameters
    ----------
    s: str
        the string to be formatted or transformed.

    Returns
    -------
    h: int
        hash number in which we collapsed all the string.

    """
    return hash(strong_formatter(s))
