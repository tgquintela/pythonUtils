
# -*- coding: utf-8 -*-
"""
testing_coding_text
-------------------
Testing the coding text module functions and utilities.

"""

from codingtext import change_char_list, encode_list, encode_dictlist,\
    strip_accents, strip_lower, strong_formatter, mapper_unique_str


def test():
    ## Parameters
    s1 = "hola"
    s2 = " Hola "
    s3 = " Hólá "
    s4 = "Hóla "
    s5 = "hólá"
    d = {s1: [s2], s3: [s4]}
    l = [s1, s2, s3, s4]
    ch = {"ól": "ol"}

    ## Change and replace strings
    change_char_list(l, ch)

    ## Encoding list and dictlist
    encode_list(l)
    encode_dictlist(d)

    ## Strip and lowering
    assert(s1 == strip_accents(s5))
    s = strip_lower(s1)
    assert(s == strip_lower(s2))

    ## Strong formatting collapse
    s = strong_formatter(s1)
    assert(s == strong_formatter(s2))

    ## Hasing strings
    h = mapper_unique_str(s1)
    assert(h == mapper_unique_str(s2))
