

def encode_list(lista):
    new_lista = []
    for e in lista:
        try:
            new_lista.append(e.encode('utf-8').strip())
        except:
            new_lista.append(e)
    return new_lista


def change_char_list(lista, rep):
    new_lista = []
    for e in lista:
        for r in rep:
            new_lista.append(e.replace(r, rep[r]))
    return new_lista


def encode_dictlist(dictlist):
    new_dictlist = {}
    for e in dictlist:
        try:
            key_d = e.encode('utf-8').strip()
        except:
            key_d = e
        new_dictlist[key_d] = encode_list(dictlist[e])
    return new_dictlist
