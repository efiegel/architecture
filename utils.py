def str_to_int(characters):
    return {c: i for i, c in enumerate(characters)}


def int_to_str(characters):
    return {i: c for i, c in enumerate(characters)}


def create_encode(stoi):
    def encode(s):
        return [stoi[c] for c in s]

    return encode


def create_decode(itos):
    def decode(l):
        return "".join([itos[i] for i in l])

    return decode
