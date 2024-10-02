import torch


def str_to_int_mapping(characters: str) -> dict[str, int]:
    return {c: i for i, c in enumerate(characters)}


def int_to_str_mapping(characters: str) -> dict[int, str]:
    return {i: c for i, c in enumerate(characters)}


def create_encode(stoi) -> callable:
    def encode(string: str) -> list[int]:
        return [stoi[c] for c in string]

    return encode


def create_decode(itos) -> callable:
    def decode(integers: list[int]) -> str:
        return "".join([itos[i] for i in integers])

    return decode


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
