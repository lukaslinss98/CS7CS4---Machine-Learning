from typing import List


def create_encoder_decoder(vocabulary: List[str]):
    token_to_integer = {token: i for i, token in enumerate(vocabulary)}
    integer_to_token = {i: ch for i, ch in enumerate(vocabulary)}

    encode = lambda tokens: [token_to_integer[token] for token in tokens]
    decode = lambda integers: ''.join([integer_to_token[i] for i in integers])

    return encode, decode
