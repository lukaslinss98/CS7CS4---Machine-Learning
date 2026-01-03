import unittest
from typing import List


class NumberTokenizer:

    def __init__(self):
        self.units = ['U', 'T', 'H']

    def tokenize(self, data: str, padding=False) -> List[str]:

        def flush_number(buffer: List[str]):
            if not buffer:
                return []

            if padding:
                buffer = ''.join(buffer).zfill(3)

            tmp = []
            for index, num in enumerate(reversed(buffer)):
                tmp.append(f'{self.units[index]}{num}')

            return list(reversed(tmp))

        i = 0
        tokens: List[str] = []
        length_data = len(data)
        current_digits: List[str] = []

        while i < length_data:
            character = data[i]

            if character.isdigit():
                current_digits.append(character)
            else:
                new_tokens = flush_number(current_digits)
                tokens += new_tokens
                current_digits = []

                if character == "\n":
                    tokens.append("\n")
                elif character.isspace():
                    pass
                else:
                    tokens.append(character)

            i += 1

        new_tokens = flush_number(current_digits)
        tokens += new_tokens
        return tokens

    def token_to_string(self, data: str) -> str:
        result = []
        i = 0
        data_length = len(data)

        def read_digit_at(pos: int) -> str:
            if pos + 1 < data_length and data[pos + 1].isdigit():
                return data[pos + 1]
            return '0'

        while i < data_length:
            ch = data[i]

            if ch in self.units:
                h = t = u = '0'

                if i < data_length and data[i] == 'H':
                    h = read_digit_at(i)
                    i += 2
                if i < data_length and data[i] == 'T':
                    t = read_digit_at(i)
                    i += 2
                if i < data_length and data[i] == 'U':
                    u = read_digit_at(i)
                    i += 2

                result.append(''.join([h, t, u]).lstrip('0') or '0')
                continue

            else:
                result.append(ch)
                i += 1

        return ''.join(result)


class TestTokenizer(unittest.TestCase):

    def test_single_digit(self):
        given = '1+2=3'
        tokenized = NumberTokenizer().tokenize(given)
        self.assertEqual(['U1', '+', 'U2', '=', 'U3'], tokenized)

    def test_double_digit(self):
        given = '20+35=45'
        actual = NumberTokenizer().tokenize(given)
        self.assertEqual(['T2', 'U0', '+', 'T3', 'U5', '=', 'T4', 'U5'], actual)

    def test_triple_digit(self):
        given = '100+302=402'
        actual = NumberTokenizer().tokenize(given)
        self.assertEqual(['H1', 'T0', 'U0', '+', 'H3', 'T0', 'U2', '=', 'H4', 'T0', 'U2'], actual)

    def test_padding(self):
        given = '20+35=45'
        actual = NumberTokenizer().tokenize(given, padding=True)
        self.assertEqual(['H0', 'T2', 'U0', '+', 'H0', 'T3', 'U5', '=', 'H0', 'T4', 'U5'], actual)

    def test_token_to_string(self):
        given = 'T2U1+U2=T2U3'
        actual = NumberTokenizer().token_to_string(given)
        self.assertEqual('21+2=23', actual)

    def test_padded_token_to_string(self):
        given = 'H0T1U0+H0T2U2=H0T3U2\n'
        actual = NumberTokenizer().token_to_string(given)
        self.assertEqual('10+22=32\n', actual)

    def test_padded_token_with_only_zeros_to_string(self):
        given = 'H0T0U0+H0T0U0=H0T0U0\n'
        actual = NumberTokenizer().token_to_string(given)
        self.assertEqual('0+0=0\n', actual)
