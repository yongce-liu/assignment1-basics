from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from collections import Counter
from itertools import chain
import regex as re


def example():
    # %%
    ord("牛"), chr(29275)
    # %%
    chr(0)
    # %%
    c = "\n"
    print("Printed:", c)
    print("Repr:", repr(c))
    print("__repr__():", c.__repr__())
    # %%
    chr(0)
    print(chr(0))
    print(chr(0).__repr__())
    "this is a test" + chr(0) + "string"
    print("this is a test" + chr(0) + "string")
    # %%
    test_string = "hello! こんにちは!"
    utf8_encoded = test_string.encode("utf-8")
    print(utf8_encoded)
    print(type(utf8_encoded))
    # Get the byte values for the encoded string (integers from 0 to 255).
    list(utf8_encoded)
    # One byte does not necessarily correspond to one Unicode character!
    print(len(test_string))
    13
    print(len(utf8_encoded))
    23
    print(utf8_encoded.decode("utf-8"))
    # %%
    test_string = "hello! こんにちは!"
    utf8_encoded = test_string.encode("utf-8")
    utf32_encoded = test_string.encode("utf-32")
    import sys

    print(sys.getsizeof(utf8_encoded))
    print(sys.getsizeof(utf32_encoded))

    # %%
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])

    print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
    print(decode_utf8_bytes_to_str_wrong("牛".encode("utf-8")))
    # %%
    print(b"\xc0\x80".decode("utf-8"))
    # %%
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    import regex as re

    re.findall(PAT, "some text that i'll pre-tokenize")
    # %%
    max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])


# %%
PAT_GPT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _process_get_tokens_from_chunk(args):
    path, start, end, pattern = args

    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    text = chunk.decode("utf-8", errors="ignore")
    parts = re.split(pattern, text)

    byte_tokens = []

    for piece in parts:
        for m in PAT_GPT.finditer(piece):
            for b in m.group(0).encode("utf-8"):
                byte_tokens.append(bytes(b))

    return byte_tokens


def get_byte_tokens(
    path: str,
    num_processes: int,
    spilit_key: bytes,
    special_tokens: list[str],
) -> list[bytes]:

    boundaries = find_chunk_boundaries(open(path, "rb"), num_processes, spilit_key)
    pattern = "|".join(re.escape(t) for t in special_tokens)
    tasks = [
        (path, start, end, pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(processes=num_processes) as pool:
        chunks = pool.map(_process_get_tokens_from_chunk, tasks)

    byte_tokens = list(chain.from_iterable(chunks))
    return byte_tokens


def get_byte_pairs_with_merge(
    byte_tokens: list[bytes], merge: tuple[bytes, bytes] | None
):
    freq = Counter()
    new_byte_tokens: list[bytes] = []

    if merge is not None:
        m0, m1 = merge
        merge_id = m0 + m1

    i = 0
    n = len(byte_tokens)
    prev = None

    while i < n:
        if (
            merge is not None
            and i + 1 < n
            and byte_tokens[i] == m0
            and byte_tokens[i + 1] == m1
        ):
            cur = merge_id
            i += 2
        else:
            cur = byte_tokens[i]
            i += 1

        if prev is not None:
            freq[(prev, cur)] += 1

        new_byte_tokens.append(cur)
        prev = cur

    return new_byte_tokens, freq.most_common(1)[0][0]


if __name__ == "__main__":
    from pathlib import Path

    res = get_byte_tokens(
        Path(__file__).parent / "../data/TinyStoriesV2-GPT4-valid.txt",
        24,
        b"<|endoftext|>",
        ["<|endoftext|>"],
    )
