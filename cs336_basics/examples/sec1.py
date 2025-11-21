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


print(decode_utf8_bytes_to_str_wrong(b"hello"))
try:
    print(decode_utf8_bytes_to_str_wrong("牛".encode("utf-8")))
    print(b"\xc0\x80".decode("utf-8"))
except Exception as e:
    print(e)

# %%
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re

re.findall(PAT, "some text that i'll pre-tokenize")
# %%
max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])

# %%
