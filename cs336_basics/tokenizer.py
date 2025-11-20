from collections.abc import Iterable, Iterator
import regex as re
import yaml


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.special_tokens: list[bytes] = sorted(
            [] if special_tokens is None else [tok.encode("utf-8") for tok in special_tokens], key=len, reverse=True
        )
        self.vocab = self.add_special_tokens(vocab, self.special_tokens)
        self.merges: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        vocab_filepath: yaml
        merges_filepath: txt
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_str = yaml.safe_load(f)
        vocab = {v: k.encode("utf-8") for k, v in vocab_str.items()}

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                a, b = line.strip().split(" ")
                merges.append((a.encode("utf-8"), b.encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        text_bytes = text.encode("utf-8")
        rough_tokens = self.special_tokens_pattern.split(text_bytes)
        tok_ids = []
        for part in rough_tokens:
            if part in self.special_tokens:
                tok_ids.append(self.vocab_inverse[part])
            else:
                for word in self.gpt_pattern.finditer(part):
                    merge_word = [bytes([w]) for w in word.group(0)]
                    if len(merge_word) > 1:
                        merge_word = self.apply_merge(merge_word)
                    for piece in merge_word:
                        if piece not in self.vocab.values():
                            self.special_tokens.append(piece)
                            self.vocab[max(self.vocab.keys()) + 1] = piece
                        tok_ids.append(self.vocab_inverse[piece])
        return tok_ids

    def apply_merge(self, word_pieces: list[bytes] | tuple[bytes]) -> list[bytes]:
        while True:
            min_rank = float("inf")
            best_pair = None
            for i in range(len(word_pieces) - 1):
                pair = (word_pieces[i], word_pieces[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_pieces = []
            i = 0
            while i < len(word_pieces):
                if i < len(word_pieces) - 1 and (word_pieces[i], word_pieces[i + 1]) == best_pair:
                    new_pieces.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_pieces.append(word_pieces[i])
                    i += 1
            word_pieces = new_pieces

        return word_pieces

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        decode_str_bytes = b""
        for tok_id in ids:
            decode_str_bytes += self.vocab[tok_id]
        return decode_str_bytes.decode("utf-8", errors="replace")

    @staticmethod
    def add_special_tokens(vocab: dict[int, bytes], special_tokens: list[bytes]):
        assert isinstance(special_tokens, list)
        for tok in special_tokens:
            assert isinstance(tok, bytes)
            if tok not in vocab.values():
                tok_id = max(vocab.keys()) + 1
                vocab[tok_id] = tok
        return vocab

    @property
    def vocab_inverse(self) -> dict[bytes, int]:
        return {v: k for k, v in self.vocab.items()}

    @property
    def special_tokens_pattern(self):
        return (
            re.compile(b"$^")
            if self.special_tokens is None or len(self.special_tokens) == 0
            else re.compile(b"(" + b"|".join(re.escape(t) for t in self.special_tokens) + b")")
        )

    @property
    def gpt_pattern(self):
        return re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


if __name__ == "__main__":
    toker = Tokenizer.from_files(
        vocab_filepath="/home/yongce/aws/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json",
        merges_filepath="/home/yongce/aws/cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt",
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    prompt = "I love you."
    prompt = "Hello, how are you?"
    prompt = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    prompt = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    tok_ids = toker.encode(prompt)
    print(f"tok_ids: {tok_ids}")
    decode_str = toker.decode(tok_ids)
    print(f"original str:\t {prompt}\necode_str:\t {decode_str}")
    tokenized_string = [toker.decode([x]) for x in tok_ids]
    print(f"tokenized_string: {tokenized_string}")
    assert decode_str == prompt
