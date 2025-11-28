from collections.abc import Iterable, Iterator
from multiprocessing import Pool
import regex as re
import yaml
import pickle as pkl
import numpy as np


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.special_tokens: list[bytes] = sorted(
            [] if special_tokens is None else [tok.encode("utf-8") for tok in special_tokens], key=len, reverse=True
        )
        self.vocab = self.add_special_tokens(vocab, self.special_tokens)
        self.merges: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(merges)}
        self.vocab_inverse: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.gpt_pattern = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_tokens_pattern = (
            re.compile(b"$^")
            if self.special_tokens is None or len(self.special_tokens) == 0
            else re.compile(b"(" + b"|".join(re.escape(t) for t in self.special_tokens) + b")")
        )

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
                        if piece not in self.vocab_inverse:
                            self.special_tokens.append(piece)
                            new_tok_id = max(self.vocab.keys()) + 1
                            self.vocab[new_tok_id] = piece
                            self.vocab_inverse[piece] = new_tok_id
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


def _process_tokenize(args):
    toker, path, start, end, split_pattern = args
    with open(path, "rb") as f:
        f.seek(start)
        chunck = f.read(end - start)
    docs = re.split(split_pattern, chunck)
    token_ids = []
    for doc in docs:
        token_ids.extend(toker.encode(doc.decode("utf-8", errors="ignore")))

    return token_ids


def main(
    filepath: str,
    vocab_merges_filepath: str,
    doc_end_key: str = "<|endoftext|>",
    special_tokens: list[str] | None = None,
    num_processes: int = 16,
):
    from cs336_basics.pretokenization_example import find_chunk_boundaries

    with open(vocab_merges_filepath, "rb") as f:
        vocab_merges = pkl.load(f)
    toker = Tokenizer(vocab=vocab_merges["vocab"], merges=vocab_merges["merges"], special_tokens=special_tokens)

    # tokenize all file and store tokens
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, doc_end_key.encode("utf-8"))
    split_pattern = re.compile(b"|".join(re.escape(t.encode("utf-8")) for t in special_tokens))
    tasks = [(toker, filepath, start, end, split_pattern) for (start, end) in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        results = pool.map(_process_tokenize, tasks)
    token_ids = []
    for res in results:
        token_ids.extend(res)
    return token_ids


if __name__ == "__main__":
    toker = Tokenizer.from_files(
        vocab_filepath="./tests/fixtures/gpt2_vocab.json",
        merges_filepath="./tests/fixtures/gpt2_merges.txt",
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

    #####################
    token_ids = main(
        filepath="/home/yongce/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        vocab_merges_filepath="/home/yongce/Desktop/cs336/assignment1-basics/data/BPE-TinyStoriesV2-GPT4-train.pkl",
        doc_end_key="<|endoftext|>",
        special_tokens=["<|endoftext|>"],
        num_processes=19,
    )

    arr = np.array(token_ids, dtype=np.uint16)
    print(f"Total tokens: {len(arr)}")
    np.save("/home/yongce/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train-tokens.npy", arr)
