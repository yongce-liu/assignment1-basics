from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from collections import Counter
import regex as re


PAT_GPT = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _process_get_words_count(args) -> dict[tuple[bytes], int]:
    path, start, end, split_pattern = args
    with open(path, "rb") as f:
        f.seek(start)
        chunck = f.read(end - start)  # .decode("utf-8", errors="ignore")
    docs = re.split(split_pattern, chunck)
    freq = Counter()
    for piece in docs:
        for m in PAT_GPT.finditer(piece):
            key = tuple(bytes([b]) for b in m.group(0))
            # key = tuple(m.group(0))
            freq[key] += 1
    return freq


def pre_tokenization(
    path: str,
    num_processes: int,
    split_key: bytes,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_key)
    split_pattern = re.compile(b"|".join(re.escape(t.encode("utf-8")) for t in special_tokens))

    tasks = [(path, start, end, split_pattern) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        results = pool.map(_process_get_words_count, tasks)
    freq = Counter()
    for res in results:
        # Counter.update adds counts instead of overwriting existing keys.
        freq.update(res)
    return freq


def get_pairs_count(
    words: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    pairs = Counter()
    for word, freq in words.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(words: dict[tuple[bytes, ...], int], merge: tuple[bytes, bytes]):
    a, b = merge
    merged_words = {}
    for word, freq in words.items():
        tokens = word
        new_tokens = []
        changed = False
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                changed = True
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        if changed:
            new_word = tuple(new_tokens)
        else:
            new_word = word
        merged_words[new_word] = freq

    return merged_words


def test():
    from pathlib import Path

    words_count = pre_tokenization(
        # Path(__file__).parent / "../data/TinyStoriesV2-GPT4-train.txt",
        # "/home/unitree/Desktop/DRL/cs336/assignment1-basics/tests/fixtures/corpus.en",
        "/home/unitree/Desktop/DRL/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt",
        24,
        b"<|endoftext|>",
        ["<|endoftext|>"],
    )
    merges = []
    vocab = {}
    for i in range(500):
        # best_pair = get_pairs_count(words_count).most_common(1)[0][0]
        pairs_count = get_pairs_count(words_count)
        best_pair = max(
            pairs_count.items(),
            key=lambda item: (item[1], item[0]),  # (count, pair)
        )[0]
        words_count = merge_pair(words_count, best_pair)
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]


def main():
    from tests.adapters import run_train_bpe

    run_train_bpe(
        input_path="/home/yongce/aws/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )


if __name__ == "__main__":
    import cProfile

    # cProfile.run("test()")
    main()
