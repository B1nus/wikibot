import regex as re
import pickle

FORCED_SPLIT_PAT = re.compile(r""" ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
TOKENIZER_PATH = "tokenizer.pkl"

class Tokenizer():
    def __init__(self):
        self.vocab = {x: bytes([x]) for x in range(256)}
        self.merges = {}

    def train(self, text, vocab_size):
        split = Tokenizer.preprocess_string(text)
        while len(self.vocab) < vocab_size:
            counts = Tokenizer.pair_counts(split)

            idx = len(self.vocab)
            pair = max(counts)
            if split == self.merge(split, pair, idx):
                print("oh no")
            split = self.merge(split, pair, idx)
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        split = Tokenizer.preprocess_string(text)
        while True:
            counts = Tokenizer.pair_counts(split)
            pair = min(counts, key=self.merges.get(float("inf")))
            if pair not in self.merges:
                break
            else:
                split = self.merge(split, pair, self.merges[pair])
        return [x for xs in split for x in xs]

    def pair_counts(split):
        counts = {}
        for string in split:
            for pair in zip(string, string[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, split, pair, idx):
        self.merges[pair] = idx
        newsplit = []
        for string in split:
            newstring = []
            i = 0
            while i < len(string):
                if i < len(string) - 1 and (string[i], string[i + 1]) == pair:
                    newstring.append(idx)
                    i += 2
                else:
                    newstring.append(string[i])
                    i += 1
            newsplit.append(newstring)
        return newsplit

    def preprocess_string(text):
        split = re.findall(FORCED_SPLIT_PAT, text)
        split = list(map(lambda x: list(map(int, x.encode("utf-8"))), split))
        return split

    def decode(self, tokens):
        text = b"".join([self.vocab[token] for token in tokens])
        return text.decode("utf-8", errors="replace")

    def save(self, path=TOKENIZER_PATH):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load(path=TOKENIZER_PATH):
        with open(path, 'rb') as file:
            return pickle.load(file)

