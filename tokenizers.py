"""Tokenizers for critical-tokenization experiments.

All tokenizers are lossless (decode(tokenize(x)) == x) so L_res = 0,
which lets us isolate the tokenizer's effect on D_T, alpha_T and BPC.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def pretokenize(text: str) -> List[str]:
    chunks: List[str] = []
    buf = ""
    cat = None
    for ch in text:
        if ch.isspace():
            c = "space"
        elif ch.isalnum():
            c = "alnum"
        else:
            c = "other"
        if c == cat and c != "other":
            buf += ch
        else:
            if buf:
                chunks.append(buf)
            buf = ch
            cat = c
    if buf:
        chunks.append(buf)
    return chunks


class CharTokenizer:
    name = "char"

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab: Dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.unk_id = len(self.chars)

    def encode(self, text: str) -> List[int]:
        v = self.vocab
        unk = self.unk_id
        return [v.get(c, unk) for c in text]

    def id_to_piece(self, i: int) -> str:
        return self.chars[i] if i < len(self.chars) else "<unk>"

    @property
    def actual_vocab_size(self) -> int:
        return len(self.chars) + 1

    def pieces(self) -> List[str]:
        return list(self.chars)


class WordTokenizer:
    name = "word"

    def __init__(self, text: str):
        chunks = pretokenize(text)
        words = [c for c in chunks if not c.isspace()]
        self.vocab: Dict[str, int] = {w: i for i, w in enumerate(sorted(set(words)))}
        self.vocab[" "] = len(self.vocab)
        self._space_id = self.vocab[" "]

    def encode(self, text: str) -> List[int]:
        v = self.vocab
        unk = len(v)
        out: List[int] = []
        for c in pretokenize(text):
            if c.isspace():
                out.append(self._space_id)
            elif c in v:
                out.append(v[c])
            else:
                for ch in c:
                    out.append(v.get(ch, unk))
        return out

    @property
    def actual_vocab_size(self) -> int:
        return len(self.vocab) + 1

    def pieces(self) -> List[str]:
        return list(self.vocab.keys())


class BpeTokenizer:
    name = "bpe"

    def __init__(self, train_text: str, vocab_size: int):
        self.vocab_size = vocab_size
        chunks = pretokenize(train_text)
        self.base_chars = sorted(set(train_text))
        self.char2id: Dict[str, int] = {c: i for i, c in enumerate(self.base_chars)}
        self.merges: List[Tuple[Tuple[str, str], str]] = []
        self._train(chunks, vocab_size)
        self._merge_rank: Dict[Tuple[str, str], int] = {
            pair: i for i, (pair, _) in enumerate(self.merges)
        }
        self._merge_to_id: Dict[str, int] = {
            new: len(self.base_chars) + i for i, (_, new) in enumerate(self.merges)
        }
        self.unk_id: int = len(self.base_chars) + len(self.merges)
        self._piece_cache: Dict[str, List[int]] = {}

    def _train(self, chunks: List[str], vocab_size: int):
        word_counts = Counter(chunks)
        words = [[list(w), c] for w, c in word_counts.items()]
        base = len(self.base_chars)
        target_merges = max(0, vocab_size - base)
        pair_count: Counter = Counter()
        pair_words: Dict[Tuple[str, str], set] = defaultdict(set)
        for i, (s, c) in enumerate(words):
            for j in range(len(s) - 1):
                p = (s[j], s[j + 1])
                pair_count[p] += c
                pair_words[p].add(i)
        next_id = base
        for _ in range(target_merges):
            if not pair_count:
                break
            best, best_cnt = pair_count.most_common(1)[0]
            if best_cnt < 2:
                break
            a, bb = best
            new_sym = f"[{next_id}:{a + bb}]"
            self.merges.append((best, new_sym))
            for i in list(pair_words[best]):
                s, c = words[i]
                for j in range(len(s) - 1):
                    p = (s[j], s[j + 1])
                    pair_count[p] -= c
                    if pair_count[p] <= 0:
                        del pair_count[p]
                    pair_words[p].discard(i)
                new_s = []
                j = 0
                n = len(s)
                while j < n:
                    if j < n - 1 and s[j] == a and s[j + 1] == bb:
                        new_s.append(new_sym)
                        j += 2
                    else:
                        new_s.append(s[j])
                        j += 1
                for j in range(len(new_s) - 1):
                    p = (new_s[j], new_s[j + 1])
                    pair_count[p] += c
                    pair_words[p].add(i)
                words[i][0] = new_s
            pair_count.pop(best, None)
            pair_words.pop(best, None)
            next_id += 1

    @staticmethod
    def _merge_in(syms: Tuple[str, ...], pair: Tuple[str, str], new_sym: str):
        out = []
        i = 0
        n = len(syms)
        changed = False
        while i < n:
            if i < n - 1 and syms[i] == pair[0] and syms[i + 1] == pair[1]:
                out.append(new_sym)
                i += 2
                changed = True
            else:
                out.append(syms[i])
                i += 1
        return tuple(out) if changed else syms

    def _bpe_piece(self, word: str) -> List[int]:
        cached = self._piece_cache.get(word)
        if cached is not None:
            return cached
        syms = list(word)
        rank = self._merge_rank
        while len(syms) > 1:
            best_i = -1
            best_rank = None
            for i in range(len(syms) - 1):
                r = rank.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i == -1:
                break
            merged = self.merges[best_rank][1]
            syms[best_i : best_i + 2] = [merged]
        c2 = self.char2id
        m2id = self._merge_to_id
        unk = self.unk_id
        ids = []
        for s in syms:
            if s in c2:
                ids.append(c2[s])
            elif s in m2id:
                ids.append(m2id[s])
            else:
                ids.append(unk)
        self._piece_cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for c in pretokenize(text):
            if c.isspace():
                out.append(self.char2id.get(c[0], self.char2id.get(" ", 0)))
            else:
                out.extend(self._bpe_piece(c))
        return out

    @property
    def actual_vocab_size(self) -> int:
        return len(self.base_chars) + len(self.merges) + 1

    def restrict(self, vocab_size: int) -> "BpeTokenizer":
        n_keep = max(0, vocab_size - len(self.base_chars))
        return self.restrict_to(range(n_keep))

    def restrict_to(self, merge_indices) -> "BpeTokenizer":
        idx = sorted(set(int(i) for i in merge_indices))
        idx = [i for i in idx if 0 <= i < len(self.merges)]
        b = BpeTokenizer.__new__(BpeTokenizer)
        b.vocab_size = len(self.base_chars) + len(idx)
        b.base_chars = self.base_chars
        b.char2id = self.char2id
        b.merges = self.merges
        b._merge_rank = {self.merges[i][0]: i for i in idx}
        b._merge_to_id = {self.merges[i][1]: self._merge_to_id[self.merges[i][1]] for i in idx}
        b.unk_id = self.unk_id
        b._piece_cache = {}
        return b

    def n_active_merges(self) -> int:
        return len(self._merge_rank)

    def pieces(self) -> List[str]:
        return list(self.base_chars) + [new for _, new in self.merges]
