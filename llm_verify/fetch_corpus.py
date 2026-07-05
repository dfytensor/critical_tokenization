"""Fetch a large English corpus from Project Gutenberg (resumable, dedup)."""
import os
import time
import urllib.request

OUT = r"F:\OpenASH2605\critical_tokenization\llm_verify\en_large.txt"
DONE = r"F:\OpenASH2605\critical_tokenization\llm_verify\en_done.txt"
TARGET = 26_000_000

GUTENBERG_IDS = [
    1342, 11, 2701, 84, 98, 174, 1661, 345, 844, 1080, 1184, 1232, 1260, 1322,
    1349, 1513, 161, 1952, 203, 236, 25930, 2852, 30254, 33283, 408, 4300,
    4360, 46, 514, 5200, 55, 58269, 6130, 74, 768, 815, 105, 113, 120, 1250,
    135, 1400, 1604, 1999, 2147, 2148, 22381, 23529, 24869, 2554, 2542, 2892,
    32037, 32913, 36099, 37106, 403, 501, 730, 1001, 1064, 1399, 1902, 20792,
    23, 244, 27827, 2921, 32181, 53453, 67098, 737, 851, 9192, 10833, 11839,
    1228, 14287, 1597, 164, 1709, 1822, 19505, 2097, 219, 2363, 2500, 2620,
    2782, 28714, 30127, 3171, 33520, 3369, 34932, 36148, 376, 3798, 4085,
    4246, 4361, 4597, 4633, 50037, 5219, 55034, 5726, 60062, 6128, 64356,
    6685, 6810, 7148, 7300, 7666, 786, 8159, 833, 880016, 902, 9175, 9472,
    964, 9795, 985, 9952,
]


def load_done():
    try:
        with open(DONE, encoding="utf-8") as f:
            return set(int(x) for x in f.read().split())
    except FileNotFoundError:
        return set()


def strip(raw: str) -> str:
    for tag in ("*** START OF", "*** START OF THE PROJECT GUTENBERG"):
        i = raw.find(tag)
        if i != -1:
            raw = raw[i:]
            j = raw.find("***")
            if j != -1:
                raw = raw[j + 3:]
            break
    for tag in ("*** END OF", "*** END OF THE PROJECT GUTENBERG"):
        i = raw.find(tag)
        if i != -1:
            raw = raw[:i]
            break
    return raw.strip()


def fetch_one(bid: int) -> str:
    for url in (f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
                f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt"):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception:
            continue
    return ""


def current_size():
    return os.path.getsize(OUT) if os.path.exists(OUT) else 0


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = load_done()
    total = current_size()
    if total >= TARGET:
        print(f"already at {total:,} >= {TARGET}")
        return
    print(f"resume: {total:,} chars, {len(done)} books done", flush=True)
    got_new = 0
    with open(OUT, "a", encoding="utf-8") as fout, open(DONE, "a", encoding="utf-8") as fdone:
        for bid in GUTENBERG_IDS:
            if total >= TARGET:
                break
            if bid in done:
                continue
            raw = fetch_one(bid)
            if not raw:
                print(f"  [{bid}] empty", flush=True)
                done.add(bid); fdone.write(f"{bid}\n"); fdone.flush()
                continue
            txt = strip(raw)
            if len(txt) < 5000:
                print(f"  [{bid}] too short", flush=True)
                done.add(bid); fdone.write(f"{bid}\n"); fdone.flush()
                continue
            fout.write(txt + "\n\n"); fout.flush()
            done.add(bid); fdone.write(f"{bid}\n"); fdone.flush()
            total += len(txt); got_new += 1
            print(f"  [{bid}] +{len(txt):,} (total {total:,}, +{got_new} new)", flush=True)
            time.sleep(0.2)
    print(f"DONE: {total:,} chars ({len(done)} books tried) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
