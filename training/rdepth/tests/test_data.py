import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_roundtrip():
    tok = Tokenizer.from_file(os.path.join(HERE, "data/tok4096.json"))
    s = "Once upon a time, there was a little girl."
    assert tok.decode(tok.encode(s).ids) == s
    assert tok.get_vocab_size() == 4096

def test_bins():
    for split, floor in [("train", 1_000_000), ("val", 100_000)]:
        arr = np.memmap(os.path.join(HERE, f"data/{split}.bin"), dtype=np.uint16, mode="r")
        assert len(arr) > floor
        assert arr[:100_000].max() < 4096
        assert (arr[:1_000_000] == 0).sum() > 10  # EOT 존재
