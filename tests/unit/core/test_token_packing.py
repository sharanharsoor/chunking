"""Token packing: greedy units and sliding windows. No chars/4."""

from chunking_strategy.core.token_packing import pack_unit_ranges, token_windows


def test_pack_emits_oversize_unit_alone():
    # 3, 10, 2 with budget 5 → [0:1], [1:2], [2:3]
    assert pack_unit_ranges([3, 10, 2], max_tokens=5) == [(0, 1), (1, 2), (2, 3)]


def test_pack_overlap_walks_back_units():
    # sizes 3,3,3 budget 6 overlap 3 → [0:2] then restart at 1
    got = pack_unit_ranges([3, 3, 3], max_tokens=6, overlap_tokens=3)
    assert got[0] == (0, 2)
    assert got[1][0] == 1


def test_token_windows_step():
    assert token_windows(10, max_tokens=4, overlap_tokens=1) == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]
