from sandbag_bench.data import LETTERS, MMLUItem


def test_format_prompt_shape():
    item = MMLUItem(
        item_id="test-0",
        subject="philosophy",
        question="Which letter comes first?",
        choices=("A first", "B second", "C third", "D fourth"),
        gold="A",
    )
    p = item.format_prompt()
    assert "Question: Which letter" in p
    for L in LETTERS:
        assert f"\n{L}. " in p
    assert p.rstrip().endswith("Answer:")
