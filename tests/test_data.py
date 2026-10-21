from pr_lora.data import answer_text, split_records


def _records(n=8):
    return [
        {
            "question": str(i),
            "answer_matching_behavior": ["matching"],
            "answer_not_matching_behavior": "opposing",
        }
        for i in range(n)
    ]


def test_answer_text_accepts_strings_and_lists():
    assert answer_text(" answer ") == "answer"
    assert answer_text([" answer "]) == "answer"


def test_split_is_disjoint_and_deterministic():
    records = _records()
    first = split_records(records, direction_size=3, train_size=2, evaluation_size=3, seed=42)
    second = split_records(records, direction_size=3, train_size=2, evaluation_size=3, seed=42)
    assert first == second
    assert [len(part) for part in first] == [3, 2, 3]
    assert len({record["question"] for part in first for record in part}) == len(records)
