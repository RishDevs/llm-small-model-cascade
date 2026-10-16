from cascade.prompts import messages, parse_label

LABELS = {"book_flight", "oos", "pin_change"}


def test_plain_label():
    assert parse_label("book_flight", LABELS) == "book_flight"


def test_messy_free_form():
    assert parse_label(' "Book flight". ', LABELS) == "book_flight"
    assert parse_label("PIN-change", LABELS) == "pin_change"


def test_json_output():
    assert parse_label('{"label": "oos"}', LABELS) == "oos"


def test_rejects_unknown_and_chatter():
    assert parse_label("The intent is book_flight", LABELS) is None
    assert parse_label("{not json", LABELS) is None
    assert parse_label(None, LABELS) is None


def test_label_list_sits_in_shared_prefix():
    a = messages("first", ["a", "b"])
    b = messages("second", ["a", "b"])
    assert a[0] == b[0]
    assert a[1]["content"] == "first"
