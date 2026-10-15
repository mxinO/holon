import pytest
from shop.money import format_cents, parse_price, percent_of


@pytest.mark.parametrize("text, cents", [("3.50", 350), ("3", 300), ("0.5", 50), (" 12.05 ", 1205), ("0", 0)])
def test_parse_price(text, cents):
    assert parse_price(text) == cents


@pytest.mark.parametrize("text", ["-1.00", "1.234", "abc", "", "1.2.3", "1,50"])
def test_parse_price_rejects(text):
    with pytest.raises(ValueError):
        parse_price(text)


@pytest.mark.parametrize("cents, text", [(350, "3.50"), (5, "0.05"), (1205, "12.05"), (-125, "-1.25"), (0, "0.00")])
def test_format_cents(cents, text):
    assert format_cents(cents) == text


@pytest.mark.parametrize("cents", [0, 1, 9, 10, 99, 100, 101, 1999, 100000])
def test_format_and_parse_round_trip(cents):
    assert parse_price(format_cents(cents)) == cents


def test_percent_of_rounds_half_up():
    assert (percent_of(999, 10), percent_of(5, 10), percent_of(4, 10), percent_of(200, 50)) == (100, 1, 0, 100)
