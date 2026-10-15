import pytest
from shop import discounts
from shop.cart import Cart


def _cart(**lines):
    cart = Cart()
    for sku, (price, qty) in lines.items():
        cart.add(sku, price, qty)
    return cart


def test_percent_off_applies_to_matching_skus():
    cart = _cart(tea_green=("4.00", 1), tea_black=("3.00", 2), bread=("2.00", 1))
    assert discounts.checkout(cart, [discounts.percent_off("tea", 10)]) == [
        "bread x1 2.00", "tea_black x2 6.00", "tea_green x1 4.00", "10% off tea* -1.00", "TOTAL 11.00"]
    with pytest.raises(ValueError):
        discounts.percent_off("tea", 0)


def test_a_rule_that_does_not_apply_adds_no_line():
    cart = _cart(bread=("2.00", 1))
    assert discounts.checkout(cart, [discounts.percent_off("tea", 10)]) == ["bread x1 2.00", "TOTAL 2.00"]


@pytest.mark.parametrize("qty, free", [(1, 0), (2, 0), (3, 1), (5, 1), (6, 2), (7, 2)])
def test_buy_n_get_one(qty, free):
    """Buy 2, get one free: of every 3 units, one is free."""
    cart = _cart(soap=("1.25", qty), sponge=("0.40", 3))
    applied = discounts.buy_n_get_one("soap", 2)(cart)
    assert applied == (("buy 2 get 1 soap", 125 * free) if free else None)


def test_buy_n_get_one_needs_a_positive_n():
    with pytest.raises(ValueError):
        discounts.buy_n_get_one("soap", 0)


def test_checkout_applies_rules_in_order_and_never_goes_below_zero():
    cart = _cart(soap=("1.25", 3), sponge=("0.05", 1))
    rules = [discounts.buy_n_get_one("soap", 2), discounts.percent_off("s", 100)]
    assert discounts.checkout(cart, rules) == [
        "soap x3 3.75", "sponge x1 0.05", "buy 2 get 1 soap -1.25", "100% off s* -2.55", "TOTAL 0.00"]
