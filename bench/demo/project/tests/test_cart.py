import pytest
from shop.cart import Cart


def test_add_merges_lines_of_one_sku():
    cart = Cart()
    cart.add("apple", "0.50", 2)
    cart.add("apple", "0.50")
    cart.add("pear", "0.75")
    assert (cart.quantity("apple"), cart.quantity("pear"), cart.quantity("melon")) == (3, 1, 0)
    assert cart.subtotal() == 225


def test_add_refuses_another_price_or_no_units():
    cart = Cart()
    cart.add("apple", "0.50")
    with pytest.raises(ValueError):
        cart.add("apple", "0.55")
    with pytest.raises(ValueError):
        cart.add("pear", "1.00", 0)


def test_remove():
    cart = Cart()
    cart.add("apple", "0.50", 2)
    cart.remove("apple")
    assert cart.quantity("apple") == 1
    cart.remove("apple")
    assert "apple" not in cart.lines and cart.subtotal() == 0
    with pytest.raises(KeyError):
        cart.remove("apple")
    cart.add("pear", "0.75")
    with pytest.raises(ValueError):
        cart.remove("pear", 2)


def test_item_lines_are_sorted_with_line_totals():
    cart = Cart()
    cart.add("tea", "3.05")
    cart.add("bread", "2.50", 2)
    cart.add("gum", "0.01", 3)
    assert cart.item_lines() == ["bread x2 5.00", "gum x3 0.03", "tea x1 3.05"]
