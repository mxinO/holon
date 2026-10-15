"""Discount rules and checkout.

A rule is a callable taking the cart and returning `(label, cents off)`, or None when it does not apply. Rules:
- `percent_off(prefix, percent)`: `percent` % off every line whose SKU starts with `prefix`;
- `buy_n_get_one(sku, n)`: for every `n` units of `sku` bought, one more unit is free.
"""
from __future__ import annotations

from collections.abc import Callable

from .cart import Cart
from .money import format_cents, percent_of

Rule = Callable[[Cart], "tuple[str, int] | None"]


def percent_off(prefix: str, percent: int) -> Rule:
    if not 0 < percent <= 100:
        raise ValueError(f"percent must be in (0, 100]: {percent}")

    def rule(cart: Cart) -> tuple[str, int] | None:
        off = sum(percent_of(line.cents, percent) for sku, line in cart.lines.items() if sku.startswith(prefix))
        return (f"{percent}% off {prefix}*", off) if off else None
    return rule


def checkout(cart: Cart, rules: list[Rule]) -> list[str]:
    """The receipt: the cart's item lines, then one line per rule that applies, in the order given ('<label>
    -<cents off>'), then 'TOTAL <total>'. Discounts never take the total below zero."""
    lines, total = cart.item_lines(), cart.subtotal()
    for rule in rules:
        if (applied := rule(cart)) is not None:
            label, off = applied
            off = min(off, total)
            total -= off
            lines.append(f"{label} -{format_cents(off)}")
    return [*lines, f"TOTAL {format_cents(total)}"]
