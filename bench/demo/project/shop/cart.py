"""A shopping cart: one line per SKU with its unit price (cents) and quantity."""
from __future__ import annotations

import dataclasses

from .money import format_cents, parse_price


@dataclasses.dataclass
class Line:
    sku: str
    unit_cents: int
    qty: int

    @property
    def cents(self) -> int:
        return self.unit_cents * self.qty


@dataclasses.dataclass
class Cart:
    lines: dict[str, Line] = dataclasses.field(default_factory=dict)

    def add(self, sku: str, price: str, qty: int = 1) -> None:
        """Add `qty` units of `sku` at `price` (text, e.g. '2.50'). Adding a SKU again adds to its quantity; its price
        must be the same (ValueError otherwise), as must a positive `qty`."""
        if qty <= 0:
            raise ValueError(f"quantity must be positive: {qty}")
        cents = parse_price(price)
        line = self.lines.get(sku)
        if line is None:
            self.lines[sku] = Line(sku, cents, qty)
        elif line.unit_cents != cents:
            raise ValueError(f"{sku}: price {price} differs from {format_cents(line.unit_cents)}")
        else:
            line.qty += qty

    def remove(self, sku: str, qty: int = 1) -> None:
        """Remove `qty` units of `sku`; the line goes when none are left. KeyError for a SKU not in the cart,
        ValueError for more units than it holds."""
        line = self.lines[sku]
        if qty > line.qty:
            raise ValueError(f"{sku}: cannot remove {qty} of {line.qty}")
        line.qty -= qty
        if line.qty == 0:
            del self.lines[sku]

    def quantity(self, sku: str) -> int:
        line = self.lines.get(sku)
        return line.qty if line else 0

    def subtotal(self) -> int:
        return sum(line.cents for line in self.lines.values())

    def item_lines(self) -> list[str]:
        """One line per SKU, sorted by SKU: '<sku> x<qty> <line total>'."""
        return [f"{line.sku} x{line.qty} {format_cents(line.cents)}" for _, line in sorted(self.lines.items())]
