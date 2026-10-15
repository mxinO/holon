"""Prices as integer cents: parsed from and formatted to text with two decimals."""


def parse_price(text: str) -> int:
    """'3.50' -> 350, '3' -> 300, '0.5' -> 50, ' 12.05 ' -> 1205. A negative price, more than two decimals or anything
    that is not a number raises ValueError."""
    text = text.strip()
    whole, _, frac = text.partition(".")
    if not whole.isdigit() or (frac and not frac.isdigit()) or len(frac) > 2:
        raise ValueError(f"not a price: {text!r}")
    return int(whole) * 100 + int(frac.ljust(2, "0"))


def format_cents(cents: int) -> str:
    """350 -> '3.50', 5 -> '0.05', 1205 -> '12.05', -125 -> '-1.25'."""
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100}"


def percent_of(cents: int, percent: int) -> int:
    """`percent` % of `cents`, rounded half up to a whole cent: percent_of(999, 10) -> 100, percent_of(5, 10) -> 1."""
    return (cents * percent + 50) // 100
