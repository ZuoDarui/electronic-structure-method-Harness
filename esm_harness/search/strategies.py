"""Named search strategies. The loop asks each one how many samples to draw."""

from __future__ import annotations


class Iterative:
    name = "iterative"

    def samples(self, factors: dict) -> int:
        return 1


class BestOfN:
    name = "best_of_n"

    def samples(self, factors: dict) -> int:
        return int(factors["population_n"])


STRATEGIES = {Iterative.name: Iterative(), BestOfN.name: BestOfN()}


def strategy_for(factors: dict):
    try:
        return STRATEGIES[factors["search"]]
    except KeyError as exc:
        raise ValueError(f"unknown search {factors['search']!r}") from exc
