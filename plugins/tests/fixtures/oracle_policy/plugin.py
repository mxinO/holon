from ..oracle import ORACLE


class Plugin:
    """Asks for ORACLE.k drafts for every known prompt (mode "alternate": every other step only), and tells the
    drafter which sequence each slot holds."""

    def register(self, regs) -> None:
        pass

    def choose(self, seq):
        full = ORACLE.sequences.get(tuple(seq.tokens[: seq.prompt_len]))
        if full is None:
            return "", 0
        ORACLE.by_slot[seq.slot] = full
        if ORACLE.mode == "alternate":
            ORACLE.steps += 1
            return "tests.oracle.drafter", ORACLE.k if ORACLE.steps % 2 else 0
        return "tests.oracle.drafter", ORACLE.k

    def observe(self, seq, k, n_accept, step_ms) -> None:
        pass
