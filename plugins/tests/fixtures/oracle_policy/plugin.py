from ..oracle import ORACLE


class Plugin:
    """Asks for ORACLE.k drafts per step (mode "alternate": every other step only), and tells the drafter which
    sequence each slot holds."""

    def register(self, regs) -> None:
        pass

    def k(self, seqs) -> int:
        known = False
        for seq in seqs:
            full = ORACLE.sequences.get(tuple(seq.tokens[: seq.prompt_len]))
            if full is not None and seq.slot >= 0:
                ORACLE.by_slot[seq.slot] = full
                (ORACLE.hidden_slots.add if full[: seq.prompt_len] in map(list, ORACLE.hidden)
                 else ORACLE.hidden_slots.discard)(seq.slot)
                known = True
        if not known:
            return 0
        if ORACLE.mode == "alternate":
            ORACLE.steps += 1
            return ORACLE.k if ORACLE.steps % 2 else 0
        return ORACLE.k

    def observe(self, seq, k, n_draft, n_accept) -> None:
        ORACLE.observed.append((tuple(seq.tokens[: seq.prompt_len]), k, n_draft, n_accept))
