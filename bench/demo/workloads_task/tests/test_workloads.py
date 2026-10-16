import json

import driver
import workloads as w


def test_decode_requests_all_arrive_at_zero():
    sessions = w.decode(4, 128, 64)
    assert [s["id"] for s in sessions] == [0, 1, 2, 3]
    assert all(s["arrival"] == 0.0 and s["system"] == 0 and s["turns"] == [w.turn(128, 64)] for s in sessions)


def test_long_prompt_arrives_behind_the_decoders():
    sessions = w.long_prompt(3, 256, 1024, 30000, 2.0)
    assert len(sessions) == 4 and sessions[-1]["arrival"] == 2.0
    assert sessions[-1]["turns"] == [w.turn(30000, 32)]


def test_tokens_are_deterministic_and_inside_the_vocabulary():
    a, b = w.tokens("s", "k", 50), w.tokens("s", "k", 50)
    assert a == b != w.tokens("s", "other", 50)
    assert all(w.VOCAB[0] <= t < w.VOCAB[1] for t in a)
    corpus = list(range(10))
    assert all(t in corpus for t in w.tokens("s", "k", 25, corpus))


def test_appends_give_each_turn_its_new_tokens():
    session = {"id": 3, "arrival": 0.0, "system": 0, "turns": [w.turn(5, 1), w.turn(7, 1)]}
    assert [len(a) for a in w.appends("salt", session)] == [5, 7]


def test_context_len_counts_the_system_prompt_and_every_turn():
    session = {"id": 0, "arrival": 0.0, "system": 100, "turns": [w.turn(10, 5), w.turn(20, 7)]}
    assert w.context_len(session) == 142


def test_agent_trace_is_seeded_and_shares_its_system_prompt():
    assert w.agent(6, 0.2, 1) == w.agent(6, 0.2, 1) != w.agent(6, 0.2, 2)
    sessions = w.agent(6, 0.2, 1, system=512)
    assert all(s["system"] == 512 and 3 <= len(s["turns"]) <= 8 for s in sessions)


def test_lanes_chain_sessions_into_closed_loops():
    sessions = w.lanes(w.decode(5, 8, 8), 2, 0)
    assert [s.get("after") for s in sessions] == [None, None, 0, 1, 2]


def test_write_then_load_round_trips(tmp_path):
    sessions = w.agent(3, 0.5, 4)
    w.write(tmp_path / "t.jsonl", sessions)
    assert w.load(tmp_path / "t.jsonl") == sessions
    assert all(json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines())


def test_driver_replays_prompts_from_the_same_token_stream():
    assert driver.appends is w.appends and driver.tokens is w.tokens
