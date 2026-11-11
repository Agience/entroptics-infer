"""The command-line path, end to end.

The library's tests pin what the reads do. These pin that the *command* does the same thing, because
a proof-of-concept that quietly differs from the machinery it demonstrates is worse than none.

What is held here
-----------------
* **Refusal reaches the exit code.** A corpus that refuses must not exit 0. Anything wrapping this
  command reads the status, and a refusal that looks like success is the one failure mode that
  matters.
* **No default `far`.** It is the reader's number and the one this program does not derive, so
  invoking without it must fail rather than pick one.
* **The command binds what the library binds.** Same answer, not merely an answer.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics_infer import __main__ as cli
from entroptics_infer import bind

FAR = 0.05
SEED = 20260915


def _write(dirpath, name, frame):
    header = ",".join(f"ch{j}" for j in range(frame.shape[1]))
    np.savetxt(dirpath / f"{name}.csv", frame, delimiter=",", header=header, comments="")


def _structured(tmp_path, rng, n_keys=5, T=96, D=32, facets=6):
    """A query, a broad true match, and narrow distractors — the discriminating case."""
    loads = np.linalg.qr(rng.normal(size=(D, facets)))[0]
    latents = rng.normal(size=(T, facets))

    def emit(idx, amp):
        X = np.zeros((T, D))
        for f in idx:
            X += np.outer(latents[:, f], loads[:, f]) * amp
        return X + rng.normal(size=(T, D)) * 0.10

    broad = tuple(range(facets))
    _write(tmp_path, "query", emit(broad, 1 / np.sqrt(facets)))
    _write(tmp_path, "true_match", emit(broad, 1 / np.sqrt(facets)))
    for i in range(n_keys):
        sub = tuple(int(f) for f in rng.choice(facets, size=facets - 1, replace=False))
        _write(tmp_path, f"distractor{i}", emit(sub, 2.0 / np.sqrt(facets - 1)))
    return tmp_path


def _noise(tmp_path, rng, n=5, T=96, D=8):
    for name in ["query"] + [f"r{i}" for i in range(n - 1)]:
        _write(tmp_path, name, rng.normal(size=(T, D)))
    return tmp_path


def test_far_has_no_default():
    """Invoking without a level must fail, not choose one."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["describe", "."])
    assert exc.value.code != 0


def test_describe_reports_a_count_for_every_record(tmp_path, capsys):
    _structured(tmp_path, np.random.default_rng(SEED))
    assert cli.main(["--far", str(FAR), "describe", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "query" in out and "true_match" in out
    assert "per channel" in out


def test_a_structured_corpus_is_admitted(tmp_path, capsys):
    _structured(tmp_path, np.random.default_rng(SEED))
    code = cli.main(["--far", str(FAR), "admit", str(tmp_path),
                     "--seed", str(SEED), "--draws", "3"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "ADMITTED" in out


def test_noise_is_refused_and_says_so_in_the_exit_code(tmp_path, capsys):
    """The failure mode that matters: a refusal that looks like success to a caller."""
    _noise(tmp_path, np.random.default_rng(3))
    code = cli.main(["--far", str(FAR), "admit", str(tmp_path),
                     "--seed", str(SEED), "--draws", "3"])
    assert "REFUSED" in capsys.readouterr().out
    assert code != 0, "a refused corpus exited 0, so anything reading the status sees success"


def test_ask_refuses_on_noise_rather_than_returning_the_nearest(tmp_path, capsys):
    _noise(tmp_path, np.random.default_rng(11))
    code = cli.main(["--far", str(FAR), "ask", str(tmp_path), "--query", "query"])
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert code != 0


@pytest.mark.parametrize("build", [_structured, _noise])
def test_the_command_decides_what_the_library_decides(tmp_path, capsys, build):
    """The command's verdict is `bind.decide`'s verdict, whichever way it falls.

    Deliberately *not* "the command finds the true match". What needs holding is that the command's
    own loading and scaling did not change what is being measured — and a test that only checked the
    outcome it wanted would pass a command that agreed by luck on one corpus. So both a corpus that
    binds and one that refuses go through the same assertion, and the expected answer is read from
    the library rather than written down here.
    """
    build(tmp_path, np.random.default_rng(SEED))
    code = cli.main(["--far", str(FAR), "ask", str(tmp_path), "--query", "query"])
    out = capsys.readouterr().out

    records, _ = cli._load(tmp_path)
    others = [k for k in sorted(records) if k != "query"]
    direct = bind.decide(records["query"], [records[k] for k in others], far=FAR)

    if direct.key is None:
        assert code != 0 and "REFUSED" in out
        assert direct.why in out, "the command should report the library's own reason"
    else:
        assert code == 0 and "BOUND to" in out
        chosen = out.split("BOUND to ")[1].split("\n")[0].strip()
        assert chosen == others[direct.key]


def test_records_of_different_shapes_are_refused(tmp_path):
    """An association is a re-pairing of rows, so mismatched records have no reading."""
    rng = np.random.default_rng(SEED)
    _write(tmp_path, "a", rng.normal(size=(96, 8)))
    _write(tmp_path, "b", rng.normal(size=(48, 8)))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--far", str(FAR), "describe", str(tmp_path)])
    assert "shape" in str(exc.value)


def test_an_empty_directory_is_refused(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--far", str(FAR), "describe", str(tmp_path)])
    assert "no .csv" in str(exc.value)


def _triples(tmp_path, rng, kinds=4, sources=4, T=96, D=32):
    """Every source related to one target per kind, so only the *relation* separates the targets.

    The construction the forward pass is built for: a method that ignores which relation is asked
    about cannot beat chance among a source's own targets.
    """
    roles = np.linalg.qr(rng.normal(size=(D, 2 * kinds)))[0]
    recs = {f"a{i}": np.zeros((T, D)) for i in range(sources)}
    for i in range(sources):
        for k in range(kinds):
            s = rng.normal(size=T)
            recs[f"a{i}"] = recs[f"a{i}"] + np.outer(s, roles[:, 2 * k])
            recs[f"a{i}_k{k}"] = np.outer(s, roles[:, 2 * k + 1])
    for name in list(recs):
        _write(tmp_path, name, recs[name] + rng.normal(size=(T, D)) * 0.10)
    return tmp_path


def test_complete_runs_the_forward_pass(tmp_path, capsys):
    """Two of a triple in, the third out — with the relation given only by example."""
    _triples(tmp_path, np.random.default_rng(SEED))
    code = cli.main(["--far", str(FAR), "complete", str(tmp_path),
                     "--exemplar", "a0", "a0_k0", "--source", "a1",
                     "--candidates", "a1_k0", "a1_k1", "a1_k2", "a1_k3"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "COMPLETED with a1_k0" in out, out


def test_complete_refuses_where_nothing_stands_in_the_relation(tmp_path, capsys):
    """A corpus with no relational structure completes nothing, and says so in the exit code."""
    _noise(tmp_path, np.random.default_rng(5), n=6)
    names = sorted(p.stem for p in tmp_path.glob("*.csv"))
    code = cli.main(["--far", str(FAR), "complete", str(tmp_path),
                     "--exemplar", names[0], names[1], "--source", names[2]])
    out = capsys.readouterr().out
    assert "REFUSED" in out, out
    assert code != 0


def test_complete_names_a_record_that_is_not_there(tmp_path):
    _triples(tmp_path, np.random.default_rng(SEED))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--far", str(FAR), "complete", str(tmp_path),
                  "--exemplar", "a0", "nope", "--source", "a1"])
    assert "nope" in str(exc.value)


def _chain(tmp_path, rng, length=12, transmission=0.7, T=96, D=32, channels=4):
    """A line of records, each carrying a fraction of the one before it.

    The origin's surviving share at step `t` is `transmission ** t`, so the signal decays
    geometrically and crosses the floor after finitely many steps. Where it crosses is what `reach`
    reads off, and it is never told the transmission.
    """
    def structured():
        dirs = np.linalg.qr(rng.normal(size=(D, channels)))[0]
        X = np.zeros((T, D))
        for k in range(channels):
            X += np.outer(rng.normal(size=T), dirs[:, k]) / np.sqrt(channels)
        return X

    carried = structured()
    keep = transmission
    fresh = np.sqrt(max(0.0, 1.0 - keep * keep))
    order = []
    for t in range(length):
        if t:
            carried = keep * carried + fresh * structured()
        name = f"c{t}"
        order.append(name)
        _write(tmp_path, name, carried + rng.normal(size=(T, D)) * 0.05)

    edges = tmp_path / "edges.csv"
    with edges.open("w", encoding="utf-8", newline="") as fh:
        fh.write("from,to\n")
        for a, b in zip(order, order[1:]):
            fh.write(f"{a},{b}\n")
    return edges, order


def test_reach_measures_a_depth_without_being_given_one(tmp_path, capsys):
    """The horizon is read, not declared. There is no depth argument to pass."""
    edges, order = _chain(tmp_path, np.random.default_rng(SEED))
    code = cli.main(["--far", str(FAR), "reach", str(tmp_path),
                     "--edges", str(edges), "--origin", order[0]])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "horizon" in out
    depth = int(out.split("horizon")[1].split()[0])
    assert 0 < depth < len(order), f"a horizon of {depth} over a chain of {len(order)}"


def test_reach_stops_short_of_the_whole_chain(tmp_path, capsys):
    """A decaying signal crosses the floor, so the walk must not run to the end of the corpus.

    This is `Horizon.lean`'s statement as a behaviour: if the walk reached everything, it would be
    running out of chain rather than out of signal, and the depth would measure the corpus.
    """
    edges, order = _chain(tmp_path, np.random.default_rng(SEED), length=24, transmission=0.5)
    cli.main(["--far", str(FAR), "reach", str(tmp_path),
              "--edges", str(edges), "--origin", order[0]])
    out = capsys.readouterr().out
    reached = int(out.split("reached")[1].split()[0])
    assert reached < len(order) - 1, (
        f"reached {reached} of {len(order) - 1}; the walk ran out of chain, not out of signal"
    )


def test_reach_refuses_an_edge_naming_a_record_that_is_not_there(tmp_path):
    edges, order = _chain(tmp_path, np.random.default_rng(SEED))
    with edges.open("a", encoding="utf-8", newline="") as fh:
        fh.write("c0,ghost\n")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--far", str(FAR), "reach", str(tmp_path),
                  "--edges", str(edges), "--origin", order[0]])
    assert "ghost" in str(exc.value)


def test_the_edge_list_is_not_read_as_a_record(tmp_path, capsys):
    """It lives beside the records and is not one, so it is skipped by name."""
    edges, order = _chain(tmp_path, np.random.default_rng(SEED))
    assert cli.main(["--far", str(FAR), "reach", str(tmp_path),
                     "--edges", str(edges), "--origin", order[0]]) == 0
    out = capsys.readouterr().out
    assert f"over {len(order)} records" in out, out


def _lattice_corpus(tmp_path, carry=0.15, sources=40, T=96, D=32, seed=5):
    """Three relations on one set of records: an exact map, a faint one, and none at all.

    The same construction `research/lib/demo_corpus.py` writes, small enough for a test. The faint arm
    is the one that matters — held by the lattice and not namable from any single pair — and the
    `unrelated` arm is what makes that mean something rather than being a reading that grows on
    anything.
    """
    import csv as _csv
    rng = np.random.default_rng(seed)

    def structured():
        loadings = np.linalg.qr(rng.normal(size=(D, 3)))[0][:, :3]
        return rng.normal(size=(T, 3)) @ loadings.T + 0.1 * rng.normal(size=(T, D))

    loud = np.linalg.qr(rng.normal(size=(D, D)))[0]
    quiet = np.linalg.qr(rng.normal(size=(D, D)))[0]
    edges = []
    for i in range(sources):
        src = structured()
        _write(tmp_path, f"s{i}", src)
        _write(tmp_path, f"loud{i}", src @ loud)
        edges.append((f"s{i}", f"loud{i}", "loud"))
        _write(tmp_path, f"quiet{i}",
               carry * (src @ quiet) + np.sqrt(1 - carry ** 2) * structured())
        edges.append((f"s{i}", f"quiet{i}", "quiet"))
        _write(tmp_path, f"none{i}", structured())
        edges.append((f"s{i}", f"none{i}", "unrelated"))
    path = tmp_path / "edges.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["src", "dst", "relation"])
        w.writerows(edges)
    return path


def test_lattice_reports_what_is_held_apart_from_what_is_namable(tmp_path, capsys):
    """The command has to show both readings, because a corpus can pass one and fail the other."""
    edges = _lattice_corpus(tmp_path)
    rc = cli.main(["--far", str(FAR), "lattice", str(tmp_path), "--edges", str(edges),
                   "--seed", "5", "--draws", "250"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "loud" in out and "quiet" in out and "unrelated" in out
    assert "Held but not namable from one example: quiet" in out, (
        "the faint relation should be held and not namable, which is the case this command "
        f"exists to show:\n{out}"
    )


def test_lattice_refuses_an_edge_to_a_record_that_is_not_there(tmp_path):
    """An edge between nothing stands for nothing, and must say so rather than being skipped."""
    edges = _lattice_corpus(tmp_path, sources=4)
    with edges.open("a", encoding="utf-8") as fh:
        fh.write("s0,not_a_record,loud\n")
    with pytest.raises(SystemExit):
        cli.main(["--far", str(FAR), "lattice", str(tmp_path), "--edges", str(edges),
                  "--seed", "5", "--draws", "60"])


def test_lattice_answers_with_a_relation_no_single_pair_could_name(tmp_path, capsys):
    """The point of deriving a relation from every edge rather than from one exemplar."""
    edges = _lattice_corpus(tmp_path)
    rc = cli.main(["--far", str(FAR), "lattice", str(tmp_path), "--edges", str(edges),
                   "--seed", "5", "--draws", "250", "--source", "s7", "--relation", "quiet",
                   "--candidates", "loud7", "quiet7", "none7"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "quiet7" in out.split("What stands in")[-1], (
        f"asked about the faint relation and did not get its destination:\n{out}"
    )
