"""Talk to it. A conversational shell over a lattice, answering in sentences.

Why a conversation rather than a command
------------------------------------------
Every read in this package returns numbers, and a number is only an answer if you already know what
it means. `z = 3.67, shared = 2` is the evidence behind an answer, not the answer. This turns the
same reads into something you can question the way you would question a student: ask what it knows,
ask whether two things are related, ask it to finish an analogy, ask what it cannot tell apart.

What it can and cannot say
----------------------------
It knows only what it measured on the lattice in front of it. It has no vocabulary, no training,
and no opinions: every sentence below is produced from a read taken when the question was asked.

It says "I don't know" and means it. Where nothing resolves above a record's own floor, or where
two candidates are equally good, the reads return nothing rather than a guess, and so does this. A
student who answers everything has told you nothing about which answers to trust.

It has no memory of the conversation beyond the last answer, which `why` explains. That is a real
limit, stated rather than hidden.
"""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]

import curriculum as C                                              # noqa: E402
from entroptics_infer import complete, engine, relations            # noqa: E402

FAR = 0.05


def _confidence(z, cutoff):
    """How sure, in words. The cutoff is the reader's level, so 'sure' is not a word I chose."""
    if z < cutoff:
        return "not at all"
    if z < 2 * cutoff:
        return "a little"
    if z < 5 * cutoff:
        return "fairly"
    return "very"


class Tutor:
    """A student that has been taught one lattice and will answer questions about it."""

    def __init__(self, grade, far=FAR):
        self.grade = grade
        self.far = far
        self.records = grade.records
        self.edges = grade.edges
        self.by_kind = grade.named_edges()
        self.last = None
        self._screen = None

    # ── what it has been taught ──────────────────────────────────────────────────────────
    def knows(self):
        n, e = len(self.records), len(self.edges)
        kinds = C.kinds(self.edges)
        resolved = [k for k in self.records
                    if engine.rank(self.records[k], far=self.far) > 0]
        head = (f"I have {n} things in front of me. I can read structure in {len(resolved)} of "
                f"them; the rest are noise as far as I can tell.")
        if not e:
            return (head + " Nothing connects to anything here, so there are no relationships "
                    "for me to know about -- ask me whether I can see a particular thing.")
        return (head + f" There are {e} connections between them, in {len(kinds)} sorts called "
                f"{self._english_list(kinds)}. A name is just a label someone wrote, though, and "
                f"whether those are really different relationships is a separate question -- ask "
                f"me 'what relationships do you know' and I will tell you which I can tell apart.")

    def kinds_report(self):
        if not self.edges:
            return ("None. This lesson has no connections in it at all -- it is only things, and "
                    "the question is whether I can see them, not how they relate.")
        s = self._kinds_screen()
        lines = []
        for name in s.names:
            if not s.present[name]:
                lines.append(f"  '{name}' -- I see no relationship here at all. The pairs called "
                             f"'{name}' are no more connected than any two things picked at random.")
            elif not s.distinct[name]:
                lines.append(f"  '{name}' -- there is a real relationship, but I cannot tell it "
                             f"apart from the others. It may be the same relationship under a "
                             f"different name.")
            else:
                lines.append(f"  '{name}' -- a real relationship, and one I can tell apart from "
                             f"the rest.")
        head = (f"I was given {len(s.names)} names. I can genuinely distinguish "
                f"{s.count} of them:")
        tail = ("" if s.count >= 2 else
                "\n\nWith fewer than two I cannot answer 'which relationship is this?' -- "
                "there would be nothing to choose between.")
        return head + "\n" + "\n".join(lines) + tail

    def confused_about(self):
        if not self.edges:
            return ("Nothing about relationships, because I have not been shown any. Ask me "
                    "whether I can see a particular thing instead.")
        s = self._kinds_screen()
        muddled = [n for n in s.names if s.present[n] and not s.distinct[n]]
        absent = [n for n in s.names if not s.present[n]]
        if not muddled and not absent:
            return "Nothing. Every relationship I was shown is real and distinct from the others."
        out = []
        if muddled:
            out.append(f"I cannot tell {self._english_list(muddled)} apart from the other "
                       f"relationships. They are real, but not distinct.")
        if absent:
            out.append(f"I see nothing behind {self._english_list(absent)} at all -- those pairs "
                       f"look no more related than any two things chosen at random. If someone "
                       f"told you that was a relationship, I disagree.")
        return " ".join(out)

    def can_you_see(self, name):
        """Is anything there? The nursery question, and the floor everything else stands on."""
        if name not in self.records:
            return self._unknown(name)
        W = np.asarray(self.records[name])
        modes = engine.rank(W, far=self.far)
        if modes == 0:
            return (f"No. I can read nothing in '{name}' -- whatever numbers are in it do not rise "
                    f"above their own noise. If you asked me to relate it to something I would "
                    f"have to decline, and I would rather say so than invent an answer.")
        return (f"Yes. '{name}' has {modes} distinct pattern{'s' if modes != 1 else ''} in it that "
                f"stand above its own noise, so there is something there for me to work with.")

    # ── questions about one thing ────────────────────────────────────────────────────────
    def about(self, name):
        if name not in self.records:
            return self._unknown(name)
        W = np.asarray(self.records[name])
        modes = engine.rank(W, far=self.far)
        out = [f"'{name}' is something I have {W.shape[0]} observations of, across "
               f"{W.shape[1]} channels."]
        if modes == 0:
            out.append("I can read no structure in it -- whatever is there does not stand above "
                       "its own noise, so I would not trust anything I said about it.")
        else:
            out.append(f"It has {modes} distinct pattern{'s' if modes != 1 else ''} in it that "
                       f"rise above its own noise.")
        links = [(d, k) for s, d, k in self.edges if s == name]
        back = [(s, k) for s, d, k in self.edges if d == name]
        if links:
            out.append("It connects out to " + self._english_list(
                [f"'{d}' ({k.removeprefix(C.KIND_PREFIX)})" for d, k in links]) + ".")
        if back:
            out.append("Things pointing at it: " + self._english_list(
                [f"'{s}' ({k.removeprefix(C.KIND_PREFIX)})" for s, k in back]) + ".")
        if not links and not back:
            out.append("Nothing connects to it, so I know it only by itself.")
        return " ".join(out)

    def related(self, a, b):
        for name in (a, b):
            if name not in self.records:
                return self._unknown(name)
        r = engine.associate(self.records[a], self.records[b], far=self.far)
        self.last = ("related", a, b, r)
        if not r.resolved:
            return (f"No. '{a}' and '{b}' are no more alike than I would expect from any two "
                    f"things here. Ask me 'why' for the numbers.")
        return (f"Yes -- and I am {_confidence(abs(r.z), r.cutoff)} sure. They share "
                f"{r.modes if hasattr(r, 'modes') else ''} structure that stands well above what "
                f"chance would give. Ask me 'why' for the numbers.").replace("  ", " ")

    def analogy(self, a, b, c, candidates=None):
        for name in (a, b, c):
            if name not in self.records:
                return self._unknown(name)
        if candidates:
            unknown = [k for k in candidates if k not in self.records]
            if unknown:
                return self._unknown(unknown[0])
            cands = list(candidates)
        else:
            cands = [k for k in sorted(self.records) if k not in (a, b, c)]
        if not cands:
            return "There is nothing left to choose from once those three are set aside."
        res = complete(self.records, exemplar=(a, b), source=c, candidates=cands, far=self.far)
        self.last = ("analogy", a, b, c, cands, res)
        if res.key is None:
            if res.why == "nothing resolved":
                return (f"I don't know. Whatever relationship holds between '{a}' and '{b}', I "
                        f"cannot find it between '{c}' and anything on offer. It may be that "
                        f"'{a}' and '{b}' are not really related at all -- ask me.")
            return (f"I don't know. Several candidates fit about equally well, and choosing "
                    f"between them would be guessing. Ask me 'why' to see how close they were.")
        pick = cands[res.key]
        z = res.scores[res.key].z
        return (f"'{pick}'. Whatever '{a}' is to '{b}', '{pick}' is that to '{c}' -- and I am "
                f"{_confidence(abs(z), res.scores[res.key].cutoff)} sure. I read the relationship "
                f"off that one example; nobody told me its name and I do not have one for it. "
                f"Ask me 'why'.")

    def why(self):
        if self.last is None:
            return "You have not asked me anything yet."
        kind = self.last[0]
        if kind == "related":
            _, a, b, r = self.last
            return (f"I compared '{a}' and '{b}' against the null of shuffling their rows "
                    f"together: z = {r.z:+.2f} against a cutoff of {r.cutoff:.2f}, and it "
                    f"{'did' if r.resolved else 'did not'} resolve.")
        _, a, b, c, cands, res = self.last
        rows = [f"  {'candidate':<20}{'alignment':>11}{'shared':>8}"]
        for name, sc, mo in zip(cands, res.scores, res.modes):
            rows.append(f"  {name:<20}{sc.z:>+11.2f}{mo:>8}")
        return ("I took the relationship between '%s' and '%s' as a matrix and asked which "
                "candidate stands in it to '%s'. Alignment proposes, breadth confirms -- I answer "
                "only when the best-aligned candidate is also the unique widest.\n\n" % (a, b, c)
                + "\n".join(rows)
                + f"\n\n  verdict: {res.why}")

    # ── plumbing ────────────────────────────────────────────────────────────────────────
    def _kinds_screen(self):
        if self._screen is None:
            self._screen = relations.carries_kinds(
                self.records, self.by_kind, far=self.far, seed=5, draws=300)
        return self._screen

    def _unknown(self, name):
        near = [k for k in self.records if k.startswith(str(name)[:2])][:6]
        hint = f" Did you mean {self._english_list(near)}?" if near else ""
        return f"I have never been shown anything called '{name}'.{hint}"

    @staticmethod
    def _english_list(items):
        items = [str(i) for i in items]
        if not items:
            return "nothing"
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]


TOPICS = """
Things worth asking me
======================

Can you see it at all?  (try these on 'nursery')
  can you see real0                    is there anything in it, or is it just noise
  can you see empty3                   I should say no -- it is noise, and I should know it

Is this the same as that?  (try these on 'pre-school')
  is thing0 related to copy0           the same thing seen twice; I should say yes
  is thing0 related to stranger0       I should say no

Which one goes with this?  (try these on 'kindergarten' and 'grade-one')
  s0 : partner0 :: s1 : ?              one relationship only, so the question is just
                                       which record is the partner
  as1 : ad1 :: as2 : ?                 grade one, where two relationships are in the room
                                       but each lives on its own things

What I have been taught
  what do you know                     a summary of everything in front of me
  what relationships do you know       which sorts of connection I can actually tell apart
  what are you confused about          where I cannot tell two relationships apart,
                                       or see nothing behind one at all

About one thing
  what is s1                           what I can read in it and what it connects to
  tell me about alpha2                 the same

Between two things
  is s1 related to alpha1              whether they are connected, and how sure I am
  are s0 and loose3 related            the same question, phrased the other way

The one that needs no names
  s0 is to alpha0 as s1 is to what     finish the analogy: I read the relationship off
                                       the example and apply it
  s0 : alpha0 :: s1 : ?                the same, written short
  s0 : alpha0 :: s1 : ?, from alpha1 beta1   short form, choices restricted
  ...as s1 is to what, from alpha1 beta1     restrict the choices

After any answer
  why                                  the numbers behind what I just said

The lessons you can talk to
  nursery pre-school kindergarten grade-one grade-two grade-three
  grade-four grade-five grade-six grade-seven near-duplicates

    python research/lib/tutor.py nursery

Good questions to try
  - Ask me an analogy using a relationship that is NOT real (grade three has one called
    'hollow' whose pairs are unrelated). I should refuse rather than invent an answer,
    and measured over 116 such questions I declined 97 of them.
  - Ask me about a thing with no structure ('empty3' on nursery). I should tell you
    I cannot read it, rather than making something up.
  - Ask the same analogy with and without restricting the candidates. Unrestricted is
    much harder and I will often decline.
  - Ask 'what are you confused about' on near-duplicates, where two relationships are
    nearly the same map. I should say so rather than pretend to separate them.
  - Ask 'what relationships do you know' on grade-three. I should name three, not four:
    'hollow' is a label on the lattice with nothing behind it and I should not count it.

  quit                                 leave
"""


def _parse(line, tutor):
    q = line.strip().rstrip("?").strip()
    low = q.lower()
    if low in ("topics", "help", "?"):
        return TOPICS
    if low in ("why", "why?", "explain", "how do you know"):
        return tutor.why()
    m = re.match(r"(?:can you see|is there anything (?:in|there in)|do you see) (\S+)", low)
    if m:
        return tutor.can_you_see(m.group(1))
    if "confused" in low or "cannot tell apart" in low or "can't you tell" in low:
        return tutor.confused_about()
    if "relationship" in low and ("know" in low or "kinds" in low or "sorts" in low):
        return tutor.kinds_report()
    if low.startswith("what do you know") or low.startswith("what have you learned"):
        return tutor.knows()

    # Every phrasing of the analogy takes the same optional "from a b c" tail, which restricts
    # the candidates. The short `::` form takes it too; without it the guide would advertise a
    # restriction that the natural way of writing it silently drops, answering the much harder
    # open question instead.
    m = re.match(r"(.+?)\s*:\s*(.+?)\s*::\s*(.+?)\s*:\s*\?*"
                 r"(?:\s*,?\s*from\s+(.+))?$", q, re.IGNORECASE)
    if m:
        cands = m.group(4).replace(",", " ").split() if m.group(4) else None
        return tutor.analogy(m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), cands)
    m = re.match(r"(?:what )?is (\S+) to (\S+) as (\S+) is to (?:what|which)"
                 r"(?:\s*,?\s*from\s+(.+))?", q, re.IGNORECASE)
    if m:
        cands = m.group(4).replace(",", " ").split() if m.group(4) else None
        return tutor.analogy(m.group(1), m.group(2), m.group(3), cands)
    m = re.match(r"(\S+) is to (\S+) as (\S+) is to (?:what|which)(?:,? from (.+))?", q,
                 re.IGNORECASE)
    if m:
        cands = m.group(4).replace(",", " ").split() if m.group(4) else None
        return tutor.analogy(m.group(1), m.group(2), m.group(3), cands)
    m = re.match(r"(?:is|are) (\S+) (?:related to|and) (\S+)(?: related)?", q, re.IGNORECASE)
    if m:
        return tutor.related(m.group(1), m.group(2))
    m = re.match(r"(?:what is|tell me about|describe) (\S+)", q, re.IGNORECASE)
    if m:
        return tutor.about(m.group(1))
    return ("I did not understand that. Say 'topics' to see what I can be asked.")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    name = " ".join(argv) if argv else "grade two"
    # every lesson answers to both spellings of its name, including the sweep, which the topic
    # guide lists
    lessons = C.CURRICULUM + (C.near_duplicates,)
    makers = {m.__name__.replace("_", "-"): m for m in lessons}
    makers.update({m.__name__.replace("_", " "): m for m in lessons})
    if name not in makers:
        print(f"I do not have a lesson called '{name}'. I have: "
              + ", ".join(m.__name__.replace("_", "-") for m in lessons))
        return 1
    grade = makers[name](rng=np.random.default_rng(5))
    tutor = Tutor(grade)
    print(f"\n-- I have been taught '{grade.name}' ({grade.note}).")
    print("-- Ask me something, or say 'topics' for ideas. 'quit' to leave.\n")
    while True:
        try:
            line = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in ("quit", "exit", "bye"):
            return 0
        print("\n" + _parse(line, tutor) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
