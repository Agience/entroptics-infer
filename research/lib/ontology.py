"""The ontology as vertices and weighted edges. Nothing is a type, and nothing is enumerated.

The shape, and why it is this shape
-------------------------------------
**Every concept is a vertex, and a relation is a concept.** ``is_a`` sits in the store beside ``dog``
and ``river``, because it is a thing the ontology knows about rather than a tag on a connection. So
there is no closed set of relation names anywhere in the structure: a new relation is a new vertex,
exactly as a new animal is.

**A statement is a vertex too.** ``dog is_a mammal`` is not an edge; it is a vertex joined to
``dog``, to ``is_a`` and to ``mammal``. It fires when all three are active, which is what a neuron
does, and that coincidence is what binds a statement together. Without it, reifying ``is_a`` onto one
shared vertex would make ``dog -> is_a -> ocean`` a walkable path the ontology never asserted.

**An edge carries reach and position, and no type.** A dendrite has a strength and a place on the
tree; it does not have a name. The position is what tells the subject from the object — ``mammal is_a
dog`` is a different statement because its parts arrive in a different order, not because an edge is
labelled differently. Weight is never written here.

Why an edge has no label
--------------------------
An edge shaped ``(src, label, dst)``, with the label drawn from six strings written in this file,
would put a closed vocabulary into the *shape* of the data, and everything downstream would inherit
it: the coordinate basis would be ``label x direction x concept``, 720 axes, every one of them
existing because six names were written down. Measured against six such names, the edges occupy
more independent directions than there are names -- so a name is a coarse label for a region, and
the structure asserts a distinction the edges do not carry.

That count is not re-derivable in this repository: measuring it needs a `labels()` this store
does not have, so no figure is quoted for it here. The same question asked of this store is
`research/checks/test_how_many_kinds_the_store_still_carries.py` (CLAIMS §47.118), and its answer
is the opposite one — these relations are all but independent as operators, which is what a
structure with no names left to be coarse about should look like.
"""
from __future__ import annotations

import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2] / "src")]

from entroptics_infer.wrapper import Engine                     # noqa: E402

#: Statements, each a tuple of concepts in order. Read them left to right: "a dog is_a mammal".
#: The order is the roles -- nothing here says which position means what, because nothing needs to.
STATEMENTS: tuple[tuple[str, ...], ...] = (
    # --- living things -------------------------------------------------------------
    ("dog", "is_a", "mammal"),
    ("cat", "is_a", "mammal"),
    ("horse", "is_a", "mammal"),
    ("whale", "is_a", "mammal"),
    ("sparrow", "is_a", "bird"),
    ("eagle", "is_a", "bird"),
    ("penguin", "is_a", "bird"),
    ("salmon", "is_a", "fish"),
    ("shark", "is_a", "fish"),
    ("mammal", "is_a", "animal"),
    ("bird", "is_a", "animal"),
    ("fish", "is_a", "animal"),
    ("animal", "is_a", "organism"),
    ("oak", "is_a", "tree"),
    ("pine", "is_a", "tree"),
    ("tree", "is_a", "plant"),
    ("grass", "is_a", "plant"),
    ("plant", "is_a", "organism"),

    ("tail", "part_of", "dog"),
    ("tail", "part_of", "cat"),
    ("paw", "part_of", "dog"),
    ("paw", "part_of", "cat"),
    ("wing", "part_of", "bird"),
    ("feather", "part_of", "wing"),
    ("beak", "part_of", "bird"),
    ("fin", "part_of", "fish"),
    ("gill", "part_of", "fish"),
    ("scale", "part_of", "fish"),
    ("leaf", "part_of", "tree"),
    ("branch", "part_of", "tree"),
    ("root", "part_of", "plant"),
    ("trunk", "part_of", "tree"),

    ("dog", "lives_in", "house"),
    ("cat", "lives_in", "house"),
    ("horse", "lives_in", "stable"),
    ("whale", "lives_in", "ocean"),
    ("salmon", "lives_in", "river"),
    ("shark", "lives_in", "ocean"),
    ("eagle", "lives_in", "mountain"),
    ("penguin", "lives_in", "ice"),
    ("oak", "lives_in", "forest"),
    ("pine", "lives_in", "forest"),

    # --- made things ---------------------------------------------------------------
    ("car", "is_a", "vehicle"),
    ("lorry", "is_a", "vehicle"),
    ("bicycle", "is_a", "vehicle"),
    ("boat", "is_a", "vehicle"),
    ("vehicle", "is_a", "artefact"),
    ("hammer", "is_a", "tool"),
    ("saw", "is_a", "tool"),
    ("chisel", "is_a", "tool"),
    ("tool", "is_a", "artefact"),
    ("house", "is_a", "building"),
    ("stable", "is_a", "building"),
    ("building", "is_a", "artefact"),

    ("wheel", "part_of", "car"),
    ("wheel", "part_of", "lorry"),
    ("wheel", "part_of", "bicycle"),
    ("engine", "part_of", "car"),
    ("engine", "part_of", "lorry"),
    ("engine", "part_of", "boat"),
    ("sail", "part_of", "boat"),
    ("handle", "part_of", "hammer"),
    ("handle", "part_of", "saw"),
    ("blade", "part_of", "saw"),
    ("blade", "part_of", "chisel"),
    ("roof", "part_of", "house"),
    ("door", "part_of", "house"),
    ("door", "part_of", "stable"),
    ("wall", "part_of", "building"),

    ("hammer", "used_for", "building"),
    ("saw", "used_for", "cutting"),
    ("chisel", "used_for", "cutting"),
    ("car", "used_for", "travel"),
    ("boat", "used_for", "travel"),
    ("bicycle", "used_for", "travel"),
    ("horse", "used_for", "travel"),

    # --- where things are ------------------------------------------------------------
    ("forest", "is_a", "place"),
    ("ocean", "is_a", "place"),
    ("river", "is_a", "place"),
    ("mountain", "is_a", "place"),
    ("ice", "is_a", "place"),
    ("river", "flows_into", "ocean"),
    ("tree", "grows_in", "forest"),
    ("grass", "grows_in", "forest"),
)

#: What each concept means, in one line. Relations are concepts and are glossed like any other.
#: Used only to write an answer back and to land a query; no read infers anything from the wording.
GLOSS: dict[str, str] = {
    "dog": "a domesticated carnivore kept as a pet or working animal",
    "cat": "a small domesticated carnivore kept as a pet",
    "horse": "a large hoofed mammal ridden or used to pull loads",
    "whale": "a very large marine mammal",
    "sparrow": "a small common perching bird",
    "eagle": "a large bird of prey",
    "penguin": "a flightless seabird of cold southern waters",
    "salmon": "a fish that migrates between river and sea",
    "shark": "a large predatory cartilaginous fish",
    "mammal": "a warm-blooded animal that feeds its young on milk",
    "bird": "a warm-blooded egg-laying animal with feathers and wings",
    "fish": "a cold-blooded animal that lives in water and breathes through gills",
    "animal": "a living thing that moves of its own accord",
    "organism": "any individual living thing",
    "oak": "a large hardwood tree bearing acorns",
    "pine": "an evergreen tree with needles and cones",
    "tree": "a tall plant with a woody trunk",
    "grass": "a low green plant with narrow leaves",
    "plant": "a living thing that grows in one place and makes its own food",
    "tail": "the part extending from the rear of an animal",
    "paw": "the foot of an animal with claws",
    "wing": "the limb a bird flies with",
    "feather": "one of the light structures covering a bird",
    "beak": "the hard projecting mouth of a bird",
    "fin": "the flat limb a fish steers and swims with",
    "gill": "the organ a fish breathes water through",
    "scale": "one of the small plates covering a fish",
    "leaf": "the flat green part a plant gathers light with",
    "branch": "a limb growing from the trunk of a tree",
    "root": "the part of a plant that holds it and draws water",
    "trunk": "the main woody stem of a tree",
    "house": "a building people live in",
    "stable": "a building horses are kept in",
    "building": "a made structure with walls and a roof",
    "car": "a road vehicle carrying a few people",
    "lorry": "a large road vehicle carrying goods",
    "bicycle": "a two-wheeled vehicle driven by pedals",
    "boat": "a vessel that travels on water",
    "vehicle": "a made thing used to carry people or goods",
    "artefact": "a thing made by people",
    "hammer": "a tool for driving nails",
    "saw": "a tool with a toothed blade for cutting",
    "chisel": "a tool with a sharp blade for shaping",
    "tool": "a made thing used to do work by hand",
    "wheel": "a round part that turns to let a thing roll",
    "engine": "the part that drives a vehicle",
    "sail": "the sheet a boat catches wind with",
    "handle": "the part of a tool that is held",
    "blade": "the cutting part of a tool",
    "roof": "the covering over the top of a building",
    "door": "the panel closing an opening in a building",
    "wall": "an upright side of a building",
    "forest": "a place thick with trees",
    "ocean": "a very large body of salt water",
    "river": "a large flow of fresh water across land",
    "mountain": "a high steep landform",
    "ice": "frozen water",
    "place": "somewhere a thing can be",
    "travel": "going from one place to another",
    "cutting": "dividing a thing with a blade",
    "is_a": "the broader kind a thing belongs to",
    "part_of": "the whole a thing is a piece of",
    "lives_in": "where a living thing makes its home",
    "used_for": "what a made thing is for",
    "grows_in": "where a plant takes root",
    "flows_into": "where water runs to",
}


def statement_id(i: int) -> str:
    """A statement's name. It is a vertex like any other and needs one.

    The shipped engine's own name for it, so a research read and a shipped read name the same
    vertices.
    """
    return Engine.statement_id(i)


#: every statement's name, to its position in the store
_STATEMENT_INDEX = {statement_id(i): i for i in range(len(STATEMENTS))}


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


def render(parts) -> str:
    """A statement as its own content: "a dog is a mammal".

    **The statement is the larger concept and its parts are its decomposition.** Both exist, both
    are vertices, and both are reachable -- which of them a query uses is settled by the walk at the
    time, not by one of them being the real one.

    So the whole needs content of its own, or it is only a binding node: a query could never land on
    it, and the colimit would run one way. This renders it from the parts, with the relation read as
    the verb and its underscores opened out. Nothing here knows any relation by name -- a relation
    ending in its own article ("is a") does not get a second one, checked against the words English
    uses as articles rather than against a list of relations.
    """
    try:
        src, verb, dst = parts
    except ValueError:
        # not a triple, so there is no verb to put between two ends; the guard is the unpacking
        # itself rather than a length written beside it, which cannot then drift from it
        return " ".join(parts)
    verb = verb.replace("_", " ")
    lead = "" if verb.split()[-1] in ("a", "an", "the") else _article(dst) + " "
    return f"{_article(src)} {src} {verb} {lead}{dst}"


def content(vertex: str) -> str:
    """What a vertex holds. Concepts hold their gloss; a statement holds itself, written out."""
    if vertex in GLOSS:
        return GLOSS[vertex]
    if vertex in _STATEMENT_INDEX:
        return render(parts_of(vertex))
    return ""


def concepts() -> tuple[str, ...]:
    """Every concept vertex, in a fixed order. Relations are in here; statements are not."""
    seen = []
    for parts in STATEMENTS:
        for p in parts:
            if p not in seen:
                seen.append(p)
    return tuple(sorted(seen))


def statements() -> tuple[str, ...]:
    """Every statement vertex, in a fixed order."""
    return tuple(statement_id(i) for i in range(len(STATEMENTS)))


def vertices() -> tuple[str, ...]:
    """Everything the store holds: concepts and statements together, one kind of thing."""
    return concepts() + statements()


def edges() -> tuple[tuple[str, str, int], ...]:
    """``(statement, concept, position)`` — untyped, carrying only reach and where it arrives.

    Position is what tells a subject from an object, and it is a place rather than a name: the
    ontology never says "position 1 means the relation", and no read here branches on the number.
    """
    out = []
    for i, parts in enumerate(STATEMENTS):
        sid = statement_id(i)
        for position, concept in enumerate(parts):
            out.append((sid, concept, position))
    return tuple(out)


def parts_of(sid: str) -> tuple[str, ...]:
    """The concepts a statement joins, in the order they arrive."""
    return STATEMENTS[_STATEMENT_INDEX[sid]]


def summary() -> str:
    c, s, e = concepts(), statements(), edges()
    widest = max(len(p) for p in STATEMENTS)
    return (f"{len(c)} concepts + {len(s)} statements = {len(c) + len(s)} vertices, "
            f"{len(e)} edges, no types; statements join up to {widest} concepts each")


if __name__ == "__main__":
    print(summary())
    missing = sorted(set(concepts()) - set(GLOSS))
    print(f"no gloss for: {', '.join(missing)}" if missing else "every concept is glossed")
    relations = sorted({p[1] for p in STATEMENTS})
    print(f"\nconcepts that happen to be used as relations: {', '.join(relations)}")
    print("  -- they are vertices like any other; nothing in the code knows this list exists")
