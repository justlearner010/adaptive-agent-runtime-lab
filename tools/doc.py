"""Doc tool: deterministic local long-document store with paged reads (v2).

Gives the benchmark a task family that genuinely needs decomposition:
documents are long enough that any agent must read them in several pages,
and a subagent strategy can parallelize page reads across workers.

Content is synthetic and invented so the model cannot answer from world
knowledge — the only source of facts is the doc tool itself, and every
fact carries a distinctive keyword for deterministic checking.
"""

from __future__ import annotations

import random

# doc id -> {"topic": str, "facts": [(keyword, sentence), ...]}
_DOC_FACTS: dict[str, dict] = {
    "deltas": {
        "topic": "mining stations in the Delta sector",
        "facts": [
            ("aria-7", "Station Aria-7 has an ore purity of 92.4 percent"),
            ("bruma-3", "Station Bruma-3 runs on a 14-hour shift cycle"),
            ("celso-9", "Station Celso-9 produces 3,200 tons of nickel per cycle"),
            ("drava-2", "Station Drava-2 is powered by a 6.1 terawatt reactor"),
            ("eira-5", "Station Eira-5 employs a crew of 41"),
            ("fenno-8", "Station Fenno-8 extracts ore from the Kessar trench"),
            ("golde-1", "Station Golde-1 has a failure rate of 0.7 percent"),
            ("halo-6", "Station Halo-6 was commissioned in year 2187"),
        ],
    },
    "nexus": {
        "topic": "algorithms in the Nexus library",
        "facts": [
            ("quell", "The quell algorithm sorts in O(n log n) time"),
            ("vanish", "The vanish algorithm compresses logs by 38 percent"),
            ("wist", "The wist algorithm routes packets with 1.2 ms latency"),
            ("xenon", "The xenon algorithm caches 64 keys per node"),
            ("yarrow", "The yarrow algorithm hashes in 0.03 ms"),
            ("zephyr", "The zephyr algorithm balances load every 5 seconds"),
            ("amber", "The amber algorithm encrypts with a 256-bit key"),
            ("cobalt", "The cobalt algorithm deduplicates with 99.1 percent accuracy"),
        ],
    },
    "haven": {
        "topic": "research stations in the Haven archipelago",
        "facts": [
            ("tide-1", "Tide-1 studies coral growth at 30 meters depth"),
            ("spray-4", "Spray-4 monitors salinity every 6 hours"),
            ("kelp-7", "Kelp-7 houses a crew of 12 biologists"),
            ("reef-2", "Reef-2 tracks 58 tagged sea turtles"),
            ("marsh-9", "Marsh-9 measures tides with 0.4 meter precision"),
            ("delta-3", "Delta-3 runs a 9-month plankton survey"),
            ("lago-6", "Lago-6 stores samples at minus 80 degrees"),
            ("cove-8", "Cove-8 studies jellyfish migration patterns"),
        ],
    },
    "meridian": {
        "topic": "observatories in the Meridian cluster",
        "facts": [
            ("pico-1", "Observatory Pico-1 tracks 23 near-earth asteroids"),
            ("luna-2", "Observatory Luna-2 detects flares at 480 nanometers"),
            ("vesta-4", "Observatory Vesta-4 maps craters at 3.1 meters resolution"),
            ("cosmo-7", "Observatory Cosmo-7 logs radio bursts every 4 seconds"),
            ("solar-3", "Observatory Solar-3 predicts storms 2 days ahead"),
            ("orbita-9", "Observatory Orbita-9 measures solar wind at 5.2 AU"),
            ("quasar-5", "Observatory Quasar-5 archives 18 terabytes per day"),
            ("nebula-6", "Observatory Nebula-6 watches 41 variable stars"),
        ],
    },
}

_PAGE_CAP = 2500  # max chars per read; forces paging

_FILLER_WORDS = (
    "basalt ridge thermal vent silicate deposit crystalline shale carbon field "
    "ion lattice magnetite ore strata pressure gradient seismic echo drill core "
    "trace element flux density survey grid sample bay holding tank conduit "
    "porosity index corrosion profile elevation datum moisture band resonance "
    "anchor bolt calibration log telemetry feed buffer tank wall liner"
).split()


def _filler(rng: random.Random) -> str:
    n = rng.randint(10, 16)
    words = [rng.choice(_FILLER_WORDS) for _ in range(n)]
    return "The " + " ".join(words) + " was recorded and archived."


def _build_docs() -> dict[str, str]:
    """Deterministic documents: fact sentences padded with filler (~10KB)."""
    rng = random.Random(42)
    docs: dict[str, str] = {}
    for doc_id, meta in _DOC_FACTS.items():
        parts = [f"# {doc_id.upper()} — {meta['topic']}"]
        for _keyword, sentence in meta["facts"]:
            parts.append(sentence + ".")
            for _ in range(12):
                parts.append(_filler(rng))
        docs[doc_id] = "\n\n".join(parts)
    return docs


DOCS: dict[str, str] = _build_docs()


def list_docs() -> list[str]:
    return sorted(DOCS)


def read_doc(doc_id: str, start: int, length: int) -> str:
    """Return a page of `doc_id` starting at char offset `start`."""
    if doc_id not in DOCS:
        raise KeyError(doc_id)
    if start < 0:
        start = 0
    return DOCS[doc_id][start : start + min(length, _PAGE_CAP)]


class DocTool:
    name = "doc"
    description = (
        "Read pages from a local knowledge document. "
        "input: {'action': 'list'} to see available documents, or "
        "{'action': 'read', 'doc': '<id>', 'start': <char offset>, 'length': <chars>} "
        "to read a page (max 2500 chars per read). Documents are long — "
        "read them page by page until you reach '(end of <id>)'."
    )

    def run(self, args: dict) -> str:
        action = args.get("action")
        if action == "list":
            return "\n".join(
                f"- {doc_id}: {_DOC_FACTS[doc_id]['topic']}" for doc_id in list_docs()
            )
        if action == "read":
            doc = args.get("doc", "")
            if doc not in DOCS:
                return f"Error: unknown doc {doc!r}. Available: {', '.join(list_docs())}"
            try:
                start = int(args.get("start", 0))
            except (TypeError, ValueError):
                return "Error: start must be an integer"
            try:
                length = int(args.get("length", _PAGE_CAP))
            except (TypeError, ValueError):
                return "Error: length must be an integer"
            text = read_doc(doc, start, length)
            if not text:
                return f"(end of {doc})"
            return f"[{doc} @{start}+{len(text)}]\n{text}"
        return "Error: action must be 'list' or 'read'"
