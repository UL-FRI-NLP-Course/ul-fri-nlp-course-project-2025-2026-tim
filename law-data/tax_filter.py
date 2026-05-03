"""
Filter Slovenian legal JSONL rows for tax / fiscal / FURS-related RAG corpora.

Per-source rules match the law-data layout (PISRS, UradniList, USRS, SodnaPraksa).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

__all__ = [
    "SOURCE_KINDS",
    "detect_source_kind",
    "is_tax_relevant",
    "iter_tax_records",
    "text_head",
    "join_search_blob",
]

# --- Regexes (verbose, case-insensitive) ------------------------------------

_FLAGS = re.IGNORECASE | re.VERBOSE

# High-precision title / heading matches (naziv, title, act prefixes).
ALLOWLIST_SUBSTRINGS: tuple[str, ...] = (
    "Zakon o davku na dodano vrednost",
    "Zakon o dohodnini",
    "Zakon o davčnem postopku",
    "Zakon o finančni upravi",
    "Zakon o dohodku pravnih oseb",
    "Zakon o trošarinah",
    "Zakon o carinskem postopku",
    "Zakon o carini",
    "Zakon o upravnih taksah",
    "Zakon o prispevkih",
    "Zakon o izvrševanju proračuna",
    "Zakon o javnih financah",
    "Zakon o javnofinančnih",
    "Zakon o financiranju občin",
    "Zakon o začasnih ukrepih za preprečevanje pranja denarja",
    "Zakon o preprečevanju pranja denarja",
    "Zakon o preprečevanju financiranja terorizma",
    "Zakon o plačilnih storitvah in sistemih",
    "Pravilnik o dav",
    "Navodil o dav",
    "Odredba o dav",
    "Uredba o dav",
    "Sklep o dav",
)

# Drop broad “money” hits that are not tax/fiscal law.
EXCLUDE_TITLE_RE = re.compile(
    r"""
    (?ix)
    prispevek.*polit(ič|ic) |
    financiranj\w*\s+(vrtc|šol|sol|zavod|arhiv|kulturn|športn|sportn) |
    osebnem\s+imenu |
    zaščit\w*\s+žival|zascit\w*\s+zival
    """,
    _FLAGS,
)

# Primary filter for PISRS naziv / UL title / similar.
TIGHT_TAX_TITLE_RE = re.compile(
    r"""
    (?ix)
    \b(
      Zakon\s+o\s+dav |
      \bdavek\b | \bdavčn | \bdavcn |
      \bdohodn(in|a|i)\b | \bdohodek\b |
      \bDDV\b | dodano\s+vrednost |
      trošarin | trosarin |
      carinsk | \bcarina\b |
      ZDavP | ZDDV | ZDoh | ZDDPO | ZFU\b |
      finančn\w*\s+uprav | financn\w*\s+uprav | \bFURS\b |
      upravn\w*\s+taks |
      prispevk\w*\s+za\s+(socialno|zdravstveno|pokojninsk) |
      pranje\s+denarja | terorizma |
      plačiln\w*\s+storitv |
      javn\w*\s+financ | izvrševanj\w*\s+proračun |
      financiranj\w*\s+občin
    )
    """,
    _FLAGS,
)

# Finančna uprava / procedural anchors (text head, courts, USRS).
FURS_AND_PROC_RE = re.compile(
    r"""
    (?ix)
    \bFURS\b |
    Finančn\w*\s+uprav\w* | Financn\w*\s+uprav\w* |
    davčn\w*\s+zavez | davcn\w*\s+zavez |
    davčn\w*\s+inšpekt | davcn\w*\s+inspekt |
    Zakon\s+o\s+dav(č|c) |
    \bZDavP\b | \bZDDV\b | \bZDoh\b
    """,
    _FLAGS,
)

# Ustavno sodišče blob.
USRS_TAX_RE = re.compile(
    r"""
    (?ix)
    Zakon\s+o\s+dav(č|c) |
    \bZDavP\b | \bZDDV\b | \bZDoh\b | \bZDDPO\b | \bZFU\b |
    dohodnina | davčn\w*\s+postop |
    Finančn\w*\s+uprav\w* | \bFURS\b
    """,
    _FLAGS,
)

# Sodna praksa — combined metadata + core fields.
COURTS_TAX_RE = re.compile(
    r"""
    (?ix)
    \bFURS\b |
    Finančn\w*\s+uprav\w* | Financn\w*\s+uprav\w* |
    \bdavčn | \bdavcn |
    \bdohodn(in|a)\b | \bDDV\b |
    trošarin | trosarin | carinsk |
    \bZDavP\b | \bZDDV\b | \bZDoh\b
    """,
    _FLAGS,
)

# ul-razglasni — narrow.
UL_RAZGLASNI_RE = re.compile(
    r"""(?ix)\b(FURS|davčn|davcn|\bdavek\b|dohodnina|\bDDV\b|Finančn\w*\s+uprav\w*)""",
    _FLAGS,
)

SOURCE_KINDS = frozenset(
    {
        "pisrs_register",
        "pisrs_druga",
        "pisrs_splosni_javna_pooblastila",
        "pisrs_neveljavni",
        "pisrs_obsoletni",
        "pisrs_predpisi_priprava",
        "pisrs_evidenca_normodajalcev",
        "ul_uredbeni",
        "ul_razglasni",
        "usrs",
        "sp_courts",
        "sp_claims",
        "unknown",
    }
)


def text_head(s: str | None, n: int) -> str:
    if not s:
        return ""
    return s[:n]


def _lower(s: str) -> str:
    return s.casefold()


def allowlist_match(title: str) -> bool:
    t = _lower(title)
    return any(_lower(a) in t for a in ALLOWLIST_SUBSTRINGS)


def tight_title_match(title: str) -> bool:
    if not title or EXCLUDE_TITLE_RE.search(title):
        return False
    if allowlist_match(title):
        return True
    return bool(TIGHT_TAX_TITLE_RE.search(title))


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [_stringify(x) for x in value]
        return " ".join(p for p in parts if p)
    if isinstance(value, Mapping):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def join_search_blob(*parts: Any) -> str:
    return " ".join(
        p.strip() for p in (_stringify(x) for x in parts) if p and p.strip()
    )


def detect_source_kind(path: str | Path) -> str:
    name = Path(path).name.lower()
    mapping = {
        "register-predpisov.jsonl": "pisrs_register",
        "drugi-splosni-in-posamicni-akti.jsonl": "pisrs_druga",
        "splosni-akti-za-izvrsevanje-javnih-pooblastil.jsonl": "pisrs_splosni_javna_pooblastila",
        "neveljavni-predpisi.jsonl": "pisrs_neveljavni",
        "obsoletni-in-konzumirani-predpisi.jsonl": "pisrs_obsoletni",
        "predpisi-v-pripravi.jsonl": "pisrs_predpisi_priprava",
        "evidenca-normodajalcev.jsonl": "pisrs_evidenca_normodajalcev",
        "ul-uredbeni.jsonl": "ul_uredbeni",
        "ul-razglasni.jsonl": "ul_razglasni",
        "usrs.jsonl": "usrs",
        "sp_courts.jsonl": "sp_courts",
        "sp_claims.jsonl": "sp_claims",
    }
    return mapping.get(name, "unknown")


def is_tax_relevant(record: Mapping[str, Any], kind: str) -> bool:
    """
    Return True if the JSON object should be indexed for tax/fiscal RAG.

    Parameters
    ----------
    record :
        One decoded JSON object from a law-data .jsonl line.
    kind :
        Source discriminator; use detect_source_kind(path) or a value from SOURCE_KINDS.
    """
    if kind == "sp_claims":
        return False

    if kind == "pisrs_register":
        naziv = _stringify(record.get("naziv"))
        if tight_title_match(naziv):
            return True
        return bool(
            FURS_AND_PROC_RE.search(text_head(_stringify(record.get("text")), 4000))
        )

    if kind in (
        "pisrs_druga",
        "pisrs_neveljavni",
        "pisrs_obsoletni",
        "pisrs_predpisi_priprava",
    ):
        return tight_title_match(_stringify(record.get("naziv")))

    if kind == "pisrs_evidenca_normodajalcev":
        # Avoid broad "proračun" hits in body-only matches.
        return tight_title_match(_stringify(record.get("naziv")))

    if kind == "pisrs_splosni_javna_pooblastila":
        naziv = _stringify(record.get("naziv"))
        if tight_title_match(naziv):
            return True
        head = text_head(_stringify(record.get("text")), 6000)
        return bool(FURS_AND_PROC_RE.search(head))

    if kind == "ul_uredbeni":
        title = _stringify(record.get("title"))
        if tight_title_match(title):
            return True
        head = text_head(_stringify(record.get("text")), 3000)
        return bool(FURS_AND_PROC_RE.search(head))

    if kind == "ul_razglasni":
        title = _stringify(record.get("title"))
        blob = join_search_blob(title, text_head(_stringify(record.get("text")), 1500))
        return bool(UL_RAZGLASNI_RE.search(blob))

    if kind == "usrs":
        blob = join_search_blob(
            record.get("act"),
            record.get("abstract"),
            record.get("legalBasis"),
            record.get("operationalProvisions"),
        )
        return bool(USRS_TAX_RE.search(blob))

    if kind == "sp_courts":
        blob = join_search_blob(
            record.get("metadata"),
            record.get("jedro"),
            record.get("izrek"),
            text_head(_stringify(record.get("obrazlozitev")), 8000),
        )
        return bool(COURTS_TAX_RE.search(blob))

    if kind == "unknown":
        # Best-effort: same fields as mixed PISRS/UL.
        for key in ("naziv", "title", "act"):
            v = _stringify(record.get(key))
            if v and tight_title_match(v):
                return True
        return bool(
            FURS_AND_PROC_RE.search(
                join_search_blob(
                    record.get("text"),
                    record.get("abstract"),
                    text_head(_stringify(record.get("obrazlozitev")), 4000),
                )
            )
        )

    return False


def iter_tax_records(
    path: str | Path,
    *,
    kind: str | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Stream a JSONL file and yield only tax-relevant records (decoded dicts).
    """
    p = Path(path)
    k = kind if kind is not None else detect_source_kind(p)
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if is_tax_relevant(obj, k):
                yield obj


def filter_jsonl_file(
    src: str | Path,
    dst: str | Path,
    *,
    kind: str | None = None,
) -> tuple[int, int]:
    """
    Write tax-relevant lines from src JSONL to dst. Returns (kept, total).
    """
    kept = total = 0
    k = kind if kind is not None else detect_source_kind(src)
    with Path(src).open(encoding="utf-8") as inf, Path(dst).open(
        "w", encoding="utf-8"
    ) as outf:
        for line in inf:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and is_tax_relevant(obj, k):
                outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    return kept, total


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Filter law-data JSONL for tax/fiscal rows."
    )
    ap.add_argument("input", type=Path, help="Input .jsonl path")
    ap.add_argument(
        "-o", "--output", type=Path, help="Output .jsonl (default: stdout names only)"
    )
    ap.add_argument(
        "-k",
        "--kind",
        choices=sorted(SOURCE_KINDS - {"unknown"}),
        help="Force source kind (default: infer from filename)",
    )
    ap.add_argument("--count", action="store_true", help="Print kept/total and exit")
    args = ap.parse_args()
    kind = args.kind or detect_source_kind(args.input)
    if args.count:
        kept, total = 0, 0
        with args.input.open(encoding="utf-8") as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and is_tax_relevant(obj, kind):
                    kept += 1
        print(f"{args.input.name}: kind={kind} kept={kept} total={total}")
    elif args.output:
        k, t = filter_jsonl_file(args.input, args.output, kind=args.kind)
        print(f"Wrote {k} / {t} lines to {args.output}")
    else:
        for rec in iter_tax_records(args.input, kind=args.kind):
            print(
                rec.get("naziv")
                or rec.get("title")
                or rec.get("act")
                or rec.get("id", "")
            )
