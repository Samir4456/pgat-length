"""Semantic schema for PHOENIX14T compositional-generalization study.

Slot and relation vocabularies + closed-domain German weather lexicons.
Kept as a single flat module so every lexicon is inspectable by a
reviewer without diving through the extractor.

Design principles:
- Every surface form maps to a canonical atom via a lower-cased key.
- Compound handling via longest-prefix match (order lexicons long-first
  in extractor).
- Ambiguous forms (nördlich = north / northerly wind) are handled by
  the extractor with context, not the schema itself.
- Temperature is discretized into 8 buckets so that "10 grad" and
  "12 grad" collapse to one atom for compound-frequency counting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
# Slot and relation types.
# ---------------------------------------------------------------------------

class Slot(str, Enum):
    EVENT = "event"
    LOCATION = "location"
    TIME = "time"
    TEMPERATURE = "temperature"
    INTENSITY = "intensity"
    WIND_DIRECTION = "wind_direction"
    POLARITY = "polarity"


class Relation(str, Enum):
    EVENT_LOCATION = "event_location"
    TEMPERATURE_LOCATION = "temperature_location"
    EVENT_TIME = "event_time"
    NEGATION_SCOPE = "negation_scope"
    WIND_DIRECTION_LOCATION = "wind_direction_location"


# ---------------------------------------------------------------------------
# Temperature buckets. Each entry: (lower_inclusive, upper_exclusive, label).
# The last bucket has upper=None => open-ended >= lower.
# ---------------------------------------------------------------------------

TEMPERATURE_BUCKETS: Final[list[tuple[int, int | None, str]]] = [
    (-999, -5, "lt_-5"),
    (-5, 0, "-5..0"),
    (0, 5, "0..5"),
    (5, 10, "5..10"),
    (10, 15, "10..15"),
    (15, 20, "15..20"),
    (20, 25, "20..25"),
    (25, None, "gte_25"),
]


def temperature_bucket(value: int) -> str:
    for lower, upper, label in TEMPERATURE_BUCKETS:
        if value < lower:
            continue
        if upper is None or value < upper:
            return label
    return TEMPERATURE_BUCKETS[-1][2]


# ---------------------------------------------------------------------------
# Event lexicon: surface form -> canonical atom.
# Includes common inflections and derived compounds.
# ---------------------------------------------------------------------------

EVENT_LEXICON: Final[dict[str, str]] = {
    # Rain
    "regen": "regen",
    "regens": "regen",
    "regnen": "regen",
    "regnet": "regen",
    "regnete": "regen",
    "regenschauer": "regen",
    "regengebiete": "regen",
    "regengebiet": "regen",
    "niederschlag": "regen",
    "niederschläge": "regen",
    "niederschlaege": "regen",
    # Sun
    "sonne": "sonne",
    "sonnenschein": "sonne",
    "sonnig": "sonne",
    "sonnige": "sonne",
    "sonniger": "sonne",
    "sonniges": "sonne",
    "sonnigen": "sonne",
    # Snow
    "schnee": "schnee",
    "schneit": "schnee",
    "schneien": "schnee",
    "schneefall": "schnee",
    "schneefälle": "schnee",
    "schneeschauer": "schnee",
    "schneeregen": "schnee",
    # Clouds
    "wolke": "wolken",
    "wolken": "wolken",
    "wolkig": "wolken",
    "wolkige": "wolken",
    "wolkiger": "wolken",
    "wolkiges": "wolken",
    "wolkigen": "wolken",
    "bewölkt": "wolken",
    "bewölkung": "wolken",
    "bewoelkt": "wolken",
    "bewoelkung": "wolken",
    # Storm
    "sturm": "sturm",
    "stürme": "sturm",
    "stuerme": "sturm",
    "stürmisch": "sturm",
    "stuermisch": "sturm",
    "sturmböen": "sturm",
    "sturmboeen": "sturm",
    # Wind
    "wind": "wind",
    "winde": "wind",
    "windig": "wind",
    "windige": "wind",
    "windiger": "wind",
    "windiges": "wind",
    "windigen": "wind",
    "böen": "wind",
    "boeen": "wind",
    # Thunderstorm
    "gewitter": "gewitter",
    "gewittrig": "gewitter",
    "gewitterschauer": "gewitter",
    "blitze": "gewitter",
    "donner": "gewitter",
    # Fog
    "nebel": "nebel",
    "neblig": "nebel",
    "neblige": "nebel",
    "hochnebel": "nebel",
    # Frost
    "frost": "frost",
    "frostig": "frost",
    "frostige": "frost",
    "bodenfrost": "frost",
    # Hail
    "hagel": "hagel",
    "hagelt": "hagel",
    # Heat
    "hitze": "hitze",
    "heiß": "hitze",
    "heiss": "hitze",
    "heiße": "hitze",
    "heisse": "hitze",
    "heißer": "hitze",
    "heisser": "hitze",
    # Cold
    "kälte": "kälte",
    "kaelte": "kälte",
    "kalt": "kälte",
    "kalte": "kälte",
    "kalter": "kälte",
    "kaltes": "kälte",
    "kalten": "kälte",
    "kühl": "kälte",
    "kuehl": "kälte",
    "kühle": "kälte",
    "kuehle": "kälte",
    "kühler": "kälte",
    "kuehler": "kälte",
    "kühles": "kälte",
    "kuehles": "kälte",
    "kühlen": "kälte",
    "kuehlen": "kälte",
    # Shower (generic)
    "schauer": "schauer",
    "schauern": "schauer",
    # Ice / glaze
    "eis": "eis",
    "eisig": "eis",
    "glätte": "eis",
    "glaette": "eis",
    "glatteis": "eis",
    # Warm / mild
    "warm": "warm",
    "warme": "warm",
    "warmer": "warm",
    "warmes": "warm",
    "warmen": "warm",
    "mild": "warm",
    "milde": "warm",
    "milder": "warm",
    "mildes": "warm",
    "milden": "warm",
    # Dry
    "trocken": "trocken",
    "trockene": "trocken",
    "trockener": "trocken",
    "trockenes": "trocken",
    "trockenen": "trocken",
}


# ---------------------------------------------------------------------------
# Location lexicon: German regions, directions, geographic features.
# ---------------------------------------------------------------------------

LOCATION_LEXICON: Final[dict[str, str]] = {
    # Cardinal directions
    "nord": "nord",
    "norden": "nord",
    "nördlich": "nord",
    "noerdlich": "nord",
    "nordens": "nord",
    "norddeutschland": "nord",
    "süd": "süd",
    "sued": "süd",
    "süden": "süd",
    "sueden": "süd",
    "südlich": "süd",
    "suedlich": "süd",
    "südens": "süd",
    "suedens": "süd",
    "süddeutschland": "süd",
    "sueddeutschland": "süd",
    "ost": "ost",
    "osten": "ost",
    "östlich": "ost",
    "oestlich": "ost",
    "ostens": "ost",
    "ostdeutschland": "ost",
    "west": "west",
    "westen": "west",
    "westlich": "west",
    "westens": "west",
    "westdeutschland": "west",
    # Compound directions
    "nordwest": "nordwest",
    "nordwesten": "nordwest",
    "nordwestlich": "nordwest",
    "nordost": "nordost",
    "nordosten": "nordost",
    "nordöstlich": "nordost",
    "nordoestlich": "nordost",
    "südwest": "südwest",
    "suedwest": "südwest",
    "südwesten": "südwest",
    "suedwesten": "südwest",
    "südwestlich": "südwest",
    "suedwestlich": "südwest",
    "südost": "südost",
    "suedost": "südost",
    "südosten": "südost",
    "suedosten": "südost",
    "südöstlich": "südost",
    "suedoestlich": "südost",
    # Middle
    "mitte": "mitte",
    "mittendrin": "mitte",
    "mitteldeutschland": "mitte",
    # States (Bundesländer)
    "bayern": "bayern",
    "bayerisch": "bayern",
    "bayerische": "bayern",
    "bayerischer": "bayern",
    "baden": "baden",
    "badisch": "baden",
    "badische": "baden",
    "württemberg": "baden",
    "wuerttemberg": "baden",
    "sachsen": "sachsen",
    "sächsisch": "sachsen",
    "saechsisch": "sachsen",
    "hessen": "hessen",
    "hessisch": "hessen",
    "thüringen": "thüringen",
    "thueringen": "thüringen",
    "brandenburg": "brandenburg",
    "mecklenburg": "mecklenburg-vorpommern",
    "vorpommern": "mecklenburg-vorpommern",
    "schleswig": "schleswig-holstein",
    "holstein": "schleswig-holstein",
    "niedersachsen": "niedersachsen",
    "saarland": "saarland",
    "rheinland": "rheinland",
    "pfalz": "rheinland",
    "nrw": "nordrhein-westfalen",
    "westfalen": "nordrhein-westfalen",
    "nordrhein": "nordrhein-westfalen",
    # Geographic features
    "alpen": "alpen",
    "alpenrand": "alpen",
    "alpenraum": "alpen",
    "alpennordrand": "alpen",
    "alpensüdrand": "alpen",
    "alpensuedrand": "alpen",
    "küste": "küste",
    "kueste": "küste",
    "küsten": "küste",
    "kuesten": "küste",
    "küstenregion": "küste",
    "kuestenregion": "küste",
    "nordsee": "nordsee",
    "ostsee": "ostsee",
    # Named regions
    "breisgau": "breisgau",
    "vogtland": "vogtland",
    "schwarzwald": "schwarzwald",
    "harz": "harz",
    "erzgebirge": "erzgebirge",
    "oberrhein": "oberrhein",
    "rhein": "rhein",
    "eifel": "eifel",
    "ruhrgebiet": "ruhrgebiet",
    "oder": "oder",
    "sauerland": "sauerland",
    "spessart": "spessart",
    "odenwald": "odenwald",
    "taunus": "taunus",
    "voralpenland": "voralpen",
    "voralpen": "voralpen",
    "allgäu": "allgäu",
    "allgaeu": "allgäu",
    "bodensee": "bodensee",
    "erzgebirgsvorland": "erzgebirge",
    # Big cities (occasionally used as landmarks)
    "berlin": "berlin",
    "hamburg": "hamburg",
    "münchen": "münchen",
    "muenchen": "münchen",
    "köln": "köln",
    "koeln": "köln",
    "frankfurt": "frankfurt",
    "stuttgart": "stuttgart",
    "leipzig": "leipzig",
    "dresden": "dresden",
    # Generic
    "landesweit": "landesweit",
    "deutschlandweit": "landesweit",
    "überall": "landesweit",
    "ueberall": "landesweit",
    "gebietsweise": None,     # too generic; treated as an intensity qualifier
}
LOCATION_LEXICON = {k: v for k, v in LOCATION_LEXICON.items() if v is not None}


# ---------------------------------------------------------------------------
# Time lexicon.
# ---------------------------------------------------------------------------

TIME_LEXICON: Final[dict[str, str]] = {
    "heute": "heute",
    "morgen": "morgen",
    "übermorgen": "uebermorgen",
    "uebermorgen": "uebermorgen",
    "gestern": "gestern",
    "nacht": "nacht",
    "nachts": "nacht",
    "tag": "tag",
    "tags": "tag",
    "tagsüber": "tag",
    "tagsueber": "tag",
    "mittag": "mittag",
    "mittags": "mittag",
    "abend": "abend",
    "abends": "abend",
    "vormittag": "vormittag",
    "vormittags": "vormittag",
    "nachmittag": "nachmittag",
    "nachmittags": "nachmittag",
    "wochenende": "wochenende",
    "montag": "montag",
    "dienstag": "dienstag",
    "mittwoch": "mittwoch",
    "donnerstag": "donnerstag",
    "freitag": "freitag",
    "samstag": "samstag",
    "sonntag": "sonntag",
    "früh": "früh",
    "frueh": "früh",
    "spät": "spät",
    "spaet": "spät",
}


# ---------------------------------------------------------------------------
# Intensity lexicon.
# ---------------------------------------------------------------------------

INTENSITY_LEXICON: Final[dict[str, str]] = {
    "stark": "stark",
    "starke": "stark",
    "starker": "stark",
    "starkes": "stark",
    "starken": "stark",
    "kräftig": "kräftig",
    "kraeftig": "kräftig",
    "kräftige": "kräftig",
    "kraeftige": "kräftig",
    "kräftiger": "kräftig",
    "kraeftiger": "kräftig",
    "kräftiges": "kräftig",
    "kraeftiges": "kräftig",
    "kräftigen": "kräftig",
    "kraeftigen": "kräftig",
    "leicht": "leicht",
    "leichte": "leicht",
    "leichter": "leicht",
    "leichtes": "leicht",
    "leichten": "leicht",
    "mäßig": "mäßig",
    "maessig": "mäßig",
    "mäßige": "mäßig",
    "maessige": "mäßig",
    "mäßiger": "mäßig",
    "maessiger": "mäßig",
    "mäßiges": "mäßig",
    "maessiges": "mäßig",
    "mäßigen": "mäßig",
    "maessigen": "mäßig",
    "zeitweise": "zeitweise",
    "teilweise": "teilweise",
    "meist": "meist",
    "meistens": "meist",
    "vereinzelt": "vereinzelt",
    "gebietsweise": "gebietsweise",
    "örtlich": "örtlich",
    "oertlich": "örtlich",
    "verbreitet": "verbreitet",
}


# ---------------------------------------------------------------------------
# Wind direction lexicon.
# These forms are ambiguous with location "nördlich" (in the north) unless
# they occur near a wind-context word. The extractor resolves the ambiguity.
# ---------------------------------------------------------------------------

WIND_DIRECTION_LEXICON: Final[dict[str, str]] = {
    "nördlich": "nördlich",
    "noerdlich": "nördlich",
    "nördliche": "nördlich",
    "noerdliche": "nördlich",
    "nördlicher": "nördlich",
    "noerdlicher": "nördlich",
    "südlich": "südlich",
    "suedlich": "südlich",
    "südliche": "südlich",
    "suedliche": "südlich",
    "südlicher": "südlich",
    "suedlicher": "südlich",
    "östlich": "östlich",
    "oestlich": "östlich",
    "östliche": "östlich",
    "oestliche": "östlich",
    "östlicher": "östlich",
    "oestlicher": "östlich",
    "westlich": "westlich",
    "westliche": "westlich",
    "westlicher": "westlich",
    "nordöstlich": "nordöstlich",
    "nordoestlich": "nordöstlich",
    "nordwestlich": "nordwestlich",
    "südöstlich": "südöstlich",
    "suedoestlich": "südöstlich",
    "südwestlich": "südwestlich",
    "suedwestlich": "südwestlich",
}

# Wind-context words that promote an ambiguous *lich adjective from location
# to wind direction. Extractor checks a 3-token window.
WIND_CONTEXT_WORDS: Final[frozenset[str]] = frozenset({
    "wind", "winde", "windig", "böen", "boeen", "brise", "brisen",
    "strömung", "stroemung", "sturm", "stürme", "stuerme", "stürmisch",
    "stuermisch", "sturmböen", "sturmboeen",
})


# ---------------------------------------------------------------------------
# Polarity markers (negation). Applies within a small window before the event.
# ---------------------------------------------------------------------------

POLARITY_MARKERS: Final[frozenset[str]] = frozenset({
    "kein", "keine", "keiner", "keinen", "keines", "keinem",
    "nicht", "ohne", "nirgendwo", "nirgends", "nirgendwohin",
})


# ---------------------------------------------------------------------------
# Clause boundary tokens for relation binding. A relation is only formed
# between atoms that appear in the same clause.
# ---------------------------------------------------------------------------

CLAUSE_BOUNDARY_PUNCT: Final[frozenset[str]] = frozenset({",", ";", ".", "!", "?", ":"})
CLAUSE_BOUNDARY_WORDS: Final[frozenset[str]] = frozenset({
    "und", "aber", "oder", "dann", "sowie", "sonst", "während", "waehrend",
    "während", "sondern",
})


# ---------------------------------------------------------------------------
# German number words for temperature parsing.
# ---------------------------------------------------------------------------

GERMAN_NUMBER_WORDS: Final[dict[str, int]] = {
    "null": 0,
    "eins": 1, "ein": 1, "eine": 1, "einer": 1,
    "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "fuenf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
    "elf": 11, "zwölf": 12, "zwoelf": 12, "dreizehn": 13, "vierzehn": 14,
    "fünfzehn": 15, "fuenfzehn": 15, "sechzehn": 16, "siebzehn": 17,
    "achtzehn": 18, "neunzehn": 19, "zwanzig": 20,
    "einundzwanzig": 21, "zweiundzwanzig": 22, "dreiundzwanzig": 23,
    "vierundzwanzig": 24, "fünfundzwanzig": 25, "fuenfundzwanzig": 25,
    "sechsundzwanzig": 26, "siebenundzwanzig": 27, "achtundzwanzig": 28,
    "neunundzwanzig": 29, "dreißig": 30, "dreissig": 30,
    "einunddreißig": 31, "einunddreissig": 31,
    "zweiunddreißig": 32, "zweiunddreissig": 32,
    "fünfunddreißig": 35, "fuenfunddreissig": 35,
    "vierzig": 40,
}


# ---------------------------------------------------------------------------
# Slot dispatcher: which lexicon serves which slot.
# ---------------------------------------------------------------------------

SLOT_TO_LEXICON: Final[dict[Slot, dict[str, str]]] = {
    Slot.EVENT: EVENT_LEXICON,
    Slot.LOCATION: LOCATION_LEXICON,
    Slot.TIME: TIME_LEXICON,
    Slot.INTENSITY: INTENSITY_LEXICON,
    Slot.WIND_DIRECTION: WIND_DIRECTION_LEXICON,
}


@dataclass(frozen=True)
class SchemaSummary:
    events: int
    locations: int
    times: int
    intensities: int
    wind_directions: int
    temperature_buckets: int


def schema_summary() -> SchemaSummary:
    return SchemaSummary(
        events=len(set(EVENT_LEXICON.values())),
        locations=len(set(LOCATION_LEXICON.values())),
        times=len(set(TIME_LEXICON.values())),
        intensities=len(set(INTENSITY_LEXICON.values())),
        wind_directions=len(set(WIND_DIRECTION_LEXICON.values())),
        temperature_buckets=len(TEMPERATURE_BUCKETS),
    )
