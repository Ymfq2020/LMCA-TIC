"""Offline BIE construction from internal temporal KG statistics.

This module implements the recommended no-external-network BIE pipeline:
derive entity background attributes only from the local ICEWS splits.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from lmca_tic.utils.io import ensure_dir


def _progress(iterable, total: int | None = None, desc: str = ""):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )


COUNTRY_HINTS = {
    "afghanistan",
    "algeria",
    "argentina",
    "australia",
    "austria",
    "bangladesh",
    "belgium",
    "brazil",
    "canada",
    "china",
    "colombia",
    "egypt",
    "france",
    "germany",
    "india",
    "indonesia",
    "iran",
    "iraq",
    "israel",
    "italy",
    "japan",
    "jordan",
    "lebanon",
    "libya",
    "mexico",
    "pakistan",
    "palestine",
    "qatar",
    "russia",
    "saudi arabia",
    "syria",
    "turkey",
    "ukraine",
    "united kingdom",
    "united states",
    "usa",
    "venezuela",
    "yemen",
}

PERSON_TITLES = {
    "president",
    "prime minister",
    "minister",
    "secretary",
    "general",
    "governor",
    "senator",
    "judge",
    "king",
    "queen",
    "commander",
    "chief",
    "leader",
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
}

ORG_HINTS = {
    "government",
    "ministry",
    "council",
    "agency",
    "committee",
    "army",
    "military",
    "police",
    "party",
    "university",
    "bank",
    "company",
    "corporation",
    "union",
    "organization",
    "org",
    "group",
    "forces",
}

LOCATION_HINTS = {
    "city",
    "province",
    "district",
    "state",
    "region",
    "county",
    "town",
    "capital",
    "airport",
}

INTERNATIONAL_ORGS = {
    "united nations",
    "european union",
    "nato",
    "eu",
    "un",
    "asean",
    "security council",
    "world bank",
    "international monetary fund",
}

SECTOR_KEYWORDS = {
    "Diplomacy": {
        "meet",
        "consult",
        "negotiate",
        "visit",
        "discuss",
        "mediate",
        "appeal",
        "cooperate",
        "aid",
        "support",
        "sanction",
        "sign",
        "agree",
    },
    "Security": {
        "attack",
        "arrest",
        "fight",
        "kill",
        "bomb",
        "military",
        "deploy",
        "protest",
        "clash",
        "violence",
        "threaten",
        "troop",
    },
    "Politics": {
        "elect",
        "vote",
        "appoint",
        "resign",
        "campaign",
        "parliament",
        "legislate",
        "govern",
        "cabinet",
        "party",
    },
    "Economy": {
        "trade",
        "invest",
        "economy",
        "bank",
        "oil",
        "gas",
        "market",
        "price",
        "budget",
        "tax",
    },
    "Media": {
        "interview",
        "broadcast",
        "publish",
        "announce",
        "report",
        "media",
    },
}


@dataclass
class EntityStats:
    name: str
    mention_count: int = 0
    subject_count: int = 0
    object_count: int = 0
    neighbor_counter: Counter[str] = field(default_factory=Counter)
    relation_counter: Counter[str] = field(default_factory=Counter)
    timestamp_counter: Counter[int] = field(default_factory=Counter)

    @property
    def neighbor_count(self) -> int:
        return len(self.neighbor_counter)

    def top_relations(self, k: int = 5) -> list[str]:
        return [relation for relation, _ in self.relation_counter.most_common(k)]


class OfflineBIEBuilder:
    def __init__(self, delimiter: str = "\t") -> None:
        self.delimiter = delimiter

    def build_from_dir(self, raw_dir: str | Path, output_path: str | Path) -> dict[str, dict[str, str]]:
        raw_dir = Path(raw_dir)
        stats = self.collect_stats(raw_dir)
        entity_types = self._infer_entity_types(stats)
        countries = self._infer_countries(stats, entity_types)
        records = self._materialize_records(stats, entity_types, countries)
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        with output_path.open("w", encoding="utf-8") as handle:
            for entity_id in _progress(sorted(records), total=len(records), desc="write bie"):
                handle.write(json.dumps(records[entity_id], ensure_ascii=False) + "\n")
        return records

    def collect_stats(self, raw_dir: str | Path) -> dict[str, EntityStats]:
        raw_dir = Path(raw_dir)
        stats: dict[str, EntityStats] = {}
        for split in _progress(("train", "valid", "test"), total=3, desc="bie splits"):
            path = raw_dir / f"{split}.txt"
            if not path.exists():
                continue
            total_lines = self._count_lines(path)
            with path.open("r", encoding="utf-8") as handle:
                for line in _progress(handle, total=total_lines, desc=f"bie {split}"):
                    line = line.strip()
                    if not line:
                        continue
                    subject, relation, obj, timestamp = self._split_line(line)
                    if subject not in stats:
                        stats[subject] = EntityStats(name=subject)
                    if obj not in stats:
                        stats[obj] = EntityStats(name=obj)
                    timestamp_value = int(timestamp)

                    stats[subject].mention_count += 1
                    stats[subject].subject_count += 1
                    stats[subject].neighbor_counter[obj] += 1
                    stats[subject].relation_counter[relation] += 1
                    stats[subject].timestamp_counter[timestamp_value] += 1

                    stats[obj].mention_count += 1
                    stats[obj].object_count += 1
                    stats[obj].neighbor_counter[subject] += 1
                    stats[obj].relation_counter[f"{relation}__rev"] += 1
                    stats[obj].timestamp_counter[timestamp_value] += 1
        return stats

    def _infer_entity_types(self, stats: dict[str, EntityStats]) -> dict[str, str]:
        inferred: dict[str, str] = {}
        for entity_id, entity_stats in stats.items():
            inferred[entity_id] = self._infer_entity_type(entity_id, entity_stats)
        return inferred

    def _infer_entity_type(self, entity_name: str, stats: EntityStats) -> str:
        lowered = entity_name.lower()
        normalized = re.sub(r"\s+", " ", lowered).strip()
        if normalized in INTERNATIONAL_ORGS:
            return "InternationalOrganization"
        if normalized in COUNTRY_HINTS:
            return "Country"
        if any(title in lowered for title in PERSON_TITLES):
            return "Person"
        if any(hint in lowered for hint in ORG_HINTS):
            if "government" in lowered or "ministry" in lowered:
                return "GovernmentOrganization"
            if "party" in lowered:
                return "PoliticalOrganization"
            if "bank" in lowered or "company" in lowered or "corporation" in lowered:
                return "Company"
            return "Organization"
        if any(hint in lowered for hint in LOCATION_HINTS):
            return "Location"
        if self._looks_like_person_name(entity_name):
            return "Person"

        top_relations = ",".join(stats.top_relations(5)).lower()
        if any(keyword in top_relations for keyword in ("elect", "appoint", "resign", "vote")):
            return "PoliticalActor"
        if any(keyword in top_relations for keyword in ("attack", "arrest", "bomb", "troop")):
            return "SecurityActor"
        if stats.subject_count > 0 and stats.object_count == 0:
            return "Actor"
        if stats.object_count > stats.subject_count * 2:
            return "Target"
        return "Entity"

    def _infer_countries(
        self,
        stats: dict[str, EntityStats],
        entity_types: dict[str, str],
    ) -> dict[str, str]:
        countries: dict[str, str] = {}
        country_entities = {
            entity_id
            for entity_id, entity_type in entity_types.items()
            if entity_type == "Country"
        }
        for entity_id, entity_stats in stats.items():
            if entity_types[entity_id] == "Country":
                countries[entity_id] = entity_id
                continue
            lowered = entity_id.lower()
            matched = next((country for country in COUNTRY_HINTS if country in lowered), None)
            if matched is not None:
                countries[entity_id] = self._titleize_country(matched)
                continue
            candidate_counter: Counter[str] = Counter()
            for neighbor, count in entity_stats.neighbor_counter.items():
                if neighbor in country_entities:
                    candidate_counter[neighbor] += count
            if candidate_counter:
                countries[entity_id] = candidate_counter.most_common(1)[0][0]
            elif entity_types[entity_id] == "InternationalOrganization":
                countries[entity_id] = "INTERNATIONAL"
            else:
                countries[entity_id] = "UNKNOWN"
        return countries

    def _materialize_records(
        self,
        stats: dict[str, EntityStats],
        entity_types: dict[str, str],
        countries: dict[str, str],
    ) -> dict[str, dict[str, str]]:
        records: dict[str, dict[str, str]] = {}
        for entity_id, entity_stats in stats.items():
            top_relations = entity_stats.top_relations(5)
            sector = self._infer_sector(entity_stats)
            records[entity_id] = {
                "entity_id": entity_id,
                "entity_name": entity_id,
                "entity_type": entity_types[entity_id],
                "country": countries[entity_id],
                "sector": sector,
                "top_relations": ",".join(top_relations) if top_relations else "UNKNOWN",
                "interaction_count": str(entity_stats.mention_count),
                "neighbor_count": str(entity_stats.neighbor_count),
                "description": self._build_description(
                    entity_id=entity_id,
                    entity_type=entity_types[entity_id],
                    country=countries[entity_id],
                    sector=sector,
                    entity_stats=entity_stats,
                    top_relations=top_relations,
                ),
            }
        return records

    def _infer_sector(self, stats: EntityStats) -> str:
        relation_text = " ".join(stats.relation_counter.keys()).lower()
        best_sector = "General"
        best_score = 0
        for sector, keywords in SECTOR_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in relation_text)
            if score > best_score:
                best_sector = sector
                best_score = score
        return best_sector

    def _build_description(
        self,
        entity_id: str,
        entity_type: str,
        country: str,
        sector: str,
        entity_stats: EntityStats,
        top_relations: list[str],
    ) -> str:
        relation_text = ", ".join(top_relations[:3]) if top_relations else "no dominant relations"
        return (
            f"{entity_id} is an internally inferred {entity_type}. "
            f"It is associated with country hint {country} and sector hint {sector}. "
            f"It appears in {entity_stats.mention_count} interactions with "
            f"{entity_stats.neighbor_count} unique neighbors. "
            f"Top relations are: {relation_text}."
        )

    def _looks_like_person_name(self, entity_name: str) -> bool:
        tokens = [token for token in re.split(r"[\s\-_]+", entity_name) if token]
        if len(tokens) < 2 or len(tokens) > 4:
            return False
        uppercase_tokens = sum(1 for token in tokens if token[:1].isupper())
        return uppercase_tokens == len(tokens)

    def _titleize_country(self, country: str) -> str:
        return " ".join(part.capitalize() for part in country.split())

    def _split_line(self, line: str) -> tuple[str, str, str, str]:
        if self.delimiter == " ":
            parts = line.split()
        else:
            parts = line.split(self.delimiter)
        if len(parts) != 4:
            raise ValueError(f"Expected 4 columns, got {len(parts)} in line: {line}")
        return parts[0], parts[1], parts[2], parts[3]

    def _count_lines(self, path: Path) -> int:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
