from __future__ import annotations

import math
import re
from typing import Any

from dateutil import parser as date_parser


MISSING_VALUES = {"", "--", "N/A", "NA", "None", "nan", "NaN"}


def normalize_missing(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return None if text in MISSING_VALUES else text


def parse_int(value: Any) -> int | None:
    text = normalize_missing(value)
    if text is None:
        return None
    match = re.search(r"-?\d+", str(text))
    return int(match.group(0)) if match else None


def parse_float(value: Any) -> float | None:
    text = normalize_missing(value)
    if text is None:
        return None
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def parse_percentage(value: Any) -> float | None:
    text = normalize_missing(value)
    if text is None:
        return None
    text = str(text).replace("%", "")
    parsed = parse_float(text)
    if parsed is None:
        return None
    return parsed / 100.0 if parsed > 1 else parsed


def parse_height_to_cm(value: Any) -> float | None:
    text = normalize_missing(value)
    if text is None:
        return None
    match = re.search(r"(\d+)\s*'\s*(\d+)", str(text))
    if not match:
        return None
    feet = int(match.group(1))
    inches = int(match.group(2))
    return round((feet * 12 + inches) * 2.54, 1)


def parse_reach_to_cm(value: Any) -> float | None:
    text = normalize_missing(value)
    if text is None:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(text))
    return round(float(match.group(1)) * 2.54, 1) if match else None


def parse_weight_to_lbs(value: Any) -> int | None:
    return parse_int(value)


def parse_time_to_seconds(value: Any) -> int | None:
    text = normalize_missing(value)
    if text is None:
        return None
    match = re.fullmatch(r"(\d+):(\d{2})", str(text))
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def parse_control_time_to_seconds(value: Any) -> int | None:
    return parse_time_to_seconds(value)


def parse_date_to_iso(value: Any) -> str | None:
    text = normalize_missing(value)
    if text is None:
        return None
    try:
        return date_parser.parse(str(text)).date().isoformat()
    except Exception:
        return None


def split_location(value: Any) -> tuple[str | None, str | None, str | None]:
    text = normalize_missing(value)
    if text is None:
        return None, None, None
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        return parts[0], parts[1], None
    return parts[0], ", ".join(parts[1:-1]) or None, parts[-1]


def parse_record(value: Any) -> tuple[int | None, int | None, int | None]:
    text = normalize_missing(value)
    if text is None:
        return None, None, None
    match = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", str(text))
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def parse_method_group(value: Any) -> str:
    text = str(normalize_missing(value) or "").upper()
    if not text:
        return "OTHER"
    if "DRAW" in text:
        return "DRAW"
    if "NO CONTEST" in text or "COULD NOT CONTINUE" in text or re.search(r"\bNC\b", text):
        return "NC"
    if "DQ" in text or "DISQUALIFICATION" in text:
        return "DQ"
    if any(token in text for token in ["DECISION", "UNANIMOUS", "SPLIT", "MAJORITY", "DEC "]):
        return "DEC"
    if any(token in text for token in ["SUB", "SUBMISSION", "CHOKE", "ARMBAR", "TOOL"]):
        return "SUB"
    if any(token in text for token in ["TKO", "T.K.O", "KO", "K.O"]):
        return "KO_TKO"
    return "OTHER"


def parse_landed_attempted(value: Any) -> tuple[int | None, int | None]:
    text = normalize_missing(value)
    if text is None:
        return None, None
    match = re.search(r"(\d+)\s*(?:of|/)\s*(\d+)", str(text), re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    parsed = parse_int(text)
    return parsed, None


def parse_name_parts(full_name: str | None) -> tuple[str | None, str | None]:
    text = normalize_missing(full_name)
    if text is None:
        return None, None
    parts = str(text).split()
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


def seconds_to_round_time(ending_round: int | None, ending_time_seconds: int | None) -> int | None:
    if ending_round is None or ending_time_seconds is None:
        return None
    return max(0, (ending_round - 1) * 300 + ending_time_seconds)


def parse_fight_duration(ending_round: int | None, ending_time_seconds: int | None, scheduled_rounds: int | None, method_group: str) -> int | None:
    if ending_round is not None and ending_time_seconds is not None:
        return seconds_to_round_time(ending_round, ending_time_seconds)
    if method_group == "DEC" and scheduled_rounds is not None:
        return scheduled_rounds * 300
    return None