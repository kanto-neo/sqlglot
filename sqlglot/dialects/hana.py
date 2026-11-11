from __future__ import annotations

from sqlglot import tokens
from sqlglot.dialects.dialect import Dialect, NormalizationStrategy
from sqlglot.generators.hana import HanaGenerator
from sqlglot.parsers.hana import HanaParser
from sqlglot.tokens import TokenType

# HANA's format model follows Oracle's
_TIME_ELEMENTS = {
    "AM": "%p",
    "DAY": "%A",
    "DD": "%d",
    "DDD": "%j",
    "DY": "%a",
    "FF6": "%f",
    "FF7": "%f",
    "HH": "%I",
    "HH12": "%I",
    "HH24": "%H",
    "MI": "%M",
    "MM": "%m",
    "MON": "%b",
    "MONTH": "%B",
    "PM": "%p",
    "SS": "%S",
    "WW": "%W",
    "YY": "%y",
    "YYYY": "%Y",
}


class Hana(Dialect):
    NORMALIZATION_STRATEGY = NormalizationStrategy.UPPERCASE
    TABLESAMPLE_SIZE_IS_PERCENT = True

    # Format elements are case-insensitive
    TIME_MAPPING = {
        spelling: strftime
        for element, strftime in _TIME_ELEMENTS.items()
        for spelling in (element, element.lower(), element.capitalize())
    }

    INVERSE_TIME_MAPPING = {
        "%A": "DAY",
        "%a": "DY",
        "%B": "MONTH",
        "%b": "MON",
        "%d": "DD",
        "%f": "FF6",
        "%H": "HH24",
        "%I": "HH12",
        "%j": "DDD",
        "%M": "MI",
        "%m": "MM",
        "%p": "AM",
        "%S": "SS",
        "%W": "WW",
        "%Y": "YYYY",
        "%y": "YY",
    }

    class Tokenizer(tokens.Tokenizer):
        IDENTIFIERS = ['"']
        QUOTES = ["'"]
        HEX_STRINGS = [("x'", "'"), ("X'", "'"), ("0x", ""), ("0X", "")]

        KEYWORDS = {
            **tokens.Tokenizer.KEYWORDS,
            "ALPHANUM": TokenType.VARCHAR,
            "BLOB": TokenType.BLOB,
            "CURRENT_UTCDATE": TokenType.UTC_DATE,
            "CURRENT_UTCTIME": TokenType.UTC_TIME,
            "CURRENT_UTCTIMESTAMP": TokenType.UTC_TIMESTAMP,
            # A bare FLOAT is 64-bit in HANA
            "FLOAT": TokenType.DOUBLE,
            "LIKE_REGEXPR": TokenType.RLIKE,
            "NCLOB": TokenType.TEXT,
            # TIMESTAMP_S would only be understood by DuckDB
            "SECONDDATE": TokenType.TIMESTAMP,
            "SMALLDECIMAL": TokenType.DECIMAL,
            "TOP": TokenType.TOP,
        }

    Parser = HanaParser

    Generator = HanaGenerator
