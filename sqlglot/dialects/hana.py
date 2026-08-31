from __future__ import annotations

from sqlglot import tokens
from sqlglot.dialects.dialect import Dialect, NormalizationStrategy
from sqlglot.generators.hana import HanaGenerator
from sqlglot.parsers.hana import HanaParser
from sqlglot.tokens import TokenType

# Canonical (upper-case) datetime format elements. SAP HANA's format model follows Oracle's,
# which is also how Hibernate's HANADialect implements it -- it delegates straight to
# OracleDialect.datetimeFormat.
# https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/to-varchar-function-data-type-conversion
_TIME_ELEMENTS = {
    "DAY": "%A",  # name of day
    "DD": "%d",  # day of month (01-31)
    "DDD": "%j",  # day of year (001-366)
    "DY": "%a",  # abbreviated name of day
    "HH12": "%I",  # hour of day (01-12)
    "HH24": "%H",  # hour of day (00-23)
    "MI": "%M",  # minute (00-59)
    "MM": "%m",  # month (01-12)
    "MON": "%b",  # abbreviated name of month
    "MONTH": "%B",  # name of month
    "SS": "%S",  # second (00-59)
    "WW": "%W",  # week of year
    "YY": "%y",  # 26
    "YYYY": "%Y",  # 2026
    # HANA's default timestamp format carries 7 fractional digits, but Python's %f is
    # microseconds, so FF7 is a lossy match on the last digit.
    "FF6": "%f",
    "FF7": "%f",
}


class Hana(Dialect):
    # Undelimited identifiers are folded to upper case; delimited ones keep their case.
    # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/identifiers
    NORMALIZATION_STRATEGY = NormalizationStrategy.UPPERCASE

    # HANA matches format elements case-insensitively in practice, but sqlglot's format_time
    # matches literally against a prebuilt trie, so every element is registered in its upper-,
    # lower- and title-case spellings. Normalizing the format string instead would corrupt the
    # literal pass-through text a format model may contain.
    TIME_MAPPING = {
        spelling: strftime
        for element, strftime in _TIME_ELEMENTS.items()
        for spelling in (element, element.lower(), element.capitalize())
    }

    # TIME_MAPPING is inverted for generation, so its duplicate values (the case variants above,
    # and FF6/FF7) would otherwise resolve by dict order. Pin the canonical spelling instead of
    # relying on that ordering.
    INVERSE_TIME_MAPPING = {
        "%A": "DAY",
        "%a": "DY",
        "%B": "MONTH",
        "%b": "MON",
        "%d": "DD",
        "%f": "FF7",
        "%H": "HH24",
        "%I": "HH12",
        "%j": "DDD",
        "%M": "MI",
        "%m": "MM",
        "%S": "SS",
        "%W": "WW",
        "%Y": "YYYY",
        "%y": "YY",
    }

    class Tokenizer(tokens.Tokenizer):
        # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/identifiers
        IDENTIFIERS = ['"']
        QUOTES = ["'"]

        KEYWORDS = {
            **tokens.Tokenizer.KEYWORDS,
            # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/data-types
            "ALPHANUM": TokenType.VARCHAR,
            "NCLOB": TokenType.TEXT,
            # SECONDDATE is second-granular, so it maps to the second-granular type rather than
            # to TIMESTAMP -- otherwise a HANA round trip silently widens the column.
            "SECONDDATE": TokenType.TIMESTAMP_S,
            # SMALLDECIMAL is a variable-precision decimal FLOAT (p 1-16); no sqlglot type models
            # that, so it degrades to DECIMAL (p 1-34). This round trip is deliberately not an
            # identity: the alternative, leaving it user-defined, leaks SMALLDECIMAL into every
            # other dialect, which is worse.
            "SMALLDECIMAL": TokenType.DECIMAL,
        }

    Parser = HanaParser

    Generator = HanaGenerator
