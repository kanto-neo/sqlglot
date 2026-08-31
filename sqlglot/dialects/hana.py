from __future__ import annotations

from sqlglot import tokens
from sqlglot.dialects.dialect import Dialect, NormalizationStrategy
from sqlglot.generators.hana import HanaGenerator
from sqlglot.parsers.hana import HanaParser
from sqlglot.tokens import TokenType


class Hana(Dialect):
    # Undelimited identifiers are folded to upper case; delimited ones keep their case.
    # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/identifiers
    NORMALIZATION_STRATEGY = NormalizationStrategy.UPPERCASE

    # SAP HANA's datetime format model follows Oracle's, which is also how Hibernate's
    # HANADialect implements it (it delegates straight to OracleDialect.datetimeFormat).
    # The model is documented as case insensitive, but this mapping is matched literally, so
    # the lower-case spellings of the common parts are listed alongside the upper-case ones.
    # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/to-varchar-function-data-type-conversion
    TIME_MAPPING = {
        "DAY": "%A",  # name of day
        "DD": "%d",  # day of month (01-31)
        "dd": "%d",
        "DDD": "%j",  # day of year (001-366)
        "DY": "%a",  # abbreviated name of day
        "HH12": "%I",  # hour of day (01-12)
        "HH24": "%H",  # hour of day (00-23)
        "MI": "%M",  # minute (00-59)
        "MM": "%m",  # month (01-12)
        "mm": "%m",
        "MON": "%b",  # abbreviated name of month
        "MONTH": "%B",  # name of month
        "SS": "%S",  # second (00-59)
        "WW": "%W",  # week of year
        "YY": "%y",  # 26
        "yy": "%y",
        "YYYY": "%Y",  # 2026
        "yyyy": "%Y",
        # HANA's default timestamp format carries 7 fractional digits, but Python's %f is
        # microseconds, so FF7 is a lossy match on the last digit.
        "FF6": "%f",
        "FF7": "%f",
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
            "SECONDDATE": TokenType.TIMESTAMP,
            "SMALLDECIMAL": TokenType.DECIMAL,
        }

    Parser = HanaParser

    Generator = HanaGenerator
