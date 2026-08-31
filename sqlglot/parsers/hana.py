from __future__ import annotations

import typing as t

from sqlglot import exp, parser
from sqlglot.dialects.dialect import build_date_delta, build_timetostr_or_tochar
from sqlglot.helper import seq_get

# SAP HANA exposes one ADD_<unit>S and one <unit>S_BETWEEN function per supported unit. The set
# is narrower than most engines': there is no ADD_WEEKS and no WEEKS_BETWEEN, so WEEK must not be
# added here — it would parse SQL that HANA cannot run.
# https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/add-days-function-datetime
# https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/days-between-function-datetime
DATE_UNITS = {"YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"}


def _build_date_diff(unit: str) -> t.Callable[[list], exp.DateDiff]:
    """Build a DateDiff from HANA's <unit>S_BETWEEN, whose arguments run the other way.

    DAYS_BETWEEN(<date_1>, <date_2>) counts forward from the first argument, i.e. it evaluates
    to date_2 - date_1, whereas exp.DateDiff means this - expression. The two are swapped here
    so the parsed tree carries HANA's meaning rather than its argument order; the generator
    swaps back, which keeps HANA -> HANA an identity.
    """

    def _builder(args: list) -> exp.DateDiff:
        return exp.DateDiff(
            this=seq_get(args, 1),
            expression=seq_get(args, 0),
            unit=exp.Literal.string(unit),
        )

    return _builder


class HanaParser(parser.Parser):
    FUNCTIONS = {
        **parser.Parser.FUNCTIONS,
        **{f"ADD_{unit}S": build_date_delta(exp.DateAdd, default_unit=unit) for unit in DATE_UNITS},
        **{f"{unit}S_BETWEEN": _build_date_diff(unit) for unit in DATE_UNITS},
        # TO_VARCHAR is both a cast and a datetime formatter, depending on whether a format model
        # is supplied. TO_NVARCHAR is its Unicode twin and behaves identically here.
        # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/to-varchar-function-data-type-conversion
        "TO_VARCHAR": build_timetostr_or_tochar,
        "TO_NVARCHAR": build_timetostr_or_tochar,
        # LOCATE(<haystack>, <needle>) — note the argument order is the opposite of the ANSI
        # POSITION(<needle> IN <haystack>) form that exp.StrPosition models.
        # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/locate-function-string
        "LOCATE": lambda args: exp.StrPosition(
            this=seq_get(args, 0), substr=seq_get(args, 1), position=seq_get(args, 2)
        ),
    }

    # HASH_MD5 / HASH_SHA256 are deliberately NOT mapped onto exp.MD5 / exp.SHA2. They take a
    # BINARY argument, so the generator emits them wrapped in TO_BINARY(...); mapping them back
    # here would re-wrap on every round trip. Left anonymous, HANA -> HANA is already identity.
