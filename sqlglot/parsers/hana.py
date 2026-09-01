from __future__ import annotations

import typing as t

from sqlglot import exp, parser
from sqlglot.dialects.dialect import (
    Dialect,
    DialectType,
    build_date_delta,
    build_formatted_time,
    build_timetostr_or_tochar,
)
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType

# SAP HANA exposes an ADD_<unit>S / <unit>S_BETWEEN pair for exactly these four units. The set is
# much narrower than most engines': there is no ADD_HOURS, ADD_MINUTES, ADD_WEEKS, HOURS_BETWEEN,
# MINUTES_BETWEEN or WEEKS_BETWEEN. Adding a unit here would parse SQL that HANA cannot run.
# The remaining members of the family -- ADD_NANO100/NANO100_BETWEEN and
# ADD_WORKDAYS/WORKDAYS_BETWEEN -- deliberately stay out: they do not fit the pluralized naming
# this comprehension builds, and ADD_WORKDAYS is not a 2-argument delta (it takes a factory
# calendar id). Left unmapped they stay anonymous, which already round-trips.
# https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/datetime-functions
DATE_UNITS = {"YEAR", "MONTH", "DAY", "SECOND"}


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


def _build_to_datetime(
    exp_class: type[exp.StrToDate | exp.StrToTime], func_name: str
) -> t.Callable[[list, DialectType], exp.Expr]:
    """Build the TO_DATE / TO_TIMESTAMP parse functions, guarding the one-argument form.

    TO_DATE(<expr>) / TO_TIMESTAMP(<expr>) with no format model is a plain conversion, not a
    parse. Turning it into a StrToDate/StrToTime with format=None makes every target that
    requires a format emit a call missing its mandatory argument (BigQuery PARSE_DATE(x),
    DuckDB STRPTIME(x)). Left anonymous, the one-argument form round-trips correctly.
    """

    def _builder(args: list, dialect: DialectType) -> exp.Expr:
        if len(args) == 1:
            return exp.Anonymous(this=func_name, expressions=args)

        return build_formatted_time(exp_class)(args, t.cast(Dialect, dialect))

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
        # ... and TO_DATE / TO_TIMESTAMP are the inbound direction. Without these, HANA's own
        # format models leak out unconverted as Python strftime strings.
        "TO_DATE": _build_to_datetime(exp.StrToDate, "TO_DATE"),
        "TO_TIMESTAMP": _build_to_datetime(exp.StrToTime, "TO_TIMESTAMP"),
        # LOCATE(<haystack>, <needle>[, <start_position>[, <occurrences>]]) -- note the argument
        # order is the opposite of the ANSI POSITION(<needle> IN <haystack>) form that
        # exp.StrPosition models. A negative <start_position> means "search right to left" in
        # HANA; that does not survive translation to dialects which emulate `position`.
        # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/locate-function-string
        # HANA's array length function, paired with ARRAY_SIZE_NAME on the generator so the
        # round trip is an identity rather than degrading to the default ARRAY_LENGTH.
        "CARDINALITY": exp.ArraySize.from_arg_list,
        # HANA's MAP(<expr>, <search1>, <result1>, ..., [<default>]) is a variadic CASE-like
        # selector, nothing like the key/value map constructor exp.Map models -- and exp.Map
        # caps at two arguments, so the base entry turns any real HANA MAP into a parse error.
        # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/map-function-miscellaneous
        "MAP": lambda args: exp.Anonymous(this="MAP", expressions=args),
        # HANA's NEXT_DAY takes ONE argument and means "the day after <date>", whereas
        # exp.NextDay models Oracle's two-argument NEXT_DAY(<date>, <weekday>) and requires the
        # second, so the base entry rejects valid HANA. Expressed as a day delta instead, which
        # both round-trips as ADD_DAYS(<date>, 1) and carries the meaning to other dialects.
        # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/next-day-function-datetime
        "NEXT_DAY": lambda args: exp.DateAdd(
            this=seq_get(args, 0),
            expression=exp.Literal.number(1),
            unit=exp.Literal.string("DAY"),
        ),
        "LOCATE": lambda args: exp.StrPosition(
            this=seq_get(args, 0),
            substr=seq_get(args, 1),
            position=seq_get(args, 2),
            occurrence=seq_get(args, 3),
        ),
    }

    RANGE_PARSERS = {
        **parser.Parser.RANGE_PARSERS,
        # LIKE_REGEXPR is tokenized as RLIKE (see the dialect's Tokenizer), but it carries an
        # optional trailing FLAG that binary_range_parser would leave behind as a stray token.
        TokenType.RLIKE: lambda self, this: self._parse_like_regexpr(this),
    }

    def _parse_like_regexpr(self, this: exp.Expr) -> exp.RegexpLike:
        """Parse `<expr> LIKE_REGEXPR <pattern> [FLAG <flags>]`.

        exp.RegexpLike already carries an optional `flag`, so the modifier survives the round
        trip instead of being dropped or failing to parse.
        https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/like-regexpr-predicate
        """
        expression = self._parse_bitwise()
        flag = self._parse_string() if self._match_text_seq("FLAG") else None

        return self.expression(exp.RegexpLike(this=this, expression=expression, flag=flag))

    def _parse_locks(self) -> list[exp.Lock]:
        """Parse SAP HANA's locking-read clauses.

        HANA spells the shared lock FOR SHARE LOCK rather than FOR SHARE, and SKIP LOCKED as
        IGNORE LOCKED; it has no Postgres-style KEY variants. The base implementation stops after
        FOR SHARE and then fails on the trailing LOCK, so the generator's output would not parse
        back without this.
        https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/select-statement-data-manipulation
        """
        locks = []

        while True:
            if self._match_text_seq("FOR", "UPDATE"):
                update = True
            elif self._match_text_seq("FOR", "SHARE"):
                self._match_text_seq("LOCK")
                update = False
            else:
                break

            expressions = None
            if self._match_text_seq("OF"):
                expressions = self._parse_csv(lambda: self._parse_table(schema=True))

            wait: bool | exp.Expr | None = None
            if self._match_text_seq("NOWAIT"):
                wait = True
            elif self._match_text_seq("WAIT"):
                wait = self._parse_primary()
            elif self._match_text_seq("IGNORE", "LOCKED"):
                wait = False

            locks.append(
                self.expression(exp.Lock(update=update, expressions=expressions, wait=wait))
            )

        return locks

    # HASH_MD5 / HASH_SHA256 are deliberately NOT mapped onto exp.MD5 / exp.SHA2. They take a
    # BINARY argument, so the generator emits them wrapped in TO_BINARY(...); mapping them back
    # here would re-wrap on every round trip. Left anonymous, HANA -> HANA is already identity.
    # STRING_AGG is likewise left to the base FUNCTION_PARSERS entry, which already handles it.
