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

# The only units with an ADD_<unit>S / <unit>S_BETWEEN pair
DATE_UNITS = {"YEAR", "MONTH", "DAY", "SECOND"}


def _build_date_diff(unit: str) -> t.Callable[[list], exp.DateDiff]:
    # <unit>S_BETWEEN(a, b) evaluates to b - a
    def _builder(args: list) -> exp.DateDiff:
        return exp.DateDiff(
            this=seq_get(args, 1), expression=seq_get(args, 0), unit=exp.Literal.string(unit)
        )

    return _builder


def _build_to_datetime(
    parse_class: type[exp.StrToDate | exp.StrToTime], to: str
) -> t.Callable[[list, DialectType], exp.Expr]:
    # Without a format model these are plain conversions
    def _builder(args: list, dialect: DialectType) -> exp.Expr:
        if len(args) == 1:
            return exp.cast(args[0], to)
        return build_formatted_time(parse_class)(args, t.cast(Dialect, dialect))

    return _builder


def _build_cast(to: str) -> t.Callable[[list], exp.Expr]:
    def _builder(args: list) -> exp.Expr:
        if len(args) != 1:
            return exp.Anonymous(this=f"TO_{to}", expressions=args)
        return exp.cast(args[0], to)

    return _builder


def _build_to_decimal(args: list) -> exp.Expr:
    if len(args) > 3:
        return exp.Anonymous(this="TO_DECIMAL", expressions=args)
    params = [exp.DataTypeParam(this=arg) for arg in args[1:]]
    return exp.cast(args[0], exp.DataType(this=exp.DType.DECIMAL, expressions=params or None))


class HanaParser(parser.Parser):
    FUNCTIONS = {
        **parser.Parser.FUNCTIONS,
        **{f"ADD_{unit}S": build_date_delta(exp.DateAdd, default_unit=unit) for unit in DATE_UNITS},
        **{f"{unit}S_BETWEEN": _build_date_diff(unit) for unit in DATE_UNITS},
        "CARDINALITY": exp.ArraySize.from_arg_list,
        "LOCATE": lambda args: exp.StrPosition(
            this=seq_get(args, 0),
            substr=seq_get(args, 1),
            position=seq_get(args, 2),
            occurrence=seq_get(args, 3),
        ),
        "MAP": lambda args: exp.DecodeCase(expressions=args),
        # HANA's NEXT_DAY(<date>) is the day after <date>
        "NEXT_DAY": lambda args: (
            exp.DateAdd(this=args[0], expression=exp.Literal.number(1), unit=exp.var("DAY"))
            if len(args) == 1
            else exp.Anonymous(this="NEXT_DAY", expressions=args)
        ),
        # The third argument is a rounding mode such as ROUND_DOWN
        "ROUND": lambda args: (
            exp.Anonymous(this="ROUND", expressions=args)
            if len(args) > 2
            else exp.Round.from_arg_list(args)
        ),
        "TO_BIGINT": _build_cast("BIGINT"),
        "TO_DATE": _build_to_datetime(exp.StrToDate, "DATE"),
        "TO_DECIMAL": _build_to_decimal,
        "TO_DOUBLE": _build_cast("DOUBLE"),
        "TO_INT": _build_cast("INT"),
        "TO_INTEGER": _build_cast("INT"),
        "TO_NVARCHAR": build_timetostr_or_tochar,
        "TO_SMALLINT": _build_cast("SMALLINT"),
        "TO_TIMESTAMP": _build_to_datetime(exp.StrToTime, "TIMESTAMP"),
        "TO_VARCHAR": build_timetostr_or_tochar,
    }

    FUNCTION_PARSERS = {
        **parser.Parser.FUNCTION_PARSERS,
        "SUBSTR_REGEXPR": lambda self: self._parse_substr_regexpr(),
    }

    NO_PAREN_FUNCTIONS = {
        **parser.Parser.NO_PAREN_FUNCTIONS,
        TokenType.UTC_DATE: exp.UtcDate,
        TokenType.UTC_TIME: exp.UtcTime,
        TokenType.UTC_TIMESTAMP: exp.UtcTimestamp,
    }

    RANGE_PARSERS = {
        **parser.Parser.RANGE_PARSERS,
        TokenType.RLIKE: lambda self, this: self._parse_like_regexpr(this),
    }

    def _parse_like_regexpr(self, this: exp.Expr) -> exp.RegexpLike:
        expression = self._parse_bitwise()
        flag = self._parse_string() if self._match_text_seq("FLAG") else None
        return self.expression(exp.RegexpLike(this=this, expression=expression, flag=flag))

    def _parse_substr_regexpr(self) -> exp.RegexpExtract:
        # SUBSTR_REGEXPR(<pattern> [FLAG <flags>] IN <string> [FROM <n>] [OCCURRENCE <n>] [GROUP <n>])
        pattern = self._parse_bitwise()
        flags = self._parse_string() if self._match_text_seq("FLAG") else None
        self._match(TokenType.IN)
        args: dict[str, t.Any] = {"this": self._parse_bitwise(), "expression": pattern}
        if flags:
            args["parameters"] = flags

        for key, keyword in (
            ("position", "FROM"),
            ("occurrence", "OCCURRENCE"),
            ("group", "GROUP"),
        ):
            if self._match_text_seq(keyword):
                args[key] = self._parse_bitwise()

        return self.expression(exp.RegexpExtract(**args))

    def _parse_locks(self) -> list[exp.Lock]:
        # FOR SHARE LOCK and IGNORE LOCKED are HANA's spellings of FOR SHARE and SKIP LOCKED
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
