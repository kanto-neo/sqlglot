from __future__ import annotations

import typing as t

from sqlglot import exp, generator
from sqlglot.dialects.dialect import (
    concat_to_dpipe_sql,
    concat_ws_to_dpipe_sql,
    groupconcat_sql,
    rename_func,
    trim_sql,
)
from sqlglot.parsers.hana import DATE_UNITS

# SAP HANA parses OFFSET only as part of a LIMIT clause -- `<limit_clause> ::= LIMIT
# <unsigned_integer> [ OFFSET <unsigned_integer> ]` -- so an offset-only query needs a limit
# synthesised. The value is a sentinel, not a documented bound: SAP states no maximum for LIMIT.
# It is the constant SAP's own sqlalchemy-hana dialect injects for this case, reused here so the
# two emit the same SQL.
# https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/select-statement-data-manipulation
MAX_LIMIT = 2147384648

# EXTRACT accepts exactly these six parts in SAP HANA.
# https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/extract-function-datetime
EXTRACT_NATIVE_PARTS = {"YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"}

# Parts EXTRACT rejects, but which HANA exposes as an integer-returning scalar function.
EXTRACT_FUNCTION_BY_PART = {
    "WEEK": "WEEK",
    "ISOWEEK": "ISOWEEK",
    "DAYOFMONTH": "DAYOFMONTH",
    "DAYOFYEAR": "DAYOFYEAR",
    "DOY": "DAYOFYEAR",
}

# HANA has no ADD_HOURS / ADD_MINUTES / HOURS_BETWEEN / MINUTES_BETWEEN, so those units are
# expressed in seconds instead. Same rewrite Hibernate's HANADialect uses.
SECONDS_PER_UNIT = {"HOUR": 3600, "MINUTE": 60}

# The interval-adding and interval-subtracting expressions this dialect routes through one
# pair of helpers. All are exp.Func subclasses carrying `this`, `expression` and `unit`.
DateDelta = t.Union[
    exp.DateAdd,
    exp.DateSub,
    exp.DatetimeAdd,
    exp.DatetimeSub,
    exp.TimeSub,
    exp.TimestampAdd,
    exp.TimestampSub,
    exp.TsOrDsAdd,
]


def _day_of_week_sql(self: HanaGenerator, expression: exp.DayOfWeek) -> str:
    """SAP HANA has no DAYOFWEEK. WEEKDAY is Monday(0)..Sunday(6).

    exp.DayOfWeek carries sqlglot's MySQL-style Sunday(1)..Saturday(7) convention, so the value
    has to be rotated, not just renamed -- emitting WEEKDAY alone would be off by a day and
    silently wrong.
    https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/weekday-function-datetime
    """
    return f"(MOD({self.func('WEEKDAY', expression.this)} + 1, 7) + 1)"


def _day_of_week_iso_sql(self: HanaGenerator, expression: exp.DayOfWeekIso) -> str:
    # WEEKDAY is 0-based from Monday; ISO-8601 is 1-based from Monday.
    return f"({self.func('WEEKDAY', expression.this)} + 1)"


def _extract_sql(self: HanaGenerator, expression: exp.Extract) -> str:
    part = expression.name.upper()

    if part in EXTRACT_NATIVE_PARTS:
        return generator.Generator.extract_sql(self, expression)

    func_name = EXTRACT_FUNCTION_BY_PART.get(part)
    if func_name:
        return self.func(func_name, expression.expression)

    self.unsupported(f"EXTRACT part '{part}' is not supported in SAP HANA.")
    return generator.Generator.extract_sql(self, expression)


def _repeat_sql(self: HanaGenerator, expression: exp.Repeat) -> str:
    """SAP HANA has no REPEAT function; RPAD padded with the string itself is the idiom.

    Emulated as RPAD(<str>, <times> * LENGTH(<str>), <str>), the same rewrite Hibernate's
    HANADialect uses. Note this pads to a computed length rather than concatenating, so it
    relies on <times> being non-negative -- as REPEAT itself does.
    """
    this = expression.this
    times = expression.args.get("times")

    if not times:
        self.unsupported("REPEAT requires a repetition count in SAP HANA.")
        return self.function_fallback_sql(expression)

    # `*` rather than exp.Mul(...) so a compound <times> is parenthesized by _binop; a bare
    # exp.Mul would emit `a + b * LENGTH(x)` and compute the wrong length.
    length = times * exp.func("LENGTH", this.copy())
    return self.func("RPAD", this, length, this.copy())


def _md5_sql(self: HanaGenerator, expression: exp.MD5) -> str:
    # HASH_MD5 takes a BINARY argument, so the input has to be converted first.
    # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/hash-md5-function-miscellaneous
    return self.func("HASH_MD5", exp.func("TO_BINARY", expression.this))


def _sha2_sql(self: HanaGenerator, expression: exp.SHA2) -> str:
    length = expression.text("length") or "256"

    # HASH_MD5 and HASH_SHA256 are the only hash functions SAP HANA documents; there is no
    # HASH_SHA1, HASH_SHA384 or HASH_SHA512.
    if length != "256":
        self.unsupported(f"SHA2 with length {length} is not supported in SAP HANA.")
        return self.function_fallback_sql(expression)

    return self.func("HASH_SHA256", exp.func("TO_BINARY", expression.this))


def _strposition_sql(self: HanaGenerator, expression: exp.StrPosition) -> str:
    """Render exp.StrPosition as HANA's LOCATE.

    LOCATE(<haystack>, <needle>[, <start_position>[, <occurrences>]]) takes the haystack first,
    the reverse of the MySQL/T-SQL LOCATE that the shared strposition_sql helper assumes -- which
    is why this cannot reuse it. Both optional arguments are supported natively.
    https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/locate-function-string
    """
    args: list[exp.Expr] = [expression.this, expression.args["substr"]]

    position = expression.args.get("position")
    occurrence = expression.args.get("occurrence")

    if position or occurrence:
        # <occurrences> is the fourth argument, so a bare occurrence still needs a start.
        args.append(position or exp.Literal.number(1))
    if occurrence:
        args.append(occurrence)

    return self.func("LOCATE", *args)


def _to_varchar_sql(self: HanaGenerator, expression: exp.ToChar | exp.TimeToStr) -> str:
    # TO_VARCHAR doubles as the datetime formatter; a NULL format model means a plain conversion.
    # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/to-varchar-function-data-type-conversion
    fmt = expression.args.get("format")
    return self.func("TO_VARCHAR", expression.this, self.format_time(expression) if fmt else None)


def _scale_to_seconds(offset: exp.Expr, seconds: int) -> exp.Expr:
    """Scale an hour/minute offset into seconds for the ADD_SECONDS / SECONDS_BETWEEN rewrites.

    A numeric literal is folded so the output reads `ADD_SECONDS(d, 10800)` rather than
    `3 * 3600`; this also covers the dialects that hand the offset over as a *string* literal,
    where `'3' * 3600` would otherwise be emitted. Anything else goes through the `*` operator
    overload, which parenthesizes a compound operand (a bare exp.Mul would not).
    """
    if isinstance(offset, exp.Literal):
        try:
            return exp.Literal.number(int(offset.name) * seconds)
        except ValueError:
            pass

    return offset * exp.Literal.number(seconds)


def _date_add_sql(self: HanaGenerator, expression: DateDelta) -> str:
    unit = expression.text("unit").upper() or "DAY"
    offset = expression.expression

    seconds = SECONDS_PER_UNIT.get(unit)
    if seconds:
        return self.func("ADD_SECONDS", expression.this, _scale_to_seconds(offset, seconds))

    if unit not in DATE_UNITS:
        self.unsupported(f"ADD_{unit}S does not exist in SAP HANA.")
        return self.function_fallback_sql(expression)

    return self.func(f"ADD_{unit}S", expression.this, offset)


def _date_sub_sql(self: HanaGenerator, expression: DateDelta) -> str:
    """Render the subtracting variants by negating the offset and reusing the ADD path.

    SAP HANA has no SUBTRACT_<unit>S family. The offset is negated numerically rather than
    wrapped in exp.Neg, because for several dialects it arrives as a *string* literal and
    `-'3'` is not valid SQL.
    """
    offset = expression.expression
    negated: exp.Expr

    if isinstance(offset, exp.Neg):
        negated = offset.this
    elif isinstance(offset, exp.Literal) and not offset.args.get("is_string"):
        negated = exp.Literal.number(f"-{offset.name}")
    elif isinstance(offset, exp.Literal):
        try:
            negated = exp.Literal.number(-int(offset.name))
        except ValueError:
            negated = exp.Neg(this=exp.paren(offset, copy=True))
    else:
        negated = exp.Neg(this=exp.paren(offset, copy=True))

    return _date_add_sql(
        self,
        exp.DateAdd(this=expression.this, expression=negated, unit=expression.args.get("unit")),
    )


def _date_diff_sql(self: HanaGenerator, expression: exp.DateDiff | exp.TsOrDsDiff) -> str:
    unit = expression.text("unit").upper() or "DAY"

    # <unit>S_BETWEEN(<start>, <end>) counts forward from the first argument, which is the
    # opposite of the (end, start) order exp.DateDiff stores.
    start, end = expression.expression, expression.this

    seconds = SECONDS_PER_UNIT.get(unit)
    if seconds:
        # No HOURS_BETWEEN / MINUTES_BETWEEN in HANA. Divide SECONDS_BETWEEN and cast, because
        # HANA's `/` is true division and would otherwise return a fraction where a diff is an
        # integer. This measures elapsed whole units rather than counting boundary crossings the
        # way T-SQL's DATEDIFF does.
        divided = exp.Div(
            this=exp.func("SECONDS_BETWEEN", start, end), expression=exp.Literal.number(seconds)
        )
        return self.sql(exp.cast(divided, exp.DType.BIGINT))

    if unit not in DATE_UNITS:
        self.unsupported(f"{unit}S_BETWEEN does not exist in SAP HANA.")
        return self.function_fallback_sql(expression)

    return self.func(f"{unit}S_BETWEEN", start, end)


class HanaGenerator(generator.Generator):
    # SAP HANA supports LIMIT but not FETCH FIRST, so a parsed FETCH is rewritten as a LIMIT.
    LIMIT_FETCH = "LIMIT"

    # CLUSTER BY / DISTRIBUTE BY / SORT BY are Hive-family extensions; assigning the
    # module-level dict drops them instead of emitting SQL HANA cannot parse.
    AFTER_HAVING_MODIFIER_TRANSFORMS = generator.AFTER_HAVING_MODIFIER_TRANSFORMS

    # HANA has no optimizer-hint syntax of the T-SQL/Hive shapes these flags gate, no TRY_CAST,
    # no UESCAPE, and LAST_DAY takes no date-part argument.
    JOIN_HINTS = False
    TABLE_HINTS = False
    QUERY_HINTS = False
    TRY_SUPPORTED = False
    SUPPORTS_UESCAPE = False
    LAST_DAY_SUPPORTS_DATE_PART = False

    # ... but it does support locking reads, which the base default would silently discard.
    LOCKING_READS_SUPPORTED = True

    TYPE_MAPPING = {
        **generator.Generator.TYPE_MAPPING,
        # SAP HANA has no BINARY type -- VARBINARY covers both.
        exp.DType.BINARY: "VARBINARY",
        exp.DType.DATETIME: "TIMESTAMP",
        # HANA's TEXT is a full-text search type, not a character type; NCLOB is the
        # unbounded string type. https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/data-types
        exp.DType.TEXT: "NCLOB",
        exp.DType.JSON: "NCLOB",
        exp.DType.JSONB: "NCLOB",
        # NVARCHAR is HANA's canonical Unicode string type; the base maps it down to VARCHAR.
        exp.DType.NVARCHAR: "NVARCHAR",
        exp.DType.TIMESTAMPNTZ: "TIMESTAMP",
        exp.DType.TIMESTAMPTZ: "TIMESTAMP",
        exp.DType.TIMESTAMP_S: "SECONDDATE",
        exp.DType.UUID: "VARBINARY(16)",
    }

    TRANSFORMS = {
        **generator.Generator.TRANSFORMS,
        exp.Concat: concat_to_dpipe_sql,
        exp.ConcatWs: concat_ws_to_dpipe_sql,
        exp.CurrentDate: lambda *_: "CURRENT_DATE",
        exp.CurrentTime: lambda *_: "CURRENT_TIME",
        exp.CurrentTimestamp: lambda *_: "CURRENT_TIMESTAMP",
        exp.DateAdd: _date_add_sql,
        exp.DateDiff: _date_diff_sql,
        exp.DateSub: _date_sub_sql,
        exp.DatetimeAdd: _date_add_sql,
        exp.DatetimeSub: _date_sub_sql,
        # SAP HANA spells these without separators, unlike sqlglot's default rendering.
        # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/dayofmonth-function-datetime
        exp.DayOfMonth: rename_func("DAYOFMONTH"),
        exp.DayOfWeek: _day_of_week_sql,
        exp.DayOfWeekIso: _day_of_week_iso_sql,
        exp.DayOfYear: rename_func("DAYOFYEAR"),
        exp.Extract: _extract_sql,
        # HANA's string aggregate is STRING_AGG; it has no GROUP_CONCAT.
        exp.GroupConcat: lambda self, e: groupconcat_sql(
            self, e, func_name="STRING_AGG", within_group=False
        ),
        exp.MD5: _md5_sql,
        # HANA's arithmetic operators are only unary -, +, -, *, / -- modulo is a function.
        exp.Mod: rename_func("MOD"),
        exp.Repeat: _repeat_sql,
        exp.SHA2: _sha2_sql,
        exp.StrPosition: _strposition_sql,
        exp.StrToDate: lambda self, e: self.func("TO_DATE", e.this, self.format_time(e)),
        exp.StrToTime: lambda self, e: self.func("TO_TIMESTAMP", e.this, self.format_time(e)),
        exp.TimeStrToTime: rename_func("TO_TIMESTAMP"),
        exp.TimeSub: _date_sub_sql,
        exp.TimeToStr: _to_varchar_sql,
        exp.TimestampAdd: _date_add_sql,
        exp.TimestampSub: _date_sub_sql,
        exp.ToChar: _to_varchar_sql,
        exp.Trim: trim_sql,
        exp.TsOrDsAdd: _date_add_sql,
        exp.TsOrDsDiff: _date_diff_sql,
    }

    def lock_sql(self, expression: exp.Lock) -> str:
        """Render exp.Lock using SAP HANA's spellings.

        HANA writes the shared lock as FOR SHARE LOCK rather than FOR SHARE, spells SKIP LOCKED
        as IGNORE LOCKED, and has no Postgres-style KEY variants, so the base implementation
        cannot be reused.
        https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/select-statement-data-manipulation
        """
        if expression.args.get("key"):
            self.unsupported("SAP HANA does not support FOR KEY SHARE / FOR NO KEY UPDATE.")

        sql = "FOR UPDATE" if expression.args["update"] else "FOR SHARE LOCK"

        expressions = self.expressions(expression, flat=True)
        if expressions:
            sql = f"{sql} OF {expressions}"

        if expression.args.get("wait") is not None:
            wait = expression.args["wait"]
            if isinstance(wait, exp.Expr):
                sql = f"{sql} WAIT {self.sql(wait)}"
            elif wait:
                sql = f"{sql} NOWAIT"
            else:
                sql = f"{sql} IGNORE LOCKED"

        return sql

    def offset_limit_modifiers(
        self, expression: exp.Expr, fetch: bool, limit: exp.Fetch | exp.Limit | None
    ) -> list[str]:
        offset = expression.args.get("offset")

        # SAP HANA parses OFFSET only as part of a LIMIT clause, so an offset with no limit
        # needs one synthesised rather than emitted bare.
        if offset and not limit:
            limit = exp.Limit(expression=exp.Literal.number(MAX_LIMIT))

        return [self.sql(limit), self.sql(expression, "offset")]
