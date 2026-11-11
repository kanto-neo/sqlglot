from __future__ import annotations

from sqlglot import exp, generator, transforms
from sqlglot.dialects.dialect import (
    concat_to_dpipe_sql,
    concat_ws_to_dpipe_sql,
    groupconcat_sql,
    map_date_part,
    no_ilike_sql,
    no_trycast_sql,
    rename_func,
    unit_to_var,
)
from sqlglot.parsers.hana import DATE_UNITS

# HANA only accepts OFFSET after a LIMIT; this is the value SAP's sqlalchemy-hana uses
MAX_LIMIT = 2147384648

EXTRACT_NATIVE_PARTS = {"YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"}
EXTRACT_FUNCTION_BY_PART = {
    "DAYOFMONTH": "DAYOFMONTH",
    "DAYOFYEAR": "DAYOFYEAR",
    "DOY": "DAYOFYEAR",
}

# Units without an ADD_<unit>S / <unit>S_BETWEEN function, expressed through one that has
SCALED_UNITS = {
    "HOUR": ("SECOND", 3600),
    "MINUTE": ("SECOND", 60),
    "WEEK": ("DAY", 7),
    "QUARTER": ("MONTH", 3),
}

TRUNC_INTERVAL_BY_UNIT = {
    "YEAR": "INTERVAL 1 YEAR",
    "QUARTER": "INTERVAL 3 MONTH",
    "MONTH": "INTERVAL 1 MONTH",
    "DAY": "INTERVAL 1 DAY",
    "HOUR": "INTERVAL 1 HOUR",
    "MINUTE": "INTERVAL 1 MINUTE",
    "SECOND": "INTERVAL 1 SECOND",
}

# WEEKDAY is Monday(0)..Sunday(6); exp.DayOfWeek is Sunday(1)..Saturday(7)
DAY_OF_WEEK_TEMPLATE: exp.Expr = exp.maybe_parse("(MOD(WEEKDAY(:this) + 1, 7) + 1)")
DAY_OF_WEEK_ISO_TEMPLATE: exp.Expr = exp.maybe_parse("(WEEKDAY(:this) + 1)")
# QUARTER and ISOWEEK return strings ('2024-Q2', '2024-W05')
QUARTER_TEMPLATE: exp.Expr = exp.maybe_parse("(FLOOR((MONTH(:this) - 1) / 3) + 1)")
WEEK_ISO_TEMPLATE: exp.Expr = exp.maybe_parse("CAST(SUBSTRING(ISOWEEK(:this), 7, 2) AS INT)")


def _unit_name(expression: exp.Expr) -> str:
    unit = unit_to_var(expression)
    if isinstance(unit, exp.WeekStart):
        return "WEEK"
    return map_date_part(unit).name.upper() if unit else "DAY"


def _to_number(value: exp.Expr) -> exp.Expr:
    # Several dialects carry interval amounts as string literals
    if value.is_string:
        try:
            number = float(value.name)
        except ValueError:
            return value
        return exp.Literal.number(int(number) if number.is_integer() else number)
    return value


def _scale(offset: exp.Expr, factor: int) -> exp.Expr:
    offset = _to_number(offset)
    if offset.is_number:
        number = offset.to_py() * factor
        return exp.Literal.number(int(number) if float(number).is_integer() else number)
    return offset * exp.Literal.number(factor)


def _date_add_sql(self: HanaGenerator, expression: exp.Func) -> str:
    unit = _unit_name(expression)
    offset = _to_number(expression.expression)

    if unit in SCALED_UNITS:
        unit, factor = SCALED_UNITS[unit]
        offset = _scale(offset, factor)

    if unit not in DATE_UNITS:
        self.unsupported(f"ADD_{unit}S does not exist in SAP HANA.")
        return self.function_fallback_sql(expression)

    return self.func(f"ADD_{unit}S", expression.this, offset)


def _date_sub_sql(self: HanaGenerator, expression: exp.Func) -> str:
    offset = _to_number(expression.expression)
    negated = offset.this if isinstance(offset, exp.Neg) else exp.Neg(this=exp.paren(offset))
    if offset.is_number:
        negated = exp.Literal.number(-offset.to_py())

    return _date_add_sql(
        self,
        exp.DateAdd(this=expression.this, expression=negated, unit=expression.args.get("unit")),
    )


def _date_diff_sql(self: HanaGenerator, expression: exp.Func) -> str:
    unit = _unit_name(expression)

    # <unit>S_BETWEEN(start, end) evaluates to end - start
    start, end = expression.expression, expression.this

    # Dialects such as Snowflake count unit boundaries rather than elapsed units
    if expression.args.get("date_part_boundary"):
        interval = TRUNC_INTERVAL_BY_UNIT.get(unit)
        if not interval:
            self.unsupported(f"Counting {unit} boundaries is not supported in SAP HANA.")
        else:
            start, end = (_series_round_down(e, interval) for e in (start, end))

    if unit in SCALED_UNITS:
        # Casting truncates toward zero, i.e. counts whole elapsed units
        base_unit, factor = SCALED_UNITS[unit]
        between = exp.func(f"{base_unit}S_BETWEEN", start, end)
        return self.sql(exp.cast(between / exp.Literal.number(factor), exp.DType.BIGINT))

    if unit not in DATE_UNITS:
        self.unsupported(f"{unit}S_BETWEEN does not exist in SAP HANA.")
        return self.function_fallback_sql(expression)

    return self.func(f"{unit}S_BETWEEN", start, end)


def _series_round_down(this: exp.Expr, interval: str) -> exp.Expr:
    return exp.func("SERIES_ROUND", this, exp.Literal.string(interval), exp.var("ROUND_DOWN"))


def _date_trunc_sql(self: HanaGenerator, expression: exp.DateTrunc | exp.TimestampTrunc) -> str:
    unit = _unit_name(expression)
    interval = TRUNC_INTERVAL_BY_UNIT.get(unit)

    # SERIES_ROUND aligns intervals to a fixed origin, so weeks wouldn't start on Monday
    if not interval:
        self.unsupported(f"Truncating to {unit} is not supported in SAP HANA.")
        return self.function_fallback_sql(expression)

    return self.sql(_series_round_down(expression.this, interval))


def _hash_sql(self: HanaGenerator, func_name: str, expression: exp.Func, hex: bool) -> str:
    length = expression.text("length") or "256"
    if length != "256":
        self.unsupported(f"SHA2 with length {length} is not supported in SAP HANA.")
        return self.function_fallback_sql(expression)

    # HASH_* take and return VARBINARY; other dialects' hex digests are lowercase
    digest = exp.func(func_name, exp.func("TO_BINARY", expression.this))
    return self.func("LOWER", exp.func("BINTOHEX", digest)) if hex else self.sql(digest)


def _to_varchar_sql(self: HanaGenerator, expression: exp.ToChar | exp.TimeToStr) -> str:
    fmt = expression.args.get("format")
    return self.func("TO_VARCHAR", expression.this, self.format_time(expression) if fmt else None)


def _alias_derived_table_columns(expression: exp.Expr) -> exp.Expr:
    # HANA rejects `(<query>) AS t(a, b)`, so the names become aliases of the projections
    if not isinstance(expression, exp.Select):
        return expression

    for source in (expression.args.get("from_"), *expression.args.get("joins", [])):
        subquery = source.this if source else None
        if not isinstance(subquery, exp.Subquery):
            continue

        alias = subquery.args.get("alias")
        if not alias or not alias.columns:
            continue

        columns = alias.columns
        select = subquery.this
        while isinstance(select, exp.SetOperation):
            select = select.this

        if (
            isinstance(select, exp.Select)
            and len(select.expressions) == len(columns)
            and not select.is_star
        ):
            select.set(
                "expressions",
                [exp.alias_(p.unalias(), c) for p, c in zip(select.expressions, columns)],
            )
            alias.set("columns", None)

    return expression


def _add_dummy_from(expression: exp.Expr) -> exp.Expr:
    # DUMMY is HANA's one-row table, standing in for a missing FROM or Oracle's DUAL
    if not isinstance(expression, exp.Select) or not expression.expressions:
        return expression

    source = expression.args.get("from_")
    table = source.this if source else None
    is_dual = (
        isinstance(table, exp.Table)
        and table.name.upper() == "DUAL"
        and not table.db
        and not table.this.args.get("quoted")
    )
    if not source or is_dual:
        expression.set("from_", exp.From(this=exp.to_table("DUMMY")))
    return expression


class HanaGenerator(generator.Generator):
    LIMIT_FETCH = "LIMIT"
    VALUES_AS_TABLE = False
    SUPPORTS_TABLE_ALIAS_COLUMNS = False
    TABLESAMPLE_SIZE_IS_ROWS = False
    AFTER_HAVING_MODIFIER_TRANSFORMS = generator.AFTER_HAVING_MODIFIER_TRANSFORMS
    JOIN_HINTS = False
    TABLE_HINTS = False
    QUERY_HINTS = False
    TRY_SUPPORTED = False
    SUPPORTS_UESCAPE = False
    LAST_DAY_SUPPORTS_DATE_PART = False
    LOCKING_READS_SUPPORTED = True
    ARRAY_SIZE_NAME = "CARDINALITY"
    EXCEPT_INTERSECT_SUPPORT_ALL_CLAUSE = False
    SUPPORTS_TO_NUMBER = False
    NVL2_SUPPORTED = False

    TYPE_MAPPING = {
        **generator.Generator.TYPE_MAPPING,
        exp.DType.BINARY: "VARBINARY",
        exp.DType.BLOB: "BLOB",
        exp.DType.DATETIME: "TIMESTAMP",
        # FLOAT is 64-bit in HANA; REAL is the 32-bit type
        exp.DType.FLOAT: "REAL",
        # TEXT is a full-text search type in HANA
        exp.DType.TEXT: "NCLOB",
        exp.DType.JSON: "NCLOB",
        exp.DType.JSONB: "NCLOB",
        exp.DType.NVARCHAR: "NVARCHAR",
        exp.DType.TIMESTAMPNTZ: "TIMESTAMP",
        exp.DType.TIMESTAMPTZ: "TIMESTAMP",
        exp.DType.TIMESTAMP_S: "SECONDDATE",
        exp.DType.UTINYINT: "TINYINT",
        exp.DType.UUID: "VARBINARY(16)",
    }

    TRANSFORMS = {
        **generator.Generator.TRANSFORMS,
        exp.Chr: rename_func("CHAR"),
        exp.Concat: concat_to_dpipe_sql,
        exp.ConcatWs: concat_ws_to_dpipe_sql,
        exp.CurrentUser: lambda *_: "CURRENT_USER",
        exp.DateAdd: _date_add_sql,
        exp.DateDiff: _date_diff_sql,
        exp.DatetimeAdd: _date_add_sql,
        exp.DatetimeDiff: _date_diff_sql,
        exp.DatetimeSub: _date_sub_sql,
        exp.DateSub: _date_sub_sql,
        exp.DateTrunc: _date_trunc_sql,
        exp.DayOfMonth: rename_func("DAYOFMONTH"),
        exp.DayOfYear: rename_func("DAYOFYEAR"),
        exp.DecodeCase: rename_func("MAP"),
        exp.ILike: no_ilike_sql,
        exp.MD5: lambda self, e: _hash_sql(self, "HASH_MD5", e, hex=True),
        exp.MD5Digest: lambda self, e: _hash_sql(self, "HASH_MD5", e, hex=False),
        exp.Mod: rename_func("MOD"),
        exp.SHA2: lambda self, e: _hash_sql(self, "HASH_SHA256", e, hex=True),
        exp.SHA2Digest: lambda self, e: _hash_sql(self, "HASH_SHA256", e, hex=False),
        exp.StrToDate: lambda self, e: self.func("TO_DATE", e.this, self.format_time(e)),
        exp.StrToTime: lambda self, e: self.func("TO_TIMESTAMP", e.this, self.format_time(e)),
        exp.TimeAdd: _date_add_sql,
        exp.TimeDiff: _date_diff_sql,
        exp.TimeStrToTime: rename_func("TO_TIMESTAMP"),
        exp.TimeSub: _date_sub_sql,
        exp.TimeToStr: _to_varchar_sql,
        exp.TimestampAdd: _date_add_sql,
        exp.TimestampDiff: _date_diff_sql,
        exp.TimestampSub: _date_sub_sql,
        exp.TimestampTrunc: _date_trunc_sql,
        exp.ToChar: _to_varchar_sql,
        exp.TryCast: no_trycast_sql,
        exp.TsOrDsAdd: _date_add_sql,
        exp.TsOrDsDiff: _date_diff_sql,
        exp.Select: transforms.preprocess(
            [
                _add_dummy_from,
                _alias_derived_table_columns,
                transforms.eliminate_distinct_on,
                transforms.eliminate_qualify,
                transforms.eliminate_semi_and_anti_joins,
            ]
        ),
    }

    def currentdate_sql(self, expression: exp.CurrentDate) -> str:
        return self._current_sql(expression, "CURRENT_DATE", "CURRENT_UTCDATE")

    def currenttime_sql(self, expression: exp.CurrentTime) -> str:
        return self._current_sql(expression, "CURRENT_TIME", "CURRENT_UTCTIME")

    def currenttimestamp_sql(self, expression: exp.CurrentTimestamp) -> str:
        return self._current_sql(expression, "CURRENT_TIMESTAMP", "CURRENT_UTCTIMESTAMP")

    def _current_sql(self, expression: exp.Expr, local: str, utc: str) -> str:
        zone = expression.this
        if zone and zone.name.upper() == "UTC":
            return utc
        if zone:
            self.unsupported(f"{local} with a time zone is not supported in SAP HANA.")
        return local

    def dayofweek_sql(self, expression: exp.DayOfWeek) -> str:
        return self.sql(exp.replace_placeholders(DAY_OF_WEEK_TEMPLATE.copy(), this=expression.this))

    def dayofweekiso_sql(self, expression: exp.DayOfWeekIso) -> str:
        return self.sql(
            exp.replace_placeholders(DAY_OF_WEEK_ISO_TEMPLATE.copy(), this=expression.this)
        )

    def quarter_sql(self, expression: exp.Quarter) -> str:
        return self.sql(exp.replace_placeholders(QUARTER_TEMPLATE.copy(), this=expression.this))

    def weekofyear_sql(self, expression: exp.WeekOfYear) -> str:
        return self.sql(exp.replace_placeholders(WEEK_ISO_TEMPLATE.copy(), this=expression.this))

    def extract_sql(self, expression: exp.Extract) -> str:
        part = expression.name.upper()
        this = expression.expression

        if part in EXTRACT_NATIVE_PARTS:
            return super().extract_sql(expression)

        func_name = EXTRACT_FUNCTION_BY_PART.get(part)
        if func_name:
            return self.func(func_name, this)

        if part == "QUARTER":
            return self.quarter_sql(exp.Quarter(this=this))
        if part == "WEEK":
            return self.weekofyear_sql(exp.WeekOfYear(this=this))
        # DOW is not mapped because its numbering differs between dialects
        if part == "ISODOW":
            return self.dayofweekiso_sql(exp.DayOfWeekIso(this=this))

        self.unsupported(f"EXTRACT part '{part}' is not supported in SAP HANA.")
        return super().extract_sql(expression)

    def trim_sql(self, expression: exp.Trim) -> str:
        # LTRIM/RTRIM remove a character set, but HANA has no two-argument TRIM for BOTH
        remove_set = expression.expression
        if remove_set and self.sql(expression, "position") in ("", "BOTH"):
            return self.func("LTRIM", self.func("RTRIM", expression.this, remove_set), remove_set)
        return super().trim_sql(expression)

    def groupconcat_sql(self, expression: exp.GroupConcat) -> str:
        if isinstance(expression.this, exp.Distinct):
            self.unsupported("SAP HANA's STRING_AGG does not support DISTINCT.")
        return groupconcat_sql(
            self, expression, func_name="STRING_AGG", sep=None, within_group=False
        )

    def repeat_sql(self, expression: exp.Repeat) -> str:
        this = expression.this
        times = expression.args.get("times")
        if not times:
            self.unsupported("REPEAT requires a repetition count in SAP HANA.")
            return self.function_fallback_sql(expression)
        return self.func("RPAD", this, times * exp.func("LENGTH", this.copy()), this.copy())

    def regexplike_sql(self, expression: exp.RegexpLike) -> str:
        sql = self.binary(expression, "LIKE_REGEXPR")
        flag = expression.args.get("flag")
        return f"{sql} FLAG {self.sql(flag)}" if flag else sql

    def strposition_sql(self, expression: exp.StrPosition) -> str:
        # LOCATE(<haystack>, <needle>[, <start>[, <occurrence>]])
        args: list[exp.Expr] = [expression.this, expression.args["substr"]]
        position = expression.args.get("position")
        occurrence = expression.args.get("occurrence")

        if position or occurrence:
            args.append(position or exp.Literal.number(1))
        if occurrence:
            args.append(occurrence)

        return self.func("LOCATE", *args)

    def splitpart_sql(self, expression: exp.SplitPart) -> str:
        self.unsupported("SAP HANA has no SPLIT_PART function.")
        return self.function_fallback_sql(expression)

    def interval_sql(self, expression: exp.Interval) -> str:
        self.unsupported("SAP HANA does not support INTERVAL literals.")
        return super().interval_sql(expression)

    def add_sql(self, expression: exp.Add) -> str:
        this, interval = expression.this, expression.expression
        if isinstance(this, exp.Interval):
            this, interval = interval, this

        if (
            isinstance(interval, exp.Interval)
            and interval.unit
            and not isinstance(this, exp.Interval)
        ):
            return _date_add_sql(
                self, exp.DateAdd(this=this, expression=interval.this, unit=interval.unit)
            )

        return super().add_sql(expression)

    def sub_sql(self, expression: exp.Sub) -> str:
        this, interval = expression.this, expression.expression
        if (
            isinstance(interval, exp.Interval)
            and interval.unit
            and not isinstance(this, exp.Interval)
        ):
            return _date_sub_sql(
                self, exp.DateSub(this=this, expression=interval.this, unit=interval.unit)
            )

        return super().sub_sql(expression)

    def datatype_sql(self, expression: exp.DataType) -> str:
        unbounded_binary = expression.is_type(exp.DType.BINARY, exp.DType.VARBINARY)
        in_cast = isinstance(expression.parent, exp.Cast)

        # A VARBINARY column without a length holds one byte, but in a CAST it is unbounded
        if unbounded_binary and not expression.expressions:
            return "VARBINARY" if in_cast else "BLOB"

        # A parameterized DOUBLE can only come from HANA's FLOAT(<n>)
        if expression.is_type(exp.DType.DOUBLE) and expression.expressions:
            return self.func("FLOAT", *expression.expressions)

        # LOBs can't be compared, grouped or sorted, and NVARCHAR is unbounded in a CAST
        if expression.is_type(exp.DType.TEXT) and in_cast:
            return "NVARCHAR"

        return super().datatype_sql(expression)

    def lock_sql(self, expression: exp.Lock) -> str:
        if expression.args.get("key"):
            self.unsupported("SAP HANA does not support FOR KEY SHARE / FOR NO KEY UPDATE.")

        sql = "FOR UPDATE" if expression.args["update"] else "FOR SHARE LOCK"

        expressions = self.expressions(expression, flat=True)
        if expressions:
            sql = f"{sql} OF {expressions}"

        wait = expression.args.get("wait")
        if isinstance(wait, exp.Expr):
            sql = f"{sql} WAIT {self.sql(wait)}"
        elif wait is True:
            sql = f"{sql} NOWAIT"
        elif wait is False:
            sql = f"{sql} IGNORE LOCKED"

        return sql

    def offset_limit_modifiers(
        self, expression: exp.Expr, fetch: bool, limit: exp.Fetch | exp.Limit | None
    ) -> list[str]:
        # The base generator turns FETCH into a bare LIMIT before this point
        original = expression.args.get("limit")
        options = original.args.get("limit_options") if original else None
        if options and (options.args.get("percent") or options.args.get("with_ties")):
            self.unsupported("SAP HANA's LIMIT does not support PERCENT or WITH TIES.")

        if expression.args.get("offset") and not limit:
            limit = exp.Limit(expression=exp.Literal.number(MAX_LIMIT))

        return [self.sql(limit), self.sql(expression, "offset")]
