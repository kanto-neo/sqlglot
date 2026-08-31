from __future__ import annotations

from sqlglot import exp, generator
from sqlglot.dialects.dialect import rename_func
from sqlglot.parsers.hana import DATE_UNITS

# SAP HANA rejects OFFSET unless a LIMIT accompanies it, so an offset-only query needs a limit
# synthesised. This is the largest value HANA accepts for LIMIT; it is the same constant
# SAP's own sqlalchemy-hana dialect injects for this case, chosen so the limit can never
# truncate a real result set.
MAX_LIMIT = 2147384648

# EXTRACT(<part> FROM <expr>) has no HANA equivalent — the parts are exposed as ordinary
# functions of the same name instead.
# https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/year-function-datetime
EXTRACT_FUNCTION_BY_PART = {
    "YEAR": "YEAR",
    "QUARTER": "QUARTER",
    "MONTH": "MONTH",
    "WEEK": "WEEK",
    "DAY": "DAYOFMONTH",
    "DAYOFMONTH": "DAYOFMONTH",
    "DAYOFWEEK": "DAYOFWEEK",
    "DAYOFYEAR": "DAYOFYEAR",
    "HOUR": "HOUR",
    "MINUTE": "MINUTE",
    "SECOND": "SECOND",
}


def _extract_sql(self: HanaGenerator, expression: exp.Extract) -> str:
    part = expression.name.upper()
    func_name = EXTRACT_FUNCTION_BY_PART.get(part)

    if not func_name:
        self.unsupported(f"EXTRACT part '{part}' is not supported in SAP HANA.")
        return self.function_fallback_sql(expression)

    return self.func(func_name, expression.expression)


def _repeat_sql(self: HanaGenerator, expression: exp.Repeat) -> str:
    """SAP HANA has no REPEAT function; RPAD padded with the string itself is the idiom.

    Emulated as RPAD(<str>, <times> * LENGTH(<str>), <str>), the same rewrite Hibernate's
    HANADialect uses. Note this pads to a computed length rather than concatenating, so it
    relies on <times> being non-negative — as REPEAT itself does.
    """
    this = expression.this
    times = expression.args.get("times")

    if not times:
        self.unsupported("REPEAT requires a repetition count in SAP HANA.")
        return self.function_fallback_sql(expression)

    length = exp.Mul(this=times.copy(), expression=exp.func("LENGTH", this.copy()))
    return self.func("RPAD", this, length, this.copy())


def _md5_sql(self: HanaGenerator, expression: exp.MD5) -> str:
    # HASH_MD5 takes a BINARY argument, so the input has to be converted first.
    # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/hash-md5-function-miscellaneous
    return self.func("HASH_MD5", exp.func("TO_BINARY", expression.this))


def _sha2_sql(self: HanaGenerator, expression: exp.SHA2) -> str:
    length = expression.text("length") or "256"

    if length not in ("256", "512"):
        self.unsupported(f"SHA2 with length {length} is not supported in SAP HANA.")
        return self.function_fallback_sql(expression)

    return self.func(f"HASH_SHA{length}", exp.func("TO_BINARY", expression.this))


def _date_add_sql(self: HanaGenerator, expression: exp.DateAdd | exp.TsOrDsAdd) -> str:
    unit = expression.text("unit").upper() or "DAY"

    if unit not in DATE_UNITS:
        self.unsupported(f"ADD_{unit}S does not exist in SAP HANA.")
        return self.function_fallback_sql(expression)

    return self.func(f"ADD_{unit}S", expression.this, expression.expression)


def _strposition_sql(self: HanaGenerator, expression: exp.StrPosition) -> str:
    """Render exp.StrPosition as HANA's LOCATE.

    LOCATE(<haystack>, <needle>[, <start_position>[, <occurrences>]]) takes the haystack first,
    the reverse of the MySQL/T-SQL LOCATE that the shared strposition_sql helper assumes — which
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


def _date_diff_sql(self: HanaGenerator, expression: exp.DateDiff | exp.TsOrDsDiff) -> str:
    unit = expression.text("unit").upper() or "DAY"

    if unit not in DATE_UNITS:
        self.unsupported(f"{unit}S_BETWEEN does not exist in SAP HANA.")
        return self.function_fallback_sql(expression)

    # DAYS_BETWEEN(<start>, <end>) counts forward from the first argument, which is the
    # opposite of the (end, start) order exp.DateDiff stores.
    return self.func(f"{unit}S_BETWEEN", expression.expression, expression.this)


class HanaGenerator(generator.Generator):
    # SAP HANA supports LIMIT but not FETCH FIRST, so a parsed FETCH is rewritten as a LIMIT.
    LIMIT_FETCH = "LIMIT"

    # CLUSTER BY / DISTRIBUTE BY / SORT BY are Hive-family extensions; assigning the
    # module-level dict drops them instead of emitting SQL HANA cannot parse.
    AFTER_HAVING_MODIFIER_TRANSFORMS = generator.AFTER_HAVING_MODIFIER_TRANSFORMS

    TYPE_MAPPING = {
        **generator.Generator.TYPE_MAPPING,
        # SAP HANA has no BINARY type — VARBINARY covers both.
        exp.DType.BINARY: "VARBINARY",
        exp.DType.DATETIME: "TIMESTAMP",
        # HANA's TEXT is a full-text search type, not a character type; NCLOB is the
        # unbounded string type. https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/data-types
        exp.DType.TEXT: "NCLOB",
        exp.DType.JSON: "NCLOB",
        exp.DType.JSONB: "NCLOB",
        exp.DType.TIMESTAMPTZ: "TIMESTAMP",
        exp.DType.UUID: "VARBINARY(16)",
    }

    TRANSFORMS = {
        **generator.Generator.TRANSFORMS,
        exp.DateAdd: _date_add_sql,
        exp.DateDiff: _date_diff_sql,
        exp.Extract: _extract_sql,
        exp.MD5: _md5_sql,
        exp.Repeat: _repeat_sql,
        exp.SHA2: _sha2_sql,
        exp.StrPosition: _strposition_sql,
        exp.TimeToStr: _to_varchar_sql,
        exp.ToChar: _to_varchar_sql,
        exp.TsOrDsAdd: _date_add_sql,
        exp.TsOrDsDiff: _date_diff_sql,
        # SAP HANA spells these without separators, unlike sqlglot's default rendering.
        # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/dayofmonth-function-datetime
        exp.DayOfMonth: rename_func("DAYOFMONTH"),
        exp.DayOfWeek: rename_func("DAYOFWEEK"),
        exp.DayOfYear: rename_func("DAYOFYEAR"),
        exp.CurrentTimestamp: lambda *_: "CURRENT_TIMESTAMP",
        exp.CurrentDate: lambda *_: "CURRENT_DATE",
        exp.CurrentTime: lambda *_: "CURRENT_TIME",
    }

    def offset_limit_modifiers(
        self, expression: exp.Expr, fetch: bool, limit: exp.Fetch | exp.Limit | None
    ) -> list[str]:
        offset = expression.args.get("offset")

        # SAP HANA parses OFFSET only as part of a LIMIT clause, so an offset with no limit
        # needs one synthesised rather than emitted bare.
        if offset and not limit:
            limit = exp.Limit(expression=exp.Literal.number(MAX_LIMIT))

        return [self.sql(limit), self.sql(expression, "offset")]
