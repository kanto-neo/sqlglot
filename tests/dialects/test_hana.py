from sqlglot import UnsupportedError, exp, parse_one
from sqlglot.errors import ErrorLevel
from sqlglot.dialects.hana import _TIME_ELEMENTS
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from tests.dialects.test_dialect import Validator


class TestHana(Validator):
    dialect = "hana"
    maxDiff = None

    def test_hana(self):
        self.validate_identity("SELECT 1 FROM DUMMY")
        self.validate_identity("SELECT * FROM t WHERE a = 1")
        self.validate_identity("SELECT a || b FROM t")
        self.validate_identity("SELECT COALESCE(a, b) FROM t")
        self.validate_identity("SELECT IFNULL(a, b) FROM t", "SELECT COALESCE(a, b) FROM t")
        self.validate_identity("SELECT CURRENT_TIMESTAMP")
        self.validate_identity("SELECT CURRENT_DATE")
        self.validate_identity("SELECT CURRENT_TIME")
        self.validate_identity("SELECT a FROM t1 EXCEPT SELECT a FROM t2")
        self.validate_identity("SELECT a FROM t1 INTERSECT SELECT a FROM t2")
        self.validate_identity('SELECT "MixedCase" FROM t')
        self.validate_identity("WITH x AS (SELECT a FROM t) SELECT a FROM x")
        self.validate_identity("SELECT ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) FROM t")
        self.validate_identity("SELECT CASE WHEN a > 1 THEN 'y' ELSE 'n' END FROM t")

        # CLUSTER BY / DISTRIBUTE BY / SORT BY are Hive-family extensions and must be dropped.
        self.validate_all(
            "SELECT * FROM t",
            read={"spark": "SELECT * FROM t CLUSTER BY y DISTRIBUTE BY x SORT BY z"},
        )

    def test_identifier_normalization(self):
        # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/identifiers
        self.assertEqual(
            normalize_identifiers(parse_one("SELECT a FROM tbl", read="hana"), dialect="hana").sql(
                "hana"
            ),
            "SELECT A FROM TBL",
        )
        # A quoted identifier keeps its case.
        self.assertEqual(
            normalize_identifiers(
                parse_one('SELECT "a" FROM tbl', read="hana"), dialect="hana"
            ).sql("hana"),
            'SELECT "a" FROM TBL',
        )

    def test_types(self):
        # HANA's TEXT is a full-text type, so the generic unbounded string type is NCLOB.
        self.validate_all(
            "SELECT CAST(a AS NCLOB) FROM t",
            read={
                "hana": "SELECT CAST(a AS NCLOB) FROM t",
                "postgres": "SELECT CAST(a AS TEXT) FROM t",
            },
            write={
                "hana": "SELECT CAST(a AS NCLOB) FROM t",
                "postgres": "SELECT CAST(a AS TEXT) FROM t",
            },
        )
        # SECONDDATE widens to TIMESTAMP on purpose. sqlglot's TIMESTAMP_S would preserve the
        # second granularity but is a DuckDB-only spelling, so every other target would emit an
        # unparseable TIMESTAMP_S. Assert a NON-duckdb target so the trade stays visible.
        self.validate_all(
            "SELECT CAST(a AS SECONDDATE) FROM t",
            write={
                "hana": "SELECT CAST(a AS TIMESTAMP) FROM t",
                "postgres": "SELECT CAST(a AS TIMESTAMP) FROM t",
                "oracle": "SELECT CAST(a AS TIMESTAMP) FROM t",
            },
        )
        self.validate_identity("SELECT CAST(a AS TIMESTAMP) FROM t")
        self.validate_identity("SELECT CAST(a AS NVARCHAR(10)) FROM t")
        self.validate_identity("SELECT CAST(a AS VARBINARY(10)) FROM t")
        self.validate_identity("SELECT CAST(a AS DECIMAL(10, 2)) FROM t")
        self.validate_identity("SELECT CAST(a AS BOOLEAN) FROM t")

        # SMALLDECIMAL is a variable-precision decimal float; no sqlglot type models it, so the
        # round trip deliberately degrades to DECIMAL rather than leaking a HANA-only name.
        self.validate_identity(
            "SELECT CAST(a AS SMALLDECIMAL) FROM t", "SELECT CAST(a AS DECIMAL) FROM t"
        )
        self.validate_identity(
            "SELECT CAST(a AS ALPHANUM) FROM t", "SELECT CAST(a AS VARCHAR) FROM t"
        )

        # SAP HANA has no BINARY type, and a length-less VARBINARY is one byte, so an
        # unbounded binary type becomes BLOB. See test_binary_types.
        self.validate_all(
            "SELECT CAST(a AS BLOB) FROM t",
            read={"postgres": "SELECT CAST(a AS BYTEA) FROM t"},
            write={"hana": "SELECT CAST(a AS BLOB) FROM t"},
        )
        for source, expected in (
            ("DATETIME", "TIMESTAMP"),
            ("TIMESTAMPTZ", "TIMESTAMP"),
            ("JSON", "NCLOB"),
            ("UUID", "VARBINARY(16)"),
        ):
            with self.subTest(type=source):
                self.assertEqual(
                    parse_one(f"SELECT CAST(a AS {source})").sql("hana"),
                    f"SELECT CAST(a AS {expected})",
                )

    def test_offset_requires_limit(self):
        # SAP HANA parses OFFSET only as part of a LIMIT clause, so an offset-only query needs a
        # limit synthesised; 2147384648 is the sentinel sqlalchemy-hana uses, not an engine bound.
        self.validate_all(
            "SELECT * FROM t LIMIT 2147384648 OFFSET 10",
            read={"postgres": "SELECT * FROM t OFFSET 10"},
            write={"hana": "SELECT * FROM t LIMIT 2147384648 OFFSET 10"},
        )
        self.validate_identity("SELECT * FROM t LIMIT 5 OFFSET 10")
        self.validate_identity("SELECT * FROM t LIMIT 5")
        # A plain LIMIT must NOT gain an OFFSET.
        self.validate_all(
            "SELECT * FROM t LIMIT 5",
            read={"postgres": "SELECT * FROM t LIMIT 5"},
            write={"hana": "SELECT * FROM t LIMIT 5"},
        )
        # FETCH FIRST is rewritten as LIMIT, which HANA does support.
        self.validate_all(
            "SELECT * FROM t LIMIT 3",
            read={"oracle": "SELECT * FROM t FETCH FIRST 3 ROWS ONLY"},
            write={"hana": "SELECT * FROM t LIMIT 3"},
        )

    def test_locking_reads(self):
        # HANA spells the shared lock FOR SHARE LOCK and SKIP LOCKED as IGNORE LOCKED.
        self.validate_identity("SELECT * FROM t FOR UPDATE")
        self.validate_identity("SELECT * FROM t FOR SHARE LOCK")
        self.validate_identity("SELECT * FROM t FOR UPDATE OF a")
        self.validate_all(
            "SELECT * FROM t FOR UPDATE",
            read={"postgres": "SELECT * FROM t FOR UPDATE"},
            write={"hana": "SELECT * FROM t FOR UPDATE"},
        )
        self.validate_all(
            "SELECT * FROM t FOR SHARE LOCK",
            read={"postgres": "SELECT * FROM t FOR SHARE"},
            write={"hana": "SELECT * FROM t FOR SHARE LOCK"},
        )

    def test_date_arithmetic(self):
        # ADD_<unit>S / <unit>S_BETWEEN exist for YEAR, MONTH, DAY and SECOND only.
        for unit in ("YEAR", "MONTH", "DAY", "SECOND"):
            with self.subTest(unit=unit):
                self.validate_identity(f"SELECT ADD_{unit}S(d, 3) FROM t")
                self.validate_identity(f"SELECT {unit}S_BETWEEN(a, b) FROM t")

        self.validate_all(
            "SELECT DATE_ADD(d, 3, 'DAY') FROM t",
            write={"hana": "SELECT ADD_DAYS(d, 3) FROM t"},
        )

        # DAYS_BETWEEN(<date_1>, <date_2>) evaluates to date_2 - date_1, the opposite argument
        # order to DATEDIFF(<end>, <start>) — so the two swap on the way through.
        self.validate_all(
            "SELECT DAYS_BETWEEN(d1, d2) FROM t",
            read={"hana": "SELECT DAYS_BETWEEN(d1, d2) FROM t"},
            write={
                "hana": "SELECT DAYS_BETWEEN(d1, d2) FROM t",
                "tsql": "SELECT DATEDIFF(DAY, d1, d2) FROM t",
            },
        )

        # There is no ADD_WEEKS / WEEKS_BETWEEN in SAP HANA, on either the add or the diff side.
        self.validate_all(
            "SELECT DATE_ADD(d, 1, 'WEEK') FROM t",
            write={"hana": UnsupportedError},
        )
        with self.assertRaises(UnsupportedError):
            parse_one("SELECT DATEDIFF(WEEK, a, b) FROM t", read="tsql").sql(
                "hana", unsupported_level=ErrorLevel.RAISE
            )

        # Subtraction has no SUBTRACT_<unit>S family; the offset is negated instead.
        self.validate_all(
            "SELECT ADD_DAYS(d, -3) FROM t",
            read={"mysql": "SELECT DATE_SUB(d, INTERVAL 3 DAY) FROM t"},
            write={"hana": "SELECT ADD_DAYS(d, -3) FROM t"},
        )

    def test_hour_and_minute_are_scaled_to_seconds(self):
        # HANA has no ADD_HOURS / ADD_MINUTES / HOURS_BETWEEN / MINUTES_BETWEEN, so those units
        # go through seconds. A literal offset is folded.
        self.validate_all(
            "SELECT ADD_SECONDS(d, 10800) FROM t",
            read={"mysql": "SELECT DATE_ADD(d, INTERVAL 3 HOUR) FROM t"},
            write={"hana": "SELECT ADD_SECONDS(d, 10800) FROM t"},
        )
        self.validate_all(
            "SELECT ADD_SECONDS(d, 120) FROM t",
            read={"tsql": "SELECT DATEADD(MINUTE, 2, d) FROM t"},
            write={"hana": "SELECT ADD_SECONDS(d, 120) FROM t"},
        )
        # A compound offset must be parenthesized before it is scaled.
        self.validate_all(
            "SELECT ADD_SECONDS(d, (a + b) * 60) FROM t",
            read={"mysql": "SELECT DATE_ADD(d, INTERVAL (a + b) MINUTE) FROM t"},
            write={"hana": "SELECT ADD_SECONDS(d, (a + b) * 60) FROM t"},
        )
        # The diff side divides and casts, because HANA's `/` is true division.
        self.validate_all(
            "SELECT CAST(SECONDS_BETWEEN(a, b) / 60 AS BIGINT) FROM t",
            write={"hana": "SELECT CAST(SECONDS_BETWEEN(a, b) / 60 AS BIGINT) FROM t"},
            read={"hana": "SELECT CAST(SECONDS_BETWEEN(a, b) / 60 AS BIGINT) FROM t"},
        )

        # ADD_HOURS and friends must never be emitted, from any source dialect.
        for sql, read in (
            ("SELECT DATE_ADD(d, INTERVAL 3 HOUR) FROM t", "mysql"),
            ("SELECT DATEADD(HOUR, 3, d) FROM t", "tsql"),
            ("SELECT DATEDIFF(HOUR, a, b) FROM t", "tsql"),
            ("SELECT DATEDIFF(MINUTE, a, b) FROM t", "tsql"),
        ):
            with self.subTest(sql=sql):
                out = parse_one(sql, read=read).sql("hana")
                self.assertNotIn("ADD_HOURS", out)
                self.assertNotIn("ADD_MINUTES", out)
                self.assertNotIn("HOURS_BETWEEN", out)
                self.assertNotIn("MINUTES_BETWEEN", out)

    def test_extract(self):
        # SAP HANA's EXTRACT accepts exactly six parts; those stay native.
        for part in ("YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"):
            with self.subTest(part=part):
                self.validate_identity(f"SELECT EXTRACT({part} FROM d) FROM t")

        # Everything else HANA exposes as a scalar function instead.
        self.validate_all(
            "SELECT DAYOFMONTH(d) FROM t",
            read={"hana": "SELECT EXTRACT(DAYOFMONTH FROM d) FROM t"},
            write={"hana": "SELECT DAYOFMONTH(d) FROM t"},
        )
        self.validate_identity("SELECT DAYOFYEAR(d) FROM t")
        self.validate_identity("SELECT WEEK(d) FROM t")
        self.validate_identity("SELECT ISOWEEK(d) FROM t")
        self.validate_all(
            "SELECT DAYOFYEAR(d) FROM t",
            read={"postgres": "SELECT EXTRACT(DOY FROM d) FROM t"},
            write={"hana": "SELECT DAYOFYEAR(d) FROM t"},
        )
        # An unmappable part warns rather than silently emitting a bogus function.
        self.validate_all(
            "SELECT EXTRACT(EPOCH FROM d) FROM t",
            write={"hana": UnsupportedError},
        )

    def test_day_of_week(self):
        # HANA has no DAYOFWEEK. WEEKDAY is Monday(0)..Sunday(6), whereas exp.DayOfWeek carries
        # the Sunday(1)..Saturday(7) convention, so the value is rotated rather than renamed.
        self.validate_identity("SELECT (MOD((WEEKDAY(d) + 1), 7) + 1) FROM t")
        for read, sql in (("mysql", "SELECT DAYOFWEEK(d) FROM t"),):
            with self.subTest(read=read):
                out = parse_one(sql, read=read).sql("hana")
                self.assertIn("WEEKDAY", out)
                self.assertNotIn("DAYOFWEEK", out)
        # The ISO variant is 1-based from Monday, so it only needs the +1.
        for read, sql in (
            ("duckdb", "SELECT ISODOW(d) FROM t"),
            ("snowflake", "SELECT DAYOFWEEKISO(d) FROM t"),
        ):
            with self.subTest(read=read):
                out = parse_one(sql, read=read).sql("hana")
                self.assertIn("WEEKDAY", out)
                self.assertNotIn("DAYOFWEEKISO", out)

    def test_string_functions(self):
        # SAP HANA has no REPEAT; RPAD padded with the string itself is the documented idiom.
        self.validate_all(
            "SELECT RPAD(x, 3 * LENGTH(x), x) FROM t",
            read={"postgres": "SELECT REPEAT(x, 3) FROM t"},
            write={"hana": "SELECT RPAD(x, 3 * LENGTH(x), x) FROM t"},
        )
        # A compound repeat count must be parenthesized before it is multiplied by LENGTH.
        self.validate_all(
            "SELECT RPAD(x, (a + b) * LENGTH(x), x) FROM t",
            read={"postgres": "SELECT REPEAT(x, a + b) FROM t"},
            write={"hana": "SELECT RPAD(x, (a + b) * LENGTH(x), x) FROM t"},
        )
        # A REPEAT with no count reaches the guard rather than raising AttributeError.
        with self.assertRaises(UnsupportedError):
            exp.select(exp.Repeat(this=exp.column("x"))).sql(
                "hana", unsupported_level=ErrorLevel.RAISE
            )

        # LOCATE(<haystack>, <needle>) takes the haystack first — the reverse of MySQL's LOCATE.
        self.validate_all(
            "SELECT LOCATE(haystack, needle)",
            read={
                "hana": "SELECT LOCATE(haystack, needle)",
                "postgres": "SELECT POSITION(needle IN haystack)",
            },
            write={
                "hana": "SELECT LOCATE(haystack, needle)",
                "mysql": "SELECT LOCATE(needle, haystack)",
            },
        )
        # All four arities round-trip; the occurrence argument must not be dropped.
        self.validate_identity("SELECT LOCATE(haystack, needle, 3)")
        self.validate_identity("SELECT LOCATE(haystack, needle, 3, 2)")
        self.validate_all(
            "SELECT LOCATE(haystack, needle, 3, 2)",
            read={"bigquery": "SELECT INSTR(haystack, needle, 3, 2)"},
            write={
                "hana": "SELECT LOCATE(haystack, needle, 3, 2)",
                "bigquery": "SELECT INSTR(haystack, needle, 3, 2)",
            },
        )

        # HANA has no CONCAT/CONCAT_WS of the variadic shape; || is the concatenation operator.
        self.validate_identity("SELECT CONCAT(a, b) FROM t", "SELECT a || b FROM t")
        self.validate_all(
            "SELECT COALESCE(a, '') || COALESCE(b, '') || COALESCE(c, '') FROM t",
            read={"postgres": "SELECT CONCAT(a, b, c) FROM t"},
        )
        self.assertNotIn(
            "CONCAT_WS",
            parse_one("SELECT CONCAT_WS('-', a, b) FROM t", read="postgres").sql("hana"),
        )

        # TRIM's ANSI form is supported in all four variants.
        self.validate_identity("SELECT TRIM(a) FROM t")
        self.validate_identity(
            "SELECT TRIM('x' FROM a) FROM t", "SELECT LTRIM(RTRIM(a, 'x'), 'x') FROM t"
        )
        self.validate_identity(
            "SELECT TRIM(BOTH 'x' FROM a) FROM t", "SELECT LTRIM(RTRIM(a, 'x'), 'x') FROM t"
        )
        # LEADING/TRAILING become HANA's own LTRIM/RTRIM, which carry the set semantics.
        self.validate_identity(
            "SELECT TRIM(LEADING 'x' FROM a) FROM t", "SELECT LTRIM(a, 'x') FROM t"
        )
        self.validate_identity(
            "SELECT TRIM(TRAILING 'x' FROM a) FROM t", "SELECT RTRIM(a, 'x') FROM t"
        )

    def test_mod(self):
        # HANA's arithmetic operators are only unary -, +, -, *, / — modulo is a function.
        self.validate_identity("SELECT MOD(a, 2) FROM t")
        self.validate_all(
            "SELECT MOD(a, 2) FROM t",
            read={"postgres": "SELECT a % 2 FROM t"},
            write={"hana": "SELECT MOD(a, 2) FROM t"},
        )
        self.assertNotIn("%", parse_one("SELECT a % 2 FROM t", read="postgres").sql("hana"))

    def test_group_concat(self):
        # HANA's string aggregate is STRING_AGG; it has no GROUP_CONCAT.
        self.validate_identity("SELECT STRING_AGG(a, ',') FROM t")
        self.validate_identity("SELECT STRING_AGG(a, ',' ORDER BY a) FROM t")
        self.validate_all(
            "SELECT STRING_AGG(a, ',') FROM t",
            read={"mysql": "SELECT GROUP_CONCAT(a SEPARATOR ',') FROM t"},
            write={"hana": "SELECT STRING_AGG(a, ',') FROM t"},
        )
        self.assertNotIn(
            "GROUP_CONCAT",
            parse_one("SELECT GROUP_CONCAT(a) FROM t", read="mysql").sql("hana"),
        )

    def test_hash_functions(self):
        # HASH_MD5 / HASH_SHA256 take a BINARY argument, hence the TO_BINARY wrapper.
        # The hex-wrapping round trip lives in test_hash_returns_hex; here only the bare HANA
        # spellings, which stay anonymous and therefore an identity.
        self.validate_identity("SELECT HASH_MD5(TO_BINARY(x)) FROM t")
        self.validate_identity("SELECT HASH_SHA256(TO_BINARY(x)) FROM t")
        # HASH_MD5 and HASH_SHA256 are the only hash functions HANA documents.
        for length in (384, 512):
            with self.subTest(length=length):
                self.validate_all(
                    f"SELECT SHA2(x, {length}) FROM t",
                    write={"hana": UnsupportedError},
                )
        # Left anonymous, HANA's own spellings stay an identity.
        self.validate_identity("SELECT HASH_MD5(TO_BINARY(x)) FROM t")
        self.validate_identity("SELECT HASH_SHA256(TO_BINARY(x)) FROM t")

    def test_to_varchar_and_parsing(self):
        # TO_VARCHAR doubles as the datetime formatter; TO_DATE / TO_TIMESTAMP are the inbound
        # direction. Without the latter, HANA format models leak out as strftime strings.
        self.validate_identity("SELECT TO_VARCHAR(d, 'YYYY-MM-DD') FROM t")
        self.validate_identity("SELECT TO_DATE(x, 'YYYY-MM-DD') FROM t")
        self.validate_identity("SELECT TO_TIMESTAMP(x, 'YYYY-MM-DD HH24:MI:SS') FROM t")
        # A one-argument TO_TIMESTAMP is a conversion, not a parse, so it must not gain a format.
        self.validate_identity("SELECT TO_TIMESTAMP(x) FROM t")
        self.validate_identity("SELECT TO_DATE(x) FROM t")

        self.validate_all(
            "SELECT TO_TIMESTAMP(x, 'YYYY-MM-DD') FROM t",
            read={"duckdb": "SELECT STRPTIME(x, '%Y-%m-%d') FROM t"},
            write={"hana": "SELECT TO_TIMESTAMP(x, 'YYYY-MM-DD') FROM t"},
        )
        self.validate_all(
            "SELECT TO_VARCHAR(CAST(d AS DATE), 'YYYY-MM-DD') FROM t",
            write={
                "hana": "SELECT TO_VARCHAR(CAST(d AS DATE), 'YYYY-MM-DD') FROM t",
                "duckdb": "SELECT STRFTIME(CAST(d AS DATE), '%Y-%m-%d') FROM t",
            },
        )
        for name in ("STR_TO_DATE", "STR_TO_TIME", "TIME_STR_TO_TIME"):
            with self.subTest(name=name):
                out = parse_one("SELECT STRPTIME(x, '%Y-%m-%d') FROM t", read="duckdb").sql("hana")
                self.assertNotIn(name, out)

    def test_time_mapping(self):
        # Every format element must convert in BOTH directions: TIME_MAPPING drives hana -> other,
        # INVERSE_TIME_MAPPING drives other -> hana. A one-way test leaves the inverse unpinned.
        for element, strftime in _TIME_ELEMENTS.items():
            with self.subTest(element=element):
                out = parse_one(
                    f"SELECT TO_VARCHAR(CAST(d AS DATE), '{element}')", read="hana"
                ).sql("duckdb")
                self.assertIn(strftime, out, f"{element} did not map to {strftime}")

        # ... and back. FF7 (not FF6) is the canonical spelling for %f, DAY for %A, WW for %W.
        for strftime, element in (
            ("%f", "FF7"),
            ("%A", "DAY"),
            ("%W", "WW"),
            ("%Y", "YYYY"),
            ("%H", "HH24"),
            ("%M", "MI"),
        ):
            with self.subTest(strftime=strftime):
                out = parse_one(
                    f"SELECT STRFTIME(CAST(d AS DATE), '{strftime}')", read="duckdb"
                ).sql("hana")
                self.assertIn(element, out, f"{strftime} did not map back to {element}")

        # HANA matches format elements case-insensitively, so lower and title case parse too.
        for spelling in ("yyyy-mm-dd", "Yyyy-Mm-Dd"):
            with self.subTest(spelling=spelling):
                self.assertIn(
                    "%Y",
                    parse_one(f"SELECT TO_VARCHAR(CAST(d AS DATE), '{spelling}')", read="hana").sql(
                        "duckdb"
                    ),
                )

    def test_trim_uses_set_semantics(self):
        # HANA's LTRIM/RTRIM second argument is a character SET, not a search string:
        # LTRIM('babababAabend', 'ab') is 'Aabend'. The shared ANSI trim_sql helper would emit
        # TRIM(LEADING 'ab' FROM ...), which removes a SUBSTRING and returns a different value.
        # A single-character set is the one case where the two agree, so it cannot be the only test.
        self.validate_all(
            "SELECT LTRIM('babababAabend', 'ab')",
            read={"postgres": "SELECT LTRIM('babababAabend', 'ab')"},
            write={"hana": "SELECT LTRIM('babababAabend', 'ab')"},
        )
        self.validate_all(
            "SELECT RTRIM(a, 'xyz') FROM t",
            read={"postgres": "SELECT RTRIM(a, 'xyz') FROM t"},
            write={"hana": "SELECT RTRIM(a, 'xyz') FROM t"},
        )
        # BOTH has no two-argument TRIM in HANA, so it nests instead.
        self.validate_all(
            "SELECT LTRIM(RTRIM(x, 'ab'), 'ab')",
            read={"bigquery": "SELECT TRIM(x, 'ab')"},
            write={"hana": "SELECT LTRIM(RTRIM(x, 'ab'), 'ab')"},
        )
        for sql in ("SELECT LTRIM(a, 'ab') FROM t", "SELECT RTRIM(a, 'ab') FROM t"):
            with self.subTest(sql=sql):
                self.assertNotIn("TRIM(LEADING", parse_one(sql, read="hana").sql("hana"))
                self.assertNotIn("TRIM(TRAILING", parse_one(sql, read="hana").sql("hana"))

    def test_hash_returns_hex(self):
        # HASH_MD5/HASH_SHA256 take BINARY and RETURN VARBINARY, but exp.MD5/exp.SHA2 denote the
        # hex digest, so BINTOHEX is needed on the way out as well as TO_BINARY on the way in.
        self.validate_all(
            "SELECT BINTOHEX(HASH_MD5(TO_BINARY(x))) FROM t",
            read={"postgres": "SELECT MD5(x) FROM t"},
            write={"hana": "SELECT BINTOHEX(HASH_MD5(TO_BINARY(x))) FROM t"},
        )
        self.validate_all(
            "SELECT BINTOHEX(HASH_SHA256(TO_BINARY(x))) FROM t",
            read={"postgres": "SELECT SHA256(x) FROM t"},
            write={"hana": "SELECT BINTOHEX(HASH_SHA256(TO_BINARY(x))) FROM t"},
        )

    def test_binary_types(self):
        # A VARBINARY with no length is a ONE-BYTE column in HANA, so unbounded binary is BLOB.
        self.validate_all(
            "CREATE TABLE t (b BLOB)",
            read={"postgres": "CREATE TABLE t (b BYTEA)", "hana": "CREATE TABLE t (b BLOB)"},
            write={"hana": "CREATE TABLE t (b BLOB)"},
        )
        # An explicit length is preserved rather than routed to BLOB.
        self.validate_identity("SELECT CAST(a AS VARBINARY(10)) FROM t")
        # Hex literals must not be swallowed: `0x0abc` once parsed as `0 AS x0abc`.
        self.validate_identity("SELECT x'0abc'")
        self.validate_identity("SELECT 0x0abc", "SELECT x'0abc'")

    def test_constructs_hana_lacks(self):
        # Each of these has no HANA grammar and must be rewritten, not emitted verbatim.
        self.validate_all(
            "SELECT CAST(a AS INT)",
            read={"tsql": "SELECT TRY_CAST(a AS INT)"},
            write={"hana": "SELECT CAST(a AS INT)"},
        )
        self.validate_all(
            "SELECT a FROM t WHERE LOWER(b) LIKE LOWER('x%')",
            read={"postgres": "SELECT a FROM t WHERE b ILIKE 'x%'"},
            write={"hana": "SELECT a FROM t WHERE LOWER(b) LIKE LOWER('x%')"},
        )
        for name, sql, read in (
            ("qualify", "SELECT a FROM t QUALIFY ROW_NUMBER() OVER (ORDER BY b) = 1", "snowflake"),
            ("semi join", "SELECT * FROM a LEFT SEMI JOIN b ON a.x = b.x", "spark"),
            ("distinct on", "SELECT DISTINCT ON (a) a, b FROM t", "postgres"),
        ):
            with self.subTest(construct=name):
                out = parse_one(sql, read=read).sql("hana")
                for banned in ("QUALIFY", "SEMI JOIN", "DISTINCT ON", "TRY_CAST", "ILIKE"):
                    self.assertNotIn(banned, out)

    def test_quarter_is_an_integer(self):
        # HANA's QUARTER() returns the string 'YYYY-Qn' and EXTRACT(QUARTER ...) is a syntax
        # error, so an integer quarter has to be computed. FLOOR matters: HANA's / is true
        # division, so without it two months per quarter would yield a fraction.
        self.validate_all(
            "SELECT (FLOOR((MONTH(d) - 1) / 3) + 1) FROM t",
            read={"postgres": "SELECT EXTRACT(QUARTER FROM d) FROM t"},
            write={"hana": "SELECT (FLOOR((MONTH(d) - 1) / 3) + 1) FROM t"},
        )
        self.assertNotIn("QUARTER", parse_one("SELECT QUARTER(d)", read="mysql").sql("hana"))
        # ISOWEEK also returns a string in HANA, so it has no integer equivalent and must warn.
        self.validate_all(
            "SELECT EXTRACT(ISOWEEK FROM d) FROM t",
            write={"hana": UnsupportedError},
        )

    def test_group_concat_delimiter(self):
        # HANA's STRING_AGG documents no default delimiter, so one must not be invented.
        self.validate_identity("SELECT STRING_AGG(a) FROM t")
        self.assertNotIn("','", parse_one("SELECT STRING_AGG(a) FROM t", read="hana").sql("hana"))
        # DISTINCT has no place in HANA's STRING_AGG grammar.
        self.validate_all(
            "SELECT GROUP_CONCAT(DISTINCT a) FROM t",
            write={"hana": UnsupportedError},
        )

    def test_full_date_delta_family(self):
        # Every add/diff sibling must route through the HANA helpers; an unmapped one leaks a
        # sqlglot-internal name such as TIME_ADD / DATETIME_DIFF / TIMESTAMPDIFF.
        for sql, read in (
            ("SELECT TIMESTAMPDIFF(SECOND, a, b)", "mysql"),
            ("SELECT DATETIME_DIFF(a, b, SECOND)", "bigquery"),
            ("SELECT TIME_ADD(t, INTERVAL 1 HOUR)", "bigquery"),
            ("SELECT DATETIME_ADD(d, INTERVAL 1 DAY)", "bigquery"),
        ):
            with self.subTest(sql=sql):
                out = parse_one(sql, read=read).sql("hana")
                for banned in (
                    "TIME_ADD",
                    "TIME_DIFF",
                    "DATETIME_ADD",
                    "DATETIME_DIFF",
                    "TIMESTAMPDIFF",
                    "TIMESTAMPADD",
                ):
                    self.assertNotIn(banned, out, f"{sql} leaked {banned}: {out}")
