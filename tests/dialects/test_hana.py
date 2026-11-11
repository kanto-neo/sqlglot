from sqlglot import UnsupportedError, exp, parse_one
from sqlglot.errors import ErrorLevel
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from tests.dialects.test_dialect import Validator


class TestHana(Validator):
    dialect = "hana"
    maxDiff = None

    def assert_unsupported(self, sql, read):
        with self.subTest(sql=sql), self.assertRaises(UnsupportedError):
            parse_one(sql, read=read).sql("hana", unsupported_level=ErrorLevel.RAISE)

    def test_hana(self):
        self.validate_identity("SELECT 1 FROM DUMMY")
        self.validate_identity("SELECT * FROM t WHERE a = 1")
        self.validate_identity("SELECT a || b FROM t")
        self.validate_identity("SELECT IFNULL(a, b) FROM t", "SELECT COALESCE(a, b) FROM t")
        self.validate_identity("SELECT a FROM t1 EXCEPT SELECT a FROM t2")
        self.validate_identity("SELECT a FROM t1 INTERSECT SELECT a FROM t2")
        self.validate_identity('SELECT "MixedCase" FROM t')
        self.validate_identity('SELECT "top" FROM t')
        self.validate_identity("WITH x AS (SELECT a FROM t) SELECT a FROM x")
        self.validate_identity("WITH x(a, b) AS (SELECT 1, 2 FROM DUMMY) SELECT a, b FROM x")
        self.validate_identity("SELECT ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) FROM t")
        self.validate_identity("SELECT MOD(a, 2) FROM t")
        self.validate_identity("SELECT CARDINALITY(a) FROM t")
        self.validate_identity("SELECT CURRENT_USER FROM DUMMY")
        self.validate_identity("SELECT x'0abc' FROM DUMMY")
        self.validate_identity("SELECT 0x0abc FROM DUMMY", "SELECT x'0abc' FROM DUMMY")
        self.validate_identity("SELECT TOP 5 a FROM t", "SELECT a FROM t LIMIT 5")
        self.validate_identity("SELECT TOP 5 DISTINCT a FROM t", "SELECT DISTINCT a FROM t LIMIT 5")
        self.validate_identity("SELECT * FROM t TABLESAMPLE SYSTEM (10)")

        self.validate_all(
            "SELECT 1 FROM DUMMY",
            read={"postgres": "SELECT 1", "oracle": "SELECT 1 FROM DUAL"},
        )
        self.validate_all(
            "SELECT x FROM (SELECT 1 AS x FROM DUMMY UNION ALL SELECT 2 FROM DUMMY) AS s",
            read={"postgres": "SELECT x FROM (SELECT 1 AS x UNION ALL SELECT 2) AS s"},
        )
        self.validate_all(
            "INSERT INTO t SELECT 1 FROM DUMMY",
            read={"postgres": "INSERT INTO t SELECT 1"},
        )
        self.validate_all(
            "SELECT * FROM t",
            read={"spark": "SELECT * FROM t CLUSTER BY y DISTRIBUTE BY x SORT BY z"},
        )
        self.validate_all(
            "SELECT MOD(a, 2) FROM t",
            read={"postgres": "SELECT a % 2 FROM t"},
        )
        self.validate_all(
            "SELECT CARDINALITY(a) FROM t",
            read={"bigquery": "SELECT ARRAY_LENGTH(a) FROM t"},
            write={"presto": "SELECT CARDINALITY(a) FROM t"},
        )
        self.validate_all(
            "SELECT CURRENT_USER FROM DUMMY",
            read={"postgres": "SELECT CURRENT_USER"},
        )
        self.validate_all(
            "SELECT * FROM t TABLESAMPLE BERNOULLI (10)",
            read={"postgres": "SELECT * FROM t TABLESAMPLE BERNOULLI (10)"},
            write={"postgres": "SELECT * FROM t TABLESAMPLE BERNOULLI (10)"},
        )
        self.validate_all(
            "SELECT CAST(a AS INT) FROM t",
            read={"tsql": "SELECT TRY_CAST(a AS INT) FROM t"},
        )
        self.validate_all(
            "SELECT a FROM t WHERE LOWER(b) LIKE LOWER('x%')",
            read={"postgres": "SELECT a FROM t WHERE b ILIKE 'x%'"},
        )
        self.validate_all(
            "SELECT a FROM (SELECT a, ROW_NUMBER() OVER (ORDER BY b NULLS LAST) AS _w FROM t) AS _t WHERE _w = 1",
            read={"snowflake": "SELECT a FROM t QUALIFY ROW_NUMBER() OVER (ORDER BY b) = 1"},
        )

    def test_identifier_normalization(self):
        for sql, expected in (
            ("SELECT a FROM tbl", "SELECT A FROM TBL"),
            ('SELECT "a" FROM tbl', 'SELECT "a" FROM TBL'),
        ):
            with self.subTest(sql=sql):
                tree = normalize_identifiers(parse_one(sql, read="hana"), dialect="hana")
                self.assertEqual(tree.sql("hana"), expected)

    def test_types(self):
        self.validate_identity("SELECT CAST(a AS TIMESTAMP) FROM t")
        self.validate_identity("SELECT CAST(a AS NVARCHAR(10)) FROM t")
        self.validate_identity("SELECT CAST(a AS VARBINARY(10)) FROM t")
        self.validate_identity("SELECT CAST(a AS VARBINARY) FROM t")
        self.validate_identity("SELECT CAST(a AS DECIMAL(10, 2)) FROM t")
        self.validate_identity("SELECT CAST(a AS BOOLEAN) FROM t")
        self.validate_identity("SELECT CAST(a AS TINYINT) FROM t")
        self.validate_identity("SELECT CAST(a AS REAL) FROM t")
        self.validate_identity("SELECT CAST(a AS DOUBLE) FROM t")
        self.validate_identity("SELECT CAST(a AS FLOAT) FROM t", "SELECT CAST(a AS DOUBLE) FROM t")
        self.validate_identity(
            "SELECT CAST(a AS NCLOB) FROM t", "SELECT CAST(a AS NVARCHAR) FROM t"
        )
        self.validate_identity(
            "SELECT CAST(a AS SECONDDATE) FROM t", "SELECT CAST(a AS TIMESTAMP) FROM t"
        )
        self.validate_identity(
            "SELECT CAST(a AS SMALLDECIMAL) FROM t", "SELECT CAST(a AS DECIMAL) FROM t"
        )
        self.validate_identity(
            "SELECT CAST(a AS ALPHANUM) FROM t", "SELECT CAST(a AS VARCHAR) FROM t"
        )
        self.validate_identity("CREATE TABLE t (a FLOAT(10), b REAL, c DOUBLE, d NCLOB, e BLOB)")

        self.validate_all(
            "CREATE TABLE t (a NCLOB, b BLOB, c REAL, d DOUBLE)",
            read={"postgres": "CREATE TABLE t (a TEXT, b BYTEA, c REAL, d DOUBLE PRECISION)"},
            write={"postgres": "CREATE TABLE t (a TEXT, b BYTEA, c REAL, d DOUBLE PRECISION)"},
        )
        self.validate_all(
            "SELECT CAST(a AS NVARCHAR), CAST(b AS VARBINARY) FROM t",
            read={"postgres": "SELECT CAST(a AS TEXT), CAST(b AS BYTEA) FROM t"},
        )
        self.validate_all(
            "SELECT CAST(a AS TIMESTAMP), CAST(b AS TIMESTAMP), CAST(c AS NCLOB), CAST(d AS VARBINARY(16)) FROM t",
            read={
                "duckdb": "SELECT CAST(a AS DATETIME), CAST(b AS TIMESTAMPTZ), CAST(c AS JSON), CAST(d AS UUID) FROM t"
            },
        )
        self.validate_all(
            "SELECT CAST(a AS TIMESTAMP) FROM t",
            read={"hana": "SELECT CAST(a AS SECONDDATE) FROM t"},
            write={"postgres": "SELECT CAST(a AS TIMESTAMP) FROM t"},
        )
        self.validate_all(
            "SELECT CAST(a AS TINYINT) FROM t",
            read={"duckdb": "SELECT CAST(a AS UTINYINT) FROM t"},
        )

    def test_conversion_functions(self):
        self.validate_identity("SELECT TO_DATE(x, 'YYYY-MM-DD') FROM t")
        self.validate_identity("SELECT TO_TIMESTAMP(x, 'YYYY-MM-DD HH24:MI:SS') FROM t")
        self.validate_identity("SELECT TO_VARCHAR(d, 'YYYY-MM-DD') FROM t")

        self.validate_all(
            "SELECT CAST(x AS DATE), CAST(x AS TIMESTAMP) FROM t",
            read={"hana": "SELECT TO_DATE(x), TO_TIMESTAMP(x) FROM t"},
            write={"postgres": "SELECT CAST(x AS DATE), CAST(x AS TIMESTAMP) FROM t"},
        )
        self.validate_all(
            "SELECT CAST(a AS INT), CAST(a AS INT), CAST(a AS BIGINT), CAST(a AS SMALLINT), CAST(a AS DOUBLE), CAST(a AS DECIMAL(10, 2)) FROM t",
            read={
                "hana": "SELECT TO_INTEGER(a), TO_INT(a), TO_BIGINT(a), TO_SMALLINT(a), TO_DOUBLE(a), TO_DECIMAL(a, 10, 2) FROM t"
            },
        )
        self.validate_all(
            "SELECT TO_TIMESTAMP(x, 'YYYY-MM-DD') FROM t",
            read={"duckdb": "SELECT STRPTIME(x, '%Y-%m-%d') FROM t"},
        )
        self.validate_all(
            "SELECT TO_VARCHAR(CAST(d AS TIMESTAMP), 'YYYY-MM-DD HH24:MI:SS.FF6') FROM t",
            read={"duckdb": "SELECT STRFTIME(CAST(d AS TIMESTAMP), '%Y-%m-%d %H:%M:%S.%f') FROM t"},
            write={
                "duckdb": "SELECT STRFTIME(CAST(d AS TIMESTAMP), '%Y-%m-%d %H:%M:%S.%f') FROM t"
            },
        )
        self.validate_all(
            "SELECT TO_VARCHAR(CAST(d AS TIMESTAMP), 'HH12:MI AM, DAY DD MON YYYY') FROM t",
            read={
                "hana": "SELECT TO_VARCHAR(CAST(d AS TIMESTAMP), 'hh:mi am, Day dd Mon yyyy') FROM t"
            },
            write={
                "duckdb": "SELECT STRFTIME(CAST(d AS TIMESTAMP), '%I:%M %p, %A %d %b %Y') FROM t"
            },
        )

    def test_limit(self):
        self.validate_identity("SELECT * FROM t LIMIT 5")
        self.validate_identity("SELECT * FROM t LIMIT 5 OFFSET 10")

        self.validate_all(
            "SELECT * FROM t LIMIT 2147384648 OFFSET 10",
            read={"postgres": "SELECT * FROM t OFFSET 10"},
        )
        self.validate_all(
            "SELECT * FROM t LIMIT 3",
            read={
                "oracle": "SELECT * FROM t FETCH FIRST 3 ROWS ONLY",
                "tsql": "SELECT TOP 3 * FROM t",
            },
        )
        self.assert_unsupported("SELECT * FROM t FETCH FIRST 5 ROWS WITH TIES", "oracle")
        self.assert_unsupported("SELECT TOP 10 PERCENT a FROM t", "tsql")

    def test_locking_reads(self):
        self.validate_identity("SELECT * FROM t FOR UPDATE")
        self.validate_identity("SELECT * FROM t FOR UPDATE OF a NOWAIT")
        self.validate_identity("SELECT * FROM t FOR UPDATE WAIT 5")
        self.validate_identity("SELECT * FROM t FOR SHARE LOCK")

        self.validate_all(
            "SELECT * FROM t FOR SHARE LOCK",
            read={"postgres": "SELECT * FROM t FOR SHARE"},
        )
        self.validate_all(
            "SELECT * FROM t FOR UPDATE IGNORE LOCKED",
            read={"postgres": "SELECT * FROM t FOR UPDATE SKIP LOCKED"},
        )

    def test_date_arithmetic(self):
        for unit in ("YEAR", "MONTH", "DAY", "SECOND"):
            self.validate_identity(f"SELECT ADD_{unit}S(d, 3) FROM t")
            self.validate_identity(f"SELECT {unit}S_BETWEEN(a, b) FROM t")

        self.validate_identity("SELECT NEXT_DAY(a) FROM t", "SELECT ADD_DAYS(a, 1) FROM t")
        self.validate_identity("SELECT NEXT_DAY(a, 'MON') FROM t")

        self.validate_all(
            "SELECT DAYS_BETWEEN(d1, d2) FROM t",
            write={"tsql": "SELECT DATEDIFF(DAY, d1, d2) FROM t"},
        )
        self.validate_all(
            "SELECT ADD_DAYS(d, -3), ADD_DAYS(d, -(x)) FROM t",
            read={
                "mysql": "SELECT DATE_SUB(d, INTERVAL 3 DAY), DATE_SUB(d, INTERVAL x DAY) FROM t"
            },
        )
        self.validate_all(
            "SELECT ADD_SECONDS(d, 10800), ADD_SECONDS(d, 5400), ADD_SECONDS(d, (a + b) * 60) FROM t",
            read={
                "mysql": "SELECT DATE_ADD(d, INTERVAL 3 HOUR), DATE_ADD(d, INTERVAL '1.5' HOUR), DATE_ADD(d, INTERVAL (a + b) MINUTE) FROM t"
            },
        )
        self.validate_all(
            "SELECT ADD_DAYS(d, 14), ADD_MONTHS(d, 3) FROM t",
            read={"snowflake": "SELECT DATEADD(week, 2, d), DATEADD(quarter, 1, d) FROM t"},
        )
        self.validate_all(
            "SELECT CAST(SECONDS_BETWEEN(a, b) / 3600 AS BIGINT), CAST(DAYS_BETWEEN(a, b) / 7 AS BIGINT) FROM t",
            read={"mysql": "SELECT TIMESTAMPDIFF(HOUR, a, b), TIMESTAMPDIFF(WEEK, a, b) FROM t"},
        )
        # Snowflake and BigQuery count unit boundaries crossed rather than elapsed units
        self.validate_all(
            "SELECT YEARS_BETWEEN(SERIES_ROUND(a, 'INTERVAL 1 YEAR', ROUND_DOWN), SERIES_ROUND(b, 'INTERVAL 1 YEAR', ROUND_DOWN)) FROM t",
            read={
                "snowflake": "SELECT DATEDIFF(year, a, b) FROM t",
                "bigquery": "SELECT DATE_DIFF(b, a, YEAR) FROM t",
            },
        )
        self.validate_all(
            "SELECT CAST(MONTHS_BETWEEN(SERIES_ROUND(a, 'INTERVAL 3 MONTH', ROUND_DOWN), SERIES_ROUND(b, 'INTERVAL 3 MONTH', ROUND_DOWN)) / 3 AS BIGINT) FROM t",
            read={"bigquery": "SELECT DATE_DIFF(b, a, QUARTER) FROM t"},
        )
        self.assert_unsupported("SELECT DATE_DIFF(a, b, WEEK) FROM t", "bigquery")
        self.assert_unsupported("SELECT DATEADD(millisecond, 1, d) FROM t", "snowflake")

    def test_interval_arithmetic(self):
        self.validate_all(
            "SELECT ADD_DAYS(d, 1), ADD_MONTHS(d, 3), ADD_YEARS(d, -1), ADD_SECONDS(d, -7200), ADD_DAYS(d, 7) FROM t",
            read={
                "postgres": "SELECT d + INTERVAL '1 day', INTERVAL '3 months' + d, d - INTERVAL '1 year', d - INTERVAL '2 hours', d + INTERVAL '1 week' FROM t"
            },
        )
        self.validate_all(
            "SELECT ADD_DAYS(d, 1) FROM t",
            read={"mysql": "SELECT d + INTERVAL 1 DAY FROM t"},
        )
        self.assert_unsupported("SELECT INTERVAL '1 day'", "postgres")
        self.assert_unsupported("SELECT INTERVAL '1 day' + INTERVAL '2 day'", "postgres")

    def test_date_parts(self):
        for part in ("YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"):
            self.validate_identity(f"SELECT EXTRACT({part} FROM d) FROM t")

        self.validate_identity("SELECT WEEK(d), ISOWEEK(d), DAYOFYEAR(d) FROM t")
        self.validate_identity(
            "SELECT EXTRACT(DAYOFMONTH FROM d) FROM t", "SELECT DAYOFMONTH(d) FROM t"
        )

        self.validate_all(
            "SELECT DAYOFYEAR(d), (FLOOR((MONTH(d) - 1) / 3) + 1), CAST(SUBSTRING(ISOWEEK(d), 7, 2) AS INT), (WEEKDAY(d) + 1) FROM t",
            read={
                "postgres": "SELECT EXTRACT(DOY FROM d), EXTRACT(QUARTER FROM d), EXTRACT(WEEK FROM d), EXTRACT(ISODOW FROM d) FROM t"
            },
        )
        self.validate_all(
            "SELECT (MOD((WEEKDAY(CAST(d AS DATE)) + 1), 7) + 1), (FLOOR((MONTH(d) - 1) / 3) + 1) FROM t",
            read={"mysql": "SELECT DAYOFWEEK(d), QUARTER(d) FROM t"},
        )
        self.validate_all(
            "SELECT (WEEKDAY(d) + 1), CAST(SUBSTRING(ISOWEEK(d), 7, 2) AS INT) FROM t",
            read={"snowflake": "SELECT DAYOFWEEKISO(d), WEEKISO(d) FROM t"},
        )
        self.assert_unsupported("SELECT EXTRACT(DOW FROM d) FROM t", "postgres")
        self.assert_unsupported("SELECT EXTRACT(EPOCH FROM d) FROM t", "postgres")

    def test_date_trunc(self):
        self.validate_all(
            "SELECT SERIES_ROUND(d, 'INTERVAL 1 MONTH', ROUND_DOWN), SERIES_ROUND(d, 'INTERVAL 3 MONTH', ROUND_DOWN) FROM t",
            read={
                "postgres": "SELECT DATE_TRUNC('month', d), DATE_TRUNC('quarter', d) FROM t",
                "bigquery": "SELECT DATE_TRUNC(d, MONTH), DATE_TRUNC(d, QUARTER) FROM t",
            },
        )
        self.assert_unsupported("SELECT DATE_TRUNC('week', d) FROM t", "postgres")

    def test_current_datetime(self):
        self.validate_identity("SELECT CURRENT_TIMESTAMP, CURRENT_DATE, CURRENT_TIME FROM DUMMY")
        self.validate_identity(
            "SELECT CURRENT_UTCTIMESTAMP, CURRENT_UTCDATE, CURRENT_UTCTIME FROM DUMMY"
        )

        self.validate_all(
            "SELECT CURRENT_UTCTIMESTAMP, CURRENT_UTCDATE FROM t",
            read={"mysql": "SELECT UTC_TIMESTAMP(), UTC_DATE() FROM t"},
        )
        self.validate_all(
            "SELECT CURRENT_UTCTIMESTAMP FROM t",
            write={"mysql": "SELECT UTC_TIMESTAMP() FROM t"},
        )
        self.assert_unsupported("SELECT CURRENT_DATE('America/New_York')", "bigquery")

    def test_string_functions(self):
        self.validate_identity("SELECT CHAR(65) FROM DUMMY")
        self.validate_identity("SELECT CONCAT(a, b) FROM t", "SELECT a || b FROM t")
        self.validate_identity("SELECT LOCATE(haystack, needle, 3, 2) FROM t")
        self.validate_identity("SELECT STRING_AGG(a) FROM t")
        self.validate_identity("SELECT STRING_AGG(a, ',' ORDER BY a) FROM t")
        self.validate_identity("SELECT MAP(a, 1, 'one', 2, 'two', 'other') FROM t")
        self.validate_identity("SELECT ROUND(a, 2, ROUND_DOWN) FROM t")

        self.validate_all(
            "SELECT CHAR(a) FROM t",
            read={"postgres": "SELECT CHR(a) FROM t"},
            write={"postgres": "SELECT CHR(a) FROM t"},
        )
        self.validate_all(
            "SELECT COALESCE(a, '') || COALESCE(b, '') FROM t",
            read={"postgres": "SELECT CONCAT(a, b) FROM t"},
        )
        self.validate_all(
            "SELECT LOCATE(haystack, needle) FROM t",
            read={"postgres": "SELECT POSITION(needle IN haystack) FROM t"},
            write={"mysql": "SELECT LOCATE(needle, haystack) FROM t"},
        )
        self.validate_all(
            "SELECT LOCATE(haystack, needle, 3, 2) FROM t",
            read={"bigquery": "SELECT INSTR(haystack, needle, 3, 2) FROM t"},
            write={"bigquery": "SELECT INSTR(haystack, needle, 3, 2) FROM t"},
        )
        self.validate_all(
            "SELECT RPAD(x, 3 * LENGTH(x), x), RPAD(x, (a + b) * LENGTH(x), x) FROM t",
            read={"postgres": "SELECT REPEAT(x, 3), REPEAT(x, a + b) FROM t"},
        )
        self.validate_all(
            "SELECT STRING_AGG(a, ',') FROM t",
            read={"mysql": "SELECT GROUP_CONCAT(a SEPARATOR ',') FROM t"},
        )
        self.validate_all(
            "SELECT MAP(a, 1, 2, 3) FROM t",
            read={"oracle": "SELECT DECODE(a, 1, 2, 3) FROM t"},
            write={"oracle": "SELECT DECODE(a, 1, 2, 3) FROM t"},
        )
        self.validate_all(
            "SELECT * FROM t WHERE a LIKE_REGEXPR 'x' FLAG 'i'",
            write={"hana": "SELECT * FROM t WHERE a LIKE_REGEXPR 'x' FLAG 'i'"},
        )
        self.validate_all(
            "SELECT * FROM t WHERE NOT a LIKE_REGEXPR 'x'",
            read={"postgres": "SELECT * FROM t WHERE NOT a ~ 'x'"},
            write={
                "postgres": "SELECT * FROM t WHERE NOT a ~ 'x'",
                "oracle": "SELECT * FROM t WHERE NOT REGEXP_LIKE(a, 'x')",
            },
        )
        self.assert_unsupported("SELECT GROUP_CONCAT(DISTINCT a) FROM t", "mysql")
        self.assert_unsupported("SELECT SPLIT_PART(a, ',', 2) FROM t", "postgres")
        with self.assertRaises(UnsupportedError):
            exp.select(exp.Repeat(this=exp.column("x"))).sql(
                "hana", unsupported_level=ErrorLevel.RAISE
            )

    def test_trim(self):
        # LTRIM/RTRIM remove a set of characters: LTRIM('babababAabend', 'ab') is 'Aabend'
        self.validate_identity("SELECT LTRIM(a, 'ab'), RTRIM(a, 'ab'), TRIM(a) FROM t")
        self.validate_identity(
            "SELECT TRIM(BOTH 'x' FROM a) FROM t", "SELECT LTRIM(RTRIM(a, 'x'), 'x') FROM t"
        )
        self.validate_identity(
            "SELECT TRIM(LEADING 'x' FROM a) FROM t", "SELECT LTRIM(a, 'x') FROM t"
        )
        self.validate_identity(
            "SELECT TRIM(TRAILING 'x' FROM a) FROM t", "SELECT RTRIM(a, 'x') FROM t"
        )

        self.validate_all(
            "SELECT LTRIM(RTRIM(x, 'ab'), 'ab') FROM t",
            read={
                "bigquery": "SELECT TRIM(x, 'ab') FROM t",
                "postgres": "SELECT BTRIM(x, 'ab') FROM t",
            },
        )

    def test_hashing(self):
        self.validate_identity("SELECT HASH_MD5(TO_BINARY(x)), HASH_SHA256(TO_BINARY(x)) FROM t")

        self.validate_all(
            "SELECT LOWER(BINTOHEX(HASH_MD5(TO_BINARY(x)))), LOWER(BINTOHEX(HASH_SHA256(TO_BINARY(x)))) FROM t",
            read={"mysql": "SELECT MD5(x), SHA2(x, 256) FROM t"},
        )
        self.validate_all(
            "SELECT HASH_MD5(TO_BINARY(x)), HASH_SHA256(TO_BINARY(x)) FROM t",
            read={"bigquery": "SELECT MD5(x), SHA256(x) FROM t"},
        )
        self.assert_unsupported("SELECT SHA2(x, 512) FROM t", "mysql")
        self.assert_unsupported("SELECT SHA512(x) FROM t", "bigquery")

    def test_derived_tables(self):
        self.validate_all(
            "SELECT c FROM (SELECT k AS k, COUNT(o) AS c FROM t GROUP BY k) AS s",
            read={"postgres": "SELECT c FROM (SELECT k, COUNT(o) FROM t GROUP BY k) AS s(k, c)"},
        )
        self.validate_all(
            "SELECT * FROM (SELECT 1 AS a, 2 AS b FROM DUMMY UNION ALL SELECT 3, 4 FROM DUMMY) AS t",
            read={"postgres": "SELECT * FROM (VALUES (1, 2), (3, 4)) AS t(a, b)"},
        )
        self.assert_unsupported("SELECT * FROM (SELECT * FROM x) AS t(a)", "postgres")

    def test_booleans(self):
        self.validate_identity(
            "SELECT * FROM t WHERE a LIKE 'x!%' ESCAPE '!' AND b <> ALL (SELECT c FROM u)"
        )
        self.validate_identity(
            "SELECT a LIKE 'x!%' ESCAPE '!' FROM t",
            "SELECT CASE WHEN a LIKE 'x!%' ESCAPE '!' THEN TRUE WHEN NOT (a LIKE 'x!%' ESCAPE '!') THEN FALSE END FROM t",
        )
        self.validate_identity("SELECT CASE a WHEN 0 THEN 1 ELSE 2 END FROM t")
        # Predicates are only allowed as conditions, and conditions must be predicates
        self.validate_all(
            "SELECT CASE WHEN a = b THEN TRUE WHEN NOT (a = b) THEN FALSE END FROM t WHERE flag = TRUE AND (x > 1 OR NOT active = TRUE)",
            read={"postgres": "SELECT a = b FROM t WHERE flag AND (x > 1 OR NOT active)"},
        )
        self.validate_all(
            "SELECT COUNT(CASE WHEN x > 1 THEN 1 END), SUM(CASE WHEN flag = TRUE THEN a ELSE NULL END) FROM t",
            read={
                "postgres": "SELECT COUNT(*) FILTER (WHERE x > 1), SUM(a) FILTER (WHERE flag) FROM t"
            },
        )
        self.validate_all(
            "SELECT * FROM t WHERE NOT ((a = b AND NOT a IS NULL AND NOT b IS NULL) OR (a IS NULL AND b IS NULL))",
            read={"postgres": "SELECT * FROM t WHERE a IS DISTINCT FROM b"},
        )

    def test_semi_structured(self):
        self.validate_all(
            "SELECT JSON_QUERY(j, '$.a'), JSON_VALUE(j, '$.a[0].b') FROM t",
            read={
                "bigquery": "SELECT JSON_EXTRACT(j, '$.a'), JSON_EXTRACT_SCALAR(j, '$.a[0].b') FROM t"
            },
        )
        self.validate_all(
            "SELECT MEMBER_AT(arr, 1), MEMBER_AT(arr, 2) FROM t",
            read={"bigquery": "SELECT arr[OFFSET(0)], arr[ORDINAL(2)] FROM t"},
        )
        self.validate_identity(
            "SELECT SUBSTR_REGEXPR('a' FLAG 'i' IN s FROM 2 OCCURRENCE 3 GROUP 1) FROM t"
        )
        self.validate_all(
            "SELECT SUBSTR_REGEXPR('a(b)' IN s GROUP 1) FROM t",
            read={"bigquery": "SELECT REGEXP_EXTRACT(s, r'a(b)') FROM t"},
        )
        for sql, read in (
            ("SELECT GENERATE_SERIES(1, 3)", "postgres"),
            ("SELECT * FROM a NATURAL JOIN b", "postgres"),
            ("SELECT ARG_MAX(a, b) FROM t", "duckdb"),
            ("SELECT FIRST(a IGNORE NULLS) FROM t", "duckdb"),
            ("SELECT ts AT TIME ZONE 'UTC' FROM t", "postgres"),
        ):
            self.assert_unsupported(sql, read)

    def test_constructors(self):
        self.validate_all(
            "SELECT ADD_DAYS(ADD_MONTHS(TO_DATE(LPAD(TO_VARCHAR(y), 4, '0') || '-01-01'), m - 1), d - 1) FROM t",
            read={"snowflake": "SELECT DATE_FROM_PARTS(y, m, d) FROM t"},
        )
        self.validate_all(
            "SELECT ADD_SECONDS(TO_TIME('00:00:00'), (h * 3600) + (mi * 60) + s) FROM t",
            read={"snowflake": "SELECT TIME_FROM_PARTS(h, mi, s) FROM t"},
        )
        self.validate_all(
            "SELECT ADD_SECONDS(TO_TIMESTAMP('1970-01-01 00:00:00'), x), SECONDS_BETWEEN(TO_TIMESTAMP('1970-01-01 00:00:00'), ts) FROM t",
            read={"duckdb": "SELECT TO_TIMESTAMP(x), EPOCH(ts) FROM t"},
        )
        self.validate_all(
            "SELECT CAST(ts AS DATE), COUNT(DISTINCT x), CURRENT_TIMESTAMP, CAST(a AS TIMESTAMP), CAST(b AS TIME) FROM t",
            read={
                "postgres": "SELECT DATE(ts), COUNT(DISTINCT x), LOCALTIMESTAMP, CAST(a AS TIMESTAMPTZ), CAST(b AS TIMETZ) FROM t"
            },
        )
        self.validate_all(
            "SELECT HEXTOBIN(h), COUNT(DISTINCT x) FROM t",
            read={"duckdb": "SELECT UNHEX(h), APPROX_COUNT_DISTINCT(x) FROM t"},
        )
