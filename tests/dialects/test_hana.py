from sqlglot import UnsupportedError
from tests.dialects.test_dialect import Validator


class TestHana(Validator):
    dialect = "hana"
    maxDiff = None

    def test_hana(self):
        self.validate_identity("SELECT 1 FROM DUMMY")
        self.validate_identity("SELECT * FROM t WHERE a = 1")
        self.validate_identity("SELECT a || b FROM t")
        self.validate_identity("SELECT COALESCE(a, b) FROM t")
        self.validate_identity("SELECT * FROM t LIMIT 5")
        self.validate_identity("SELECT * FROM t LIMIT 5 OFFSET 2")
        self.validate_identity("SELECT CURRENT_TIMESTAMP")
        self.validate_identity("SELECT CURRENT_DATE")
        self.validate_identity("SELECT CURRENT_TIME")
        self.validate_identity("SELECT a FROM t1 EXCEPT SELECT a FROM t2")
        self.validate_identity('SELECT "MixedCase" FROM t')

    def test_identifier_normalization(self):
        # https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/identifiers
        from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
        from sqlglot import parse_one

        self.assertEqual(
            normalize_identifiers(parse_one("SELECT a FROM tbl", read="hana"), dialect="hana").sql(
                "hana"
            ),
            "SELECT A FROM TBL",
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
        self.validate_identity(
            "SELECT CAST(a AS SECONDDATE) FROM t", "SELECT CAST(a AS TIMESTAMP) FROM t"
        )
        self.validate_identity(
            "SELECT CAST(a AS SMALLDECIMAL) FROM t", "SELECT CAST(a AS DECIMAL) FROM t"
        )
        self.validate_identity(
            "SELECT CAST(a AS ALPHANUM) FROM t", "SELECT CAST(a AS VARCHAR) FROM t"
        )
        # SAP HANA has no BINARY type; VARBINARY covers both.
        self.validate_all(
            "SELECT CAST(a AS VARBINARY) FROM t",
            read={"postgres": "SELECT CAST(a AS BYTEA) FROM t"},
            write={"hana": "SELECT CAST(a AS VARBINARY) FROM t"},
        )
        self.validate_all(
            "SELECT CAST(a AS DATETIME) FROM t",
            write={"hana": "SELECT CAST(a AS TIMESTAMP) FROM t"},
        )

    def test_offset_requires_limit(self):
        # SAP HANA parses OFFSET only as part of a LIMIT clause, so an offset-only query needs a
        # limit synthesised. 2147384648 is the largest LIMIT HANA accepts.
        self.validate_all(
            "SELECT * FROM t LIMIT 2147384648 OFFSET 10",
            read={"postgres": "SELECT * FROM t OFFSET 10"},
            write={"hana": "SELECT * FROM t LIMIT 2147384648 OFFSET 10"},
        )
        self.validate_all(
            "SELECT * FROM t LIMIT 5 OFFSET 10",
            read={"postgres": "SELECT * FROM t LIMIT 5 OFFSET 10"},
            write={"hana": "SELECT * FROM t LIMIT 5 OFFSET 10"},
        )
        # FETCH FIRST is rewritten as LIMIT, which HANA does support.
        self.validate_all(
            "SELECT * FROM t LIMIT 3",
            read={"oracle": "SELECT * FROM t FETCH FIRST 3 ROWS ONLY"},
            write={"hana": "SELECT * FROM t LIMIT 3"},
        )

    def test_date_arithmetic(self):
        # ADD_<unit>S / <unit>S_BETWEEN exist for YEAR, MONTH, DAY, HOUR, MINUTE and SECOND only.
        for unit in ("YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"):
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

        # There is no ADD_WEEKS / WEEKS_BETWEEN in SAP HANA.
        self.validate_all(
            "SELECT DATE_ADD(d, 1, 'WEEK') FROM t",
            write={"hana": UnsupportedError},
        )

    def test_extract(self):
        # SAP HANA has no EXTRACT(); the parts are plain functions instead.
        self.validate_all(
            "SELECT YEAR(d) FROM t",
            read={"postgres": "SELECT EXTRACT(YEAR FROM d) FROM t"},
            write={"hana": "SELECT YEAR(d) FROM t"},
        )
        self.validate_all(
            "SELECT DAYOFMONTH(d) FROM t",
            read={"postgres": "SELECT EXTRACT(DAY FROM d) FROM t"},
            write={"hana": "SELECT DAYOFMONTH(d) FROM t"},
        )
        self.validate_all(
            "SELECT MINUTE(d) FROM t",
            read={"postgres": "SELECT EXTRACT(MINUTE FROM d) FROM t"},
            write={"hana": "SELECT MINUTE(d) FROM t"},
        )

    def test_string_functions(self):
        # SAP HANA has no REPEAT; RPAD padded with the string itself is the documented idiom.
        self.validate_all(
            "SELECT RPAD(x, 3 * LENGTH(x), x) FROM t",
            read={"postgres": "SELECT REPEAT(x, 3) FROM t"},
            write={"hana": "SELECT RPAD(x, 3 * LENGTH(x), x) FROM t"},
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
        self.validate_identity("SELECT LOCATE(haystack, needle, 3)")

    def test_hash_functions(self):
        # HASH_MD5 / HASH_SHA256 take a BINARY argument, hence the TO_BINARY wrapper.
        self.validate_all(
            "SELECT HASH_MD5(TO_BINARY(x)) FROM t",
            read={"postgres": "SELECT MD5(x) FROM t"},
            write={"hana": "SELECT HASH_MD5(TO_BINARY(x)) FROM t"},
        )
        self.validate_all(
            "SELECT HASH_SHA256(TO_BINARY(x)) FROM t",
            read={"postgres": "SELECT SHA256(x) FROM t"},
            write={"hana": "SELECT HASH_SHA256(TO_BINARY(x)) FROM t"},
        )
        self.validate_all(
            "SELECT SHA2(x, 384) FROM t",
            write={"hana": UnsupportedError},
        )

    def test_to_varchar(self):
        # TO_VARCHAR doubles as the datetime formatter.
        self.validate_identity("SELECT TO_VARCHAR(d, 'YYYY-MM-DD') FROM t")
        self.validate_all(
            "SELECT TO_VARCHAR(CAST(d AS DATE), 'YYYY-MM-DD') FROM t",
            write={
                "hana": "SELECT TO_VARCHAR(CAST(d AS DATE), 'YYYY-MM-DD') FROM t",
                "duckdb": "SELECT STRFTIME(CAST(d AS DATE), '%Y-%m-%d') FROM t",
            },
        )
