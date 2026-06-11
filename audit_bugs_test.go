/*******************************************************************************
Regression tests for the four critical findings in AUDIT.md.

How to read this file:

  - Bug #1 and #2 are exercised with a *scripted* net.Conn (no live Firebird
    needed). They reproduce the failure deterministically: #1 hangs in an
    infinite loop, #2 swallows an error. They FAIL on the current master and
    PASS once the bug is fixed.
  - Bug #3 and #4 are pure unit tests (no server). They currently PANIC on
    master (which testify reports as a failed test), and PASS once fixed.
  - The *_Integration tests at the bottom need a live Firebird server (run like
    the rest of driver_test.go) and guard the happy paths around the same code.

Run only the audit tests:

    go test -run TestAuditBug -v

NOTE on Bug #1: while the bug is present, the test detects the infinite loop via
a timeout and leaves one goroutine parked forever (blocked on a channel, ~0 CPU).
That is the expected signature of the bug; it disappears once #1 is fixed.
*******************************************************************************/

package firebirdsql

import (
	"bytes"
	"database/sql"
	"database/sql/driver"
	"io"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// Scripted net.Conn: serves a fixed sequence of bytes to Read, then io.EOF.
// Writes are counted and (optionally) park forever after blockWritesAfter, so a
// buggy infinite loop that keeps writing can be detected without pegging a CPU.
// ---------------------------------------------------------------------------

type scriptedConn struct {
	mu               sync.Mutex
	readScript       []byte
	writes           int
	blockWritesAfter int // 0 = never block
	parked           chan struct{}
}

func newScriptedConn(readScript []byte, blockWritesAfter int) *scriptedConn {
	return &scriptedConn{
		readScript:       readScript,
		blockWritesAfter: blockWritesAfter,
		parked:           make(chan struct{}), // intentionally never closed
	}
}

func (c *scriptedConn) Read(p []byte) (int, error) {
	c.mu.Lock()
	if len(c.readScript) > 0 {
		n := copy(p, c.readScript)
		c.readScript = c.readScript[n:]
		c.mu.Unlock()
		return n, nil
	}
	c.mu.Unlock()
	return 0, io.EOF
}

func (c *scriptedConn) Write(p []byte) (int, error) {
	c.mu.Lock()
	c.writes++
	park := c.blockWritesAfter > 0 && c.writes > c.blockWritesAfter
	c.mu.Unlock()
	if park {
		<-c.parked // block forever: stops a buggy infinite write loop from spinning
		return 0, io.ErrClosedPipe
	}
	return len(p), nil
}

func (c *scriptedConn) Close() error                       { return nil }
func (c *scriptedConn) LocalAddr() net.Addr                { return fakeAddr{} }
func (c *scriptedConn) RemoteAddr() net.Addr               { return fakeAddr{} }
func (c *scriptedConn) SetDeadline(t time.Time) error      { return nil }
func (c *scriptedConn) SetReadDeadline(t time.Time) error  { return nil }
func (c *scriptedConn) SetWriteDeadline(t time.Time) error { return nil }

type fakeAddr struct{}

func (fakeAddr) Network() string { return "scripted" }
func (fakeAddr) String() string  { return "scripted" }

// newScriptedWireProtocol builds a wireProtocol wired to a scriptedConn.
func newScriptedWireProtocol(t *testing.T, readScript []byte, blockWritesAfter int) (*wireProtocol, *scriptedConn) {
	t.Helper()
	sc := newScriptedConn(readScript, blockWritesAfter)
	ch, err := newWireChannel(sc)
	require.NoError(t, err)
	p := &wireProtocol{
		buf:             make([]byte, 0, BUFFER_LEN),
		conn:            ch,
		protocolVersion: PROTOCOL_VERSION13,
		acceptType:      0, // not lazy_send
		charset:         "UTF8",
	}
	p.charsetLen()
	return p, sc
}

// opResponse wire frames (all big-endian, matching bytes_to_bint32 on read).
//
// op_response layout consumed by _parse_op_response:
//   [op_response(4)] [handle(4)] [oid(8)] [buf_len(4)] [status vector...]
// status vector is a stream of 4-byte tags terminated by isc_arg_end(0).

func okResponse(handle int32) []byte {
	var b []byte
	b = append(b, bint32_to_bytes(op_response)...)
	b = append(b, bint32_to_bytes(handle)...) // object handle
	b = append(b, make([]byte, 8)...)         // object id (2x int32)
	b = append(b, bint32_to_bytes(0)...)      // buf_len = 0
	b = append(b, bint32_to_bytes(isc_arg_end)...)
	return b
}

func errorResponse(gdsCode int32) []byte {
	var b []byte
	b = append(b, bint32_to_bytes(op_response)...)
	b = append(b, bint32_to_bytes(0)...) // object handle
	b = append(b, make([]byte, 8)...)    // object id
	b = append(b, bint32_to_bytes(0)...) // buf_len = 0
	// status vector with one GDS code -> _parse_op_response returns *FbError
	b = append(b, bint32_to_bytes(isc_arg_gds)...)
	b = append(b, bint32_to_bytes(gdsCode)...)
	b = append(b, bint32_to_bytes(isc_arg_end)...)
	return b
}

// ===========================================================================
// Bug #1: getBlobSegments infinite loop when opResponse fails mid-read.
// wireprotocol.go:897-906 — the err from opResponse() inside the loop is never
// checked, so a read error leaves more_data != 2 and the loop spins forever.
// ===========================================================================

func TestAuditBug1_GetBlobSegmentsLoopsOnReadError(t *testing.T) {
	// Script: a valid op_open_blob2 response, then nothing (EOF) for every
	// subsequent op_get_segment response — i.e. the connection died mid-blob.
	// Park writes after a generous bound so a buggy spin doesn't peg a CPU.
	p, _ := newScriptedWireProtocol(t, okResponse(1), 50)

	done := make(chan []interface{}, 1)
	go func() {
		blob, err := p.getBlobSegments([]byte{0, 0, 0, 0, 0, 0, 0, 0}, 1)
		done <- []interface{}{blob, err}
	}()

	select {
	case res := <-done:
		// Fixed behaviour: returns promptly with a non-nil error.
		err, _ := res[1].(error)
		require.Error(t, err, "getBlobSegments must surface the read error, not return success")
	case <-time.After(2 * time.Second):
		t.Fatal("BUG #1: getBlobSegments did not return — infinite loop on read error " +
			"(wireprotocol.go getBlobSegments ignores opResponse err inside the loop)")
	}
}

// ===========================================================================
// Bug #2: paramsToBlr discards createBlob errors (wireprotocol.go:1635/1686/1695
// "v, _ = p.createBlob(...)"). A failed BLOB write is silently turned into a
// bogus blobId and the statement proceeds, risking data corruption.
//
// We drive opExecute with a large []byte param (forces the createBlob path) and
// script the create-blob response to be a server error. opExecute must surface
// that error; today it returns nil.
// ===========================================================================

func TestAuditBug2_CreateBlobErrorIsSwallowed(t *testing.T) {
	// One scripted read: the op_create_blob2 response is a Firebird error.
	p, _ := newScriptedWireProtocol(t, errorResponse(335544321), 0)

	fc := &firebirdsqlConn{wp: p}
	fc.tx = &firebirdsqlTx{fc: fc, transHandle: 1}
	stmt := &firebirdsqlStmt{fc: fc, stmtHandle: 1}

	// > MAX_CHAR_LENGTH so paramsToBlr routes the value through createBlob.
	bigParam := bytes.Repeat([]byte("x"), MAX_CHAR_LENGTH+10)

	err := p.opExecute(stmt, []driver.Value{bigParam}, nil)
	require.Error(t, err,
		"BUG #2: opExecute must propagate the createBlob failure instead of swallowing it")

	var fbErr *FbError
	require.ErrorAs(t, err, &fbErr, "the surfaced error should be the server FbError from createBlob")
}

// ===========================================================================
// Bug #3: invalid / unavailable IANA timezone -> nil *time.Location -> panic.
// xsqlvar.go parse{Date,Time,Timestamp} ignore the LoadLocation error and pass
// nil to time.Date(), which panics ("missing Location"). Reproduces both on a
// typo'd ?timezone= value and on hosts without a tz database.
// ===========================================================================

func TestAuditBug3_InvalidTimezoneDoesNotPanic(t *testing.T) {
	const badTZ = "Definitely/NotARealZone"

	xDate := &xSQLVAR{sqltype: SQL_TYPE_DATE}
	rawDate := _convert_date(time.Date(2020, 3, 15, 0, 0, 0, 0, time.UTC))
	require.NotPanics(t, func() { _ = xDate.parseDate(rawDate, badTZ) },
		"BUG #3: parseDate panics on an unresolvable timezone (nil *time.Location)")

	xTime := &xSQLVAR{sqltype: SQL_TYPE_TIME}
	rawTime := _convert_time(time.Date(2020, 3, 15, 10, 11, 12, 0, time.UTC))
	require.NotPanics(t, func() { _ = xTime.parseTime(rawTime, badTZ) },
		"BUG #3: parseTime panics on an unresolvable timezone")

	xTs := &xSQLVAR{sqltype: SQL_TYPE_TIMESTAMP}
	rawTs := _convert_timestamp(time.Date(2020, 3, 15, 10, 11, 12, 0, time.UTC))
	require.NotPanics(t, func() { _ = xTs.parseTimestamp(rawTs, badTZ) },
		"BUG #3: parseTimestamp panics on an unresolvable timezone")
}

// Secondary aspect of #3: parseDSN should reject an invalid timezone up front
// (fail-fast) rather than deferring a panic to the first temporal read.
func TestAuditBug3_DSNRejectsInvalidTimezone(t *testing.T) {
	_, err := parseDSN("sysdba:masterkey@localhost/var/lib/firebird/db.fdb?timezone=Definitely/NotARealZone")
	require.Error(t, err,
		"BUG #3: parseDSN should validate ?timezone= and fail fast on an unknown zone")
}

// ===========================================================================
// Bug #4: parseDSN panics ("slice bounds out of range") on a missing/short
// database path. dsn.go:67/72 index dsn.dbName[1:] and [2:] without a length
// guard. Should return a clear error instead.
// ===========================================================================

func TestAuditBug4_ParseDSNShortPathDoesNotPanic(t *testing.T) {
	cases := []string{
		"sysdba:masterkey@localhost",      // no path at all
		"sysdba:masterkey@localhost:3050", // no path, explicit port
		"sysdba:masterkey@localhost/",     // root slash only
	}
	for _, dsn := range cases {
		t.Run(dsn, func(t *testing.T) {
			require.NotPanics(t, func() {
				if _, err := parseDSN(dsn); err == nil {
					t.Errorf("BUG #4: parseDSN(%q) should return an error for a missing database path", dsn)
				}
			}, "BUG #4: parseDSN(%q) panics on a missing/short database path", dsn)
		})
	}
}

// ===========================================================================
// Integration tests (require a live Firebird server, like driver_test.go).
// They guard the happy paths around bugs #1 and #2.
// ===========================================================================

// Bug #1 happy path: a large, multi-segment binary BLOB must round-trip with
// byte-for-byte fidelity (exercises getBlobSegments reassembly).
func TestAuditBug1_Integration_MultiSegmentBlobRoundtrip(t *testing.T) {
	test_dsn := GetTestDSN("test_audit_bug1_")
	conn, err := sql.Open("firebirdsql_createdb", test_dsn)
	require.NoError(t, err)
	defer conn.Close()

	_, err = conn.Exec("CREATE TABLE audit_blob1 (id INTEGER, data BLOB SUB_TYPE BINARY)")
	require.NoError(t, err)

	// Span several segments plus a partial tail; use a non-trivial byte pattern.
	size := 3*BLOB_SEGMENT_SIZE + 1234
	want := make([]byte, size)
	for i := range want {
		want[i] = byte((i*7 + 13) % 251)
	}

	_, err = conn.Exec("INSERT INTO audit_blob1 (id, data) VALUES (1, ?)", want)
	require.NoError(t, err)

	var got []byte
	err = conn.QueryRow("SELECT data FROM audit_blob1 WHERE id = 1").Scan(&got)
	require.NoError(t, err)
	require.Equal(t, len(want), len(got), "blob length mismatch")
	require.True(t, bytes.Equal(want, got), "blob content mismatch after round-trip")
}

// Bug #2 happy path: a string parameter larger than MAX_CHAR_LENGTH is written
// via createBlob; it must round-trip intact (and, with #2 fixed, any write
// failure on this path would surface as an error rather than silent corruption).
func TestAuditBug2_Integration_LargeStringParamRoundtrip(t *testing.T) {
	test_dsn := GetTestDSN("test_audit_bug2_")
	conn, err := sql.Open("firebirdsql_createdb", test_dsn)
	require.NoError(t, err)
	defer conn.Close()

	_, err = conn.Exec("CREATE TABLE audit_blob2 (id INTEGER, txt BLOB SUB_TYPE TEXT)")
	require.NoError(t, err)

	want := strings.Repeat("The quick brown fox. ", (MAX_CHAR_LENGTH/21)+200) // > MAX_CHAR_LENGTH
	require.Greater(t, len(want), MAX_CHAR_LENGTH)

	_, err = conn.Exec("INSERT INTO audit_blob2 (id, txt) VALUES (1, ?)", want)
	require.NoError(t, err)

	var got string
	err = conn.QueryRow("SELECT txt FROM audit_blob2 WHERE id = 1").Scan(&got)
	require.NoError(t, err)
	require.Equal(t, want, got, "text blob content mismatch after round-trip")
}
