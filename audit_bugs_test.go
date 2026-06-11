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
	"io"
	"net"
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
// Integration test (requires a live Firebird server, like driver_test.go).
// Guards the happy path around bug #1.
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
