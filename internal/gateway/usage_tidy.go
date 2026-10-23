package gateway

import (
	"bytes"
	"encoding/json"
)

// usageTidy fills, in a Messages reply relayed as it came, the token counts
// Anthropic requires on message_start when the upstream left them out.
// Some relays send "usage":{} there and put the real counts only in
// message_delta. A strict client (Cursor Private Inference) rejects that
// first event, aborts, and retries. The missing counts are written as 0;
// message_delta still carries the real ones and overwrites them. The event
// is parsed and, only when a count was missing or null, marshaled again.
// One that already has both as numbers goes byte for byte, as does every
// other event. A whole message is filled the same way.
type usageTidy struct {
	sse bool
	buf []byte
}

func (t *usageTidy) write(b []byte) []byte {
	t.buf = append(t.buf, b...)
	if !t.sse {
		return nil
	}
	i := bytes.LastIndexByte(t.buf, '\n')
	if i < 0 {
		return nil
	}
	out := t.lines(t.buf[:i+1])
	t.buf = append(t.buf[:0], t.buf[i+1:]...)
	return out
}

func (t *usageTidy) flush() []byte {
	b := t.buf
	t.buf = nil
	if !t.sse {
		if out, ok := t.event(b); ok {
			return out
		}
		return b
	}
	return t.lines(b)
}

func (t *usageTidy) lines(b []byte) []byte {
	if !bytes.Contains(b, []byte(`"message_start"`)) {
		return append([]byte(nil), b...)
	}
	var out []byte
	for len(b) > 0 {
		line := b
		if i := bytes.IndexByte(b, '\n'); i >= 0 {
			line, b = b[:i+1], b[i+1:]
		} else {
			b = nil
		}
		body := bytes.TrimRight(line, "\r\n")
		data, ok := bytes.CutPrefix(body, []byte("data:"))
		if !ok || !bytes.Contains(data, []byte(`"message_start"`)) {
			out = append(out, line...)
			continue
		}
		nb, ok := t.event(data)
		if !ok {
			out = append(out, line...)
			continue
		}
		out = append(append(append(out, "data: "...), nb...), line[len(body):]...)
	}
	return out
}

// event is a message_start, or a whole message, with a missing token count
// filled as 0, and whether that changed it.
func (t *usageTidy) event(data []byte) ([]byte, bool) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || !json.Valid(trimmed) {
		return nil, false
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.UseNumber()
	var ev map[string]any
	if dec.Decode(&ev) != nil || ev == nil {
		return nil, false
	}
	if rest := bytes.TrimSpace(trimmed[dec.InputOffset():]); len(rest) > 0 {
		return nil, false
	}
	var usage map[string]any
	switch {
	case ev["type"] == "message_start":
		msg, _ := ev["message"].(map[string]any)
		if msg == nil {
			return nil, false
		}
		usage = usageObject(msg)
	case !t.sse && ev["type"] == "message":
		usage = usageObject(ev)
	default:
		return nil, false
	}
	if usage == nil || !fillUsageTokens(usage) {
		return nil, false
	}
	nb, err := marshalPlain(ev)
	if err != nil {
		return nil, false
	}
	return nb, true
}

// usageObject is m's usage object, created when it was missing or null.
// Some other JSON type is left alone: the caller then skips the event.
func usageObject(m map[string]any) map[string]any {
	v, ok := m["usage"]
	if ok && v != nil {
		u, _ := v.(map[string]any)
		return u
	}
	u := map[string]any{}
	m["usage"] = u
	return u
}

// fillUsageTokens sets input_tokens and output_tokens to 0 where they are
// missing or null, and reports whether it changed usage. A count already
// there, number or otherwise, stays.
func fillUsageTokens(usage map[string]any) bool {
	changed := false
	for _, k := range []string{"input_tokens", "output_tokens"} {
		v, ok := usage[k]
		if ok && v != nil {
			continue
		}
		usage[k] = 0
		changed = true
	}
	return changed
}
