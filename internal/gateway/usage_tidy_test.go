package gateway

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

// An Anthropic-compatible relay that puts real token counts only in
// message_delta, leaving message_start's usage as {}, breaks Cursor Private
// Inference's zod schema (input_tokens / output_tokens required numbers).
// The passthrough fills 0 so the first event validates; message_delta keeps
// the real counts, and the rest of the stream stays as it came.
func TestPassthroughFillsEmptyMessageStartUsage(t *testing.T) {
	fresh(t)
	start := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"usage":{}}}` + "\n\n"
	delta := "event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":13,"output_tokens":4}}` + "\n\n"
	stop := "event: message_stop\n" + `data: {"type":"message_stop"}` + "\n\n"
	reply := start + anthropicText + delta + stop
	f := &fake{reply: reply}
	setup(t, "anthropic", f)
	code, body := postMessages(t, `{"model":"fake/m1","stream":true,"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	if code != 200 {
		t.Fatalf("status = %d, want 200; body: %s", code, body)
	}
	usage := messageStartUsage(t, body)
	if usage.Get("input_tokens").Type != gjson.Number || usage.Get("output_tokens").Type != gjson.Number {
		t.Fatalf("message_start.usage counts are not numbers: %s", usage.Raw)
	}
	if usage.Get("input_tokens").Num != 0 || usage.Get("output_tokens").Num != 0 {
		t.Fatalf("filled counts = %s, want 0 and 0", usage.Raw)
	}
	if !strings.Contains(body, anthropicText) || !strings.Contains(body, delta) || !strings.Contains(body, stop) {
		t.Fatalf("events after message_start were rewritten: %s", body)
	}
}

// A message_start that already has both counts is relayed byte for byte,
// including a message_delta that only carries output_tokens.
func TestPassthroughLeavesConformingMessageStartUsage(t *testing.T) {
	fresh(t)
	start := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":13,"output_tokens":1,"cache_read_input_tokens":3}}}` + "\n\n"
	reply := start + anthropicText + anthropicStop
	f := &fake{reply: reply}
	setup(t, "anthropic", f)
	code, body := postMessages(t, `{"model":"fake/m1","stream":true,"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	if code != 200 || body != reply {
		t.Fatalf("a conforming stream was changed: %d\n got %q\nwant %q", code, body, reply)
	}
}

// Only the missing count is filled. The one the upstream sent, and any
// other usage field, stay.
func TestPassthroughFillsOnlyMissingUsageField(t *testing.T) {
	fresh(t)
	start := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","content":[],"usage":{"cache_read_input_tokens":9,"input_tokens":7}}}` + "\n\n"
	f := &fake{reply: start + anthropicText + anthropicStop}
	setup(t, "anthropic", f)
	code, body := postMessages(t, `{"model":"fake/m1","stream":true,"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	if code != 200 {
		t.Fatalf("status = %d; body: %s", code, body)
	}
	usage := messageStartUsage(t, body)
	if usage.Get("input_tokens").Num != 7 || usage.Get("cache_read_input_tokens").Num != 9 {
		t.Fatalf("kept counts changed: %s", usage.Raw)
	}
	if usage.Get("output_tokens").Type != gjson.Number || usage.Get("output_tokens").Num != 0 {
		t.Fatalf("output_tokens = %s, want 0", usage.Get("output_tokens").Raw)
	}
	if !strings.Contains(body, anthropicStop) {
		t.Fatalf("message_delta was rewritten: %s", body)
	}
}

// A whole Messages reply with an empty usage is filled the same way. One
// that already has both counts is not rewritten.
func TestPassthroughFillsEmptyWholeMessageUsage(t *testing.T) {
	fresh(t)
	f := &fake{ctype: "application/json", reply: `{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"m","usage":{}}`}
	setup(t, provider.Anthropic, f)
	code, body := post(t, "/v1/messages", `{"model":"fake/m1","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	if code != 200 {
		t.Fatalf("status = %d; body: %s", code, body)
	}
	if gjson.Get(body, "usage.input_tokens").Type != gjson.Number || gjson.Get(body, "usage.output_tokens").Type != gjson.Number {
		t.Fatalf("whole usage not filled: %s", body)
	}
	if gjson.Get(body, "id").String() != "msg_1" || gjson.Get(body, "content").Raw != "[]" {
		t.Fatalf("whole message was damaged: %s", body)
	}

	f.reply = `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"usage":{"output_tokens":2,"input_tokens":5}}`
	code, body = post(t, "/v1/messages", `{"model":"fake/m1","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	if code != 200 || body != f.reply {
		t.Fatalf("a conforming whole message was changed: %d\n got %q\nwant %q", code, body, f.reply)
	}
}

func TestUsageTidyUnit(t *testing.T) {
	feed := func(sse bool, in string) string {
		t.Helper()
		u := &usageTidy{sse: sse}
		var out []byte
		for i := 0; i < len(in); i++ {
			out = append(out, u.write([]byte{in[i]})...)
		}
		return string(append(out, u.flush()...))
	}
	t.Run("empty object", func(t *testing.T) {
		got := feed(true, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{}}}\n\n")
		usage := gjson.Get(strings.TrimPrefix(strings.Split(got, "\n")[1], "data: "), "message.usage")
		if usage.Get("input_tokens").Type != gjson.Number || usage.Get("output_tokens").Type != gjson.Number {
			t.Fatalf("not filled: %s", got)
		}
	})
	t.Run("null usage", func(t *testing.T) {
		got := feed(true, "data: {\"type\":\"message_start\",\"message\":{\"usage\":null}}\n")
		data := strings.TrimPrefix(strings.TrimSpace(got), "data: ")
		if gjson.Get(data, "message.usage.input_tokens").Type != gjson.Number || gjson.Get(data, "message.usage.output_tokens").Type != gjson.Number {
			t.Fatalf("null usage not filled: %s", got)
		}
	})
	t.Run("already complete", func(t *testing.T) {
		in := "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"output_tokens\":2,\"input_tokens\":1}}}\n"
		if got := feed(true, in); got != in {
			t.Fatalf("rewrote conforming event:\n got %q\nwant %q", got, in)
		}
	})
	t.Run("message delta only output", func(t *testing.T) {
		in := "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2}}\n"
		if got := feed(true, in); got != in {
			t.Fatalf("rewrote message_delta:\n got %q\nwant %q", got, in)
		}
	})
	t.Run("cut mid line", func(t *testing.T) {
		in := "data: {\"type\":\"message_start\",\"message\":{\"usage\":{"
		if got := feed(true, in); got != in {
			t.Fatalf("rewrote a cut line:\n got %q\nwant %q", got, in)
		}
	})
	t.Run("whole message", func(t *testing.T) {
		got := feed(false, `{"type":"message","id":"m","role":"assistant","content":[],"usage":{}}`)
		if gjson.Get(got, "usage.input_tokens").Type != gjson.Number || gjson.Get(got, "usage.output_tokens").Type != gjson.Number {
			t.Fatalf("whole reply not filled: %s", got)
		}
		in := `{"type":"message","usage":{"input_tokens":1,"output_tokens":2}}`
		if got := feed(false, in); got != in {
			t.Fatalf("rewrote a conforming whole message:\n got %q\nwant %q", got, in)
		}
	})
}

func messageStartUsage(t *testing.T, body string) gjson.Result {
	t.Helper()
	var raw string
	readSSE(strings.NewReader(body), func(ev, data string) error {
		if ev == "message_start" || gjson.Get(data, "type").String() == "message_start" {
			raw = data
		}
		return nil
	})
	if raw == "" {
		t.Fatalf("no message_start in reply: %s", body)
	}
	return gjson.Get(raw, "message.usage")
}
