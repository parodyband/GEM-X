"""Connect to a running server, stream a video, and summarise what arrives."""
import json, socket, sys, time
host, port, video = "127.0.0.1", 47811, sys.argv[1]
s = socket.create_connection((host, port)); f = s.makefile("rwb")
def send(m): f.write((json.dumps(m) + "\n").encode()); f.flush()
send({"cmd": "hello", "preview": True})
counts, first, last, n_preview, t0 = {}, None, None, 0, time.time()
send({"cmd": "start", "source": "video", "path": video, "mode": sys.argv[2] if len(sys.argv) > 2 else "all"})
for line in f:
    m = json.loads(line)
    t = m["type"]; counts[t] = counts.get(t, 0) + 1
    if t == "hello": print("hello:", m["protocol"], m["server"]["device"], len(m["skeleton"]["names"]), "joints")
    elif t == "frame":
        if m["outcome"] == "pose":
            first = first or m; last = m
    elif t == "preview": n_preview += 1; plen = len(m["jpeg"])
    elif t == "state":
        print("state:", {k: m.get(k) for k in ("running", "source", "message", "error", "finished")})
        if m.get("running") is False and m.get("message") in ("finished", "stopped") and counts.get("frame"): break
    elif t == "error": print("ERROR", m); break
print("counts", counts, "elapsed %.1fs" % (time.time() - t0))
print("last frame: seq", last["seq"], "t", last["t"], "root", last["root"], "fps", last["fps"], "infer", last["infer_ms"], "lat", last["latency_ms"], "rot len", len(last["rot"]))
print("preview msgs", n_preview, "b64 bytes", plen)
send({"cmd": "shutdown"})
