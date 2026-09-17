"""Read-only local app. All market data and computation remain on this machine."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            args = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self.respond(
                        200,
                        (Path(__file__).parent / "app.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                elif url.path == "/api/catalog":
                    self.respond_json(
                        {
                            "methods": engine.methods,
                            "pairs": sorted(
                                {(s.code, s.period) for s in engine.series if s.main.any()}
                            ),
                            "model_available_after": engine.meta["model_available_after"],
                        }
                    )
                elif url.path == "/api/times":
                    code, period = args.get("code", "rb"), int(args.get("period", 60))
                    now = np.datetime64("now", "ns") + np.timedelta64(8, "h")
                    times = sorted(
                        {
                            str(s.ends[r])
                            for s in engine.series
                            if s.code == code and s.period == period
                            for r in np.flatnonzero(
                                s.main
                                & (np.arange(len(s.frame)) >= engine.window - 1)
                                & (s.ends <= now)
                            )
                        }
                    )
                    self.respond_json(times)
                elif url.path == "/api/query":
                    self.respond_json(
                        engine.query(
                            args.get("code", "rb"),
                            int(args.get("period", 60)),
                            args.get("asof"),
                            args.get("method", "rule"),
                            int(args.get("topk", 6)),
                        )
                    )
                else:
                    self.respond(404, b"Not found", "text/plain")
            except (ValueError, KeyError, IndexError) as e:
                self.respond_json({"error": str(e)}, 400)

        def respond_json(self, value, status=200):
            self.respond(
                status,
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def respond(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    return Handler


def serve(engine, port=8765):
    server = HTTPServer(("127.0.0.1", port), handler_for(engine))
    print(f"Babel → http://127.0.0.1:{port}  (read-only, Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
