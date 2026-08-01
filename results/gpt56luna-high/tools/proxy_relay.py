#!/usr/bin/env python3
"""Small TCP relay exposing a host-loopback proxy to benchmark containers."""

from __future__ import annotations

import argparse
import select
import socket
import socketserver


class RelayHandler(socketserver.BaseRequestHandler):
    upstream_host = "127.0.0.1"
    upstream_port = 7890

    def handle(self) -> None:
        with socket.create_connection(
            (self.upstream_host, self.upstream_port), timeout=15
        ) as upstream:
            peers = (self.request, upstream)
            while True:
                readable, _, _ = select.select(peers, (), (), 60)
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    destination = upstream if source is self.request else self.request
                    destination.sendall(data)


class RelayServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=7891)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=7890)
    args = parser.parse_args()

    RelayHandler.upstream_host = args.upstream_host
    RelayHandler.upstream_port = args.upstream_port
    with RelayServer((args.listen_host, args.listen_port), RelayHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
