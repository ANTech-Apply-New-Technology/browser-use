"""Production entrypoint for the browser_use FastAPI job server on Railway.

Why this exists instead of `uvicorn api:app --host ::`:

Railway's private network is IPv6-only (siblings connect via
`<service>.railway.internal`), so the server must listen on `::`. But CPython's
``asyncio.loop.create_server()`` — which both ``uvicorn`` and the plain CLI use —
unconditionally sets ``IPV6_V6ONLY=1`` on every ``AF_INET6`` socket it opens.
That makes ``::`` an *IPv6-only* listener: it serves IPv6 private-network
siblings fine, but Railway's deploy **healthcheck probes over IPv4**, gets
connection-refused, and the deploy is marked "1/1 replicas never became healthy"
even though the app is up.

The fix: create the listening socket ourselves with ``IPV6_V6ONLY=0`` (dual
stack), bind ``::`` and hand the ready socket to uvicorn via
``Server.serve(sockets=[...])``. A dual-stack ``::`` socket accepts both IPv6
(private network) and IPv4-mapped (healthcheck) connections on Linux, so both
paths work.
"""

from __future__ import annotations

import asyncio
import os
import socket

import uvicorn

from api import app


def _build_dualstack_socket(port: int) -> socket.socket:
    """Bind an IPv6 socket on `::` with dual-stack (IPv4-mapped) enabled."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # The whole point: undo asyncio's default IPV6_V6ONLY=1 so the listener
    # also answers IPv4 (e.g. the Railway healthcheck on 127.0.0.1).
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.bind(("::", port))
    sock.listen(128)
    sock.set_inheritable(True)
    return sock


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    sock = _build_dualstack_socket(port)
    config = uvicorn.Config(app, log_level="info", access_log=True)
    server = uvicorn.Server(config)
    # serve(sockets=[...]) skips uvicorn's own (IPv6-only) socket creation.
    asyncio.run(server.serve(sockets=[sock]))


if __name__ == "__main__":
    main()
