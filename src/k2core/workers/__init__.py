"""Stable worker-envelope namespace."""

from k2core.worker.protocol import CommandKind, WorkerCommand, WorkerEvent, WorkerState

WORKER_PROTOCOL_VERSION = "1"

__all__ = [
    "CommandKind",
    "WORKER_PROTOCOL_VERSION",
    "WorkerCommand",
    "WorkerEvent",
    "WorkerState",
]

