"""Short, cross-process JSONL transactions; never hold a lock across model I/O."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time


_GUARD = threading.Lock()
_LOCKS = {}


@contextmanager
def process_lock(path, *, timeout=20.0, timeout_error=None, poll_interval=0.001):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.path.normcase(str(path.resolve()))
    with _GUARD:
        mutex = _LOCKS.setdefault(key, threading.Lock())
    deadline = time.monotonic() + timeout
    if not mutex.acquire(timeout=max(0, timeout)):
        raise timeout_error or OSError("records_lock_timeout")
    try:
        with path.open("a+b") as lock:
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            while True:
                try:
                    lock.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (BlockingIOError, OSError) as exc:
                    if time.monotonic() >= deadline:
                        raise timeout_error or OSError("records_lock_timeout") from exc
                    time.sleep(poll_interval)
            try:
                yield
            finally:
                lock.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        mutex.release()


def replace_text(path, text):
    """Flush replacement contents to disk before publishing their filename."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_row(path, row, *, lock_name=".records.lock", durable=False):
    path = Path(path)
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    with process_lock(path.with_name(lock_name)):
        with path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            if durable:
                os.fsync(stream.fileno())


class JsonlTail:
    """Read only appends; reset on replacement, truncation or changed boundary."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None
        self.modified = None
        self.boundary = b""

    def read(self, decode_row=None):
        # A metadata-only fast path avoids reopening unchanged files for every
        # already-cached gate line. Every actual read still takes the lock.
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self.identity is None:
                return [], False
        else:
            if (
                self.identity == (stat.st_dev, stat.st_ino)
                and self.offset == stat.st_size
                and self.modified == stat.st_mtime_ns
            ):
                return [], False
        with process_lock(self.path.with_name(".records.lock")):
            if not self.path.exists():
                reset = self.identity is not None
                self.offset, self.identity = 0, None
                self.modified, self.boundary = None, b""
                return [], reset
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                reset = (
                    self.identity != identity
                    or stat.st_size < self.offset
                    or (
                        stat.st_size == self.offset
                        and stat.st_mtime_ns != self.modified
                    )
                )
                if not reset and self.boundary:
                    stream.seek(self.offset - len(self.boundary))
                    reset = stream.read(len(self.boundary)) != self.boundary
                start = 0 if reset else self.offset
                stream.seek(start)
                data = stream.read()
                # Parse the complete snapshot before committing the cursor.
                text = data.decode("utf-8-sig" if start == 0 else "utf-8")
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
                if decode_row is not None:
                    rows = [decode_row(row) for row in rows]
                end = stream.tell()
                stream.seek(max(0, end - 128))
                boundary = stream.read()
            self.offset, self.identity = end, identity
            self.modified, self.boundary = stat.st_mtime_ns, boundary
            return rows, reset


def read_rows(path):
    return JsonlTail(path).read()[0]
