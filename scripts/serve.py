"""Запасной сервер для out/ без Docker.

ЗАЧЕМ. Docker Desktop на этой машине падал дважды посреди работы, и вместе
с ним ложились все страницы. Демонстрация не должна зависеть от того, жив ли
докер.

ПОЧЕМУ НЕ `python -m http.server`. Он не реализует HTTP Range. Страница
реплея перематывает часовое видео, и без Range браузер тянет весь файл
целиком и не умеет искать по нему. Здесь Range реализован.

    python scripts/serve.py              # http://localhost:8080
    python scripts/serve.py --port 9000
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler плюс Range и корректный index."""

    def send_head(self):
        path = Path(self.translate_path(self.path))
        # Корень отдаёт дашборд НАПРЯМУЮ, без редиректа: редирект строил бы
        # абсолютный адрес и терял порт, как это делал nginx.
        if path.is_dir():
            idx = path / "dashboard.html"
            if idx.is_file():
                path = idx
            else:
                return super().send_head()

        rng = self.headers.get("Range")
        if not rng or not path.is_file():
            return super().send_head()

        m = RANGE_RE.match(rng.strip())
        if not m:
            return super().send_head()

        size = path.stat().st_size
        start_s, end_s = m.group(1), m.group(2)
        if start_s:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        else:                       # суффиксная форма: bytes=-N
            if not end_s:
                # Reason phrase уходит в статусную строку, которую http.server
                # кодирует latin-1 strict: кириллица там роняет обработчик, и
                # клиент не получает вообще никакого ответа.
                self.send_error(400, "malformed Range header")
                return None
            start, end = max(0, size - int(end_s)), size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        end = min(end, size - 1)

        f = path.open("rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self._range_remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        """Отдаём ровно запрошенный кусок, а не файл до конца."""
        remaining = getattr(self, "_range_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        self._range_remaining = None
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def end_headers(self):
        # Страницы перегенерируются: закешированная версия хуже отсутствующей.
        if "Cache-Control" not in self._headers_buffer_keys():
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _headers_buffer_keys(self):
        return b"".join(getattr(self, "_headers_buffer", [])).decode(
            "latin-1", "ignore")

    def log_message(self, fmt, *args):
        # self.path не существует, пока стартовая строка не разобрана: на битом
        # запросе обращение к нему давало AttributeError вместо честного 400.
        path = getattr(self, "path", "-")
        if not path.endswith((".jpg", ".png")):         # не засорять лог сеткой
            sys.stderr.write(f"{path} -> {args[1] if len(args) > 1 else ''}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("out"))
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args(argv)

    if not args.dir.is_dir():
        raise SystemExit(f"нет каталога {args.dir}")
    handler = partial(RangeHandler, directory=str(args.dir.resolve()))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"отдаю {args.dir.resolve()} на http://localhost:{args.port}")
    print("Range поддерживается: перемотка видео в реплее работает")
    print("остановить: Ctrl+C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
