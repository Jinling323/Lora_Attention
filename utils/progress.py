"""Small terminal progress display without additional dependencies."""
import sys
import time


class progress_bar:
    def __init__(self, iterable, desc='', unit='step', leave=True, **kwargs):
        self.iterable = iterable
        self.total = len(iterable)
        self.desc, self.unit, self.leave = desc, unit, leave
        self.postfix = ''

    def set_postfix(self, **values):
        self.postfix = ' '.join(f'{key}={value}' for key, value in values.items())

    def __iter__(self):
        start = last = time.monotonic()
        self._show(0, start)
        try:
            for count, item in enumerate(self.iterable, 1):
                yield item
                now = time.monotonic()
                if now - last >= 1 or count == self.total:
                    self._show(count, start)
                    last = now
        finally:
            print(file=sys.stderr, flush=True)

    def _show(self, count, start):
        elapsed = time.monotonic() - start
        fraction = count / self.total if self.total else 1
        filled = int(20 * fraction)
        eta = elapsed / count * (self.total - count) if count else 0
        print(f'\r{self.desc} [{"=" * filled}{"." * (20 - filled)}] '
              f'{count}/{self.total} {self.unit} ({fraction:.0%}) '
              f'elapsed={elapsed:.0f}s ETA={eta:.0f}s {self.postfix}   ',
              end='', file=sys.stderr, flush=True)
