"""Terminal progress display for a screening sweep.

One cell at a time: a header naming the cell and its position in the sweep, and a
bar underneath that is redrawn in place as generations land.

    Qwen | thinking OFF | HLE | Cell 30/32
      [████████████░░░░░░░░░░░░░░░░░░]  42% | 126/300 | 51/min | ~3m 20s left

Redrawing in place needs a terminal. When stdout is not one — a sweep piped to a
log file, or a CI job — the same numbers are printed as one plain line every
PROGRESS_EVERY generations instead. That is what the runner did before the bar
existed, and it keeps a redirected log readable rather than filling it with
carriage returns.
"""

from __future__ import annotations

import shutil
import sys
import time
from collections import deque

# Widest the bar is ever drawn, and the narrowest it may be squeezed to. The
# counters beside it are what carries the information, so on a narrow terminal the
# bar gives up its width first rather than pushing the line into a wrap.
BAR_WIDTH = 30
MIN_BAR_WIDTH = 8

# Redraws are capped to this interval. At 32-way concurrency several generations
# can land in the same millisecond, and repainting for each of them is all cost.
REDRAW_INTERVAL_S = 0.1

# How often the no-terminal fallback prints. Without it a 300-generation cell
# writes nothing to a redirected log for half an hour and looks like a hang.
PROGRESS_EVERY = 25

# Rate and time remaining are measured over this trailing window, not over the
# whole cell. A resumed sweep serves its cached prefix in milliseconds, and an
# average taken from the start would carry that burst into the estimate for every
# live generation that follows it — reporting minutes for work that will take
# hours, which is worse than reporting nothing.
RATE_WINDOW_S = 60.0

# Block characters where the stream can encode them, ASCII where it cannot: an
# ASCII-only stdout would otherwise raise UnicodeEncodeError mid-sweep.
BLOCK_GLYPHS = ("█", "░")
ASCII_GLYPHS = ("#", "-")


# Pick the bar characters this stream is actually able to write.
def _glyphs(stream) -> tuple[str, str]:
    encoding = getattr(stream, "encoding", None) or ""
    try:
        "".join(BLOCK_GLYPHS).encode(encoding)
    except (LookupError, UnicodeEncodeError, TypeError):
        return ASCII_GLYPHS
    return BLOCK_GLYPHS


# Seconds as something readable at a glance, at the precision the number deserves.
def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes, whole_seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {whole_seconds:02d}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


# The live progress line for one (benchmark, model, thinking mode) cell.
#
# Owns exactly one terminal row at a time. Everything drawn goes to that row until
# close() ends it, so a caller that prints while a bar is live must clear() first.
class CellProgress:
    def __init__(self, header: str, stream=None) -> None:
        self.stream = stream or sys.stdout
        # A pipe or a file has no cursor to move, so fall back to plain lines.
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self.filled_glyph, self.empty_glyph = _glyphs(self.stream)

        self.total = 0
        self.completed = 0
        self.note = ""
        self._started = time.monotonic()
        self._last_draw = 0.0

        # (time, completions) pairs inside the rate window, oldest first.
        self._samples: deque[tuple[float, int]] = deque()

        # Length of what is currently on the live row, so the next draw knows how
        # much of a longer previous line it has to paint over.
        self._line_len = 0

        # The header is a normal line: it stays on screen above the bar.
        print(header, file=self.stream, flush=True)

    # Show a transient message where the bar goes, for a phase with no count yet.
    def status(self, text: str) -> None:
        if self.enabled:
            self._draw(f"  {text}", force=True)
        else:
            print(f"  {text}", file=self.stream, flush=True)

    # Switch to the bar, now that the size of the cell is known.
    def begin(self, total: int, note: str = "") -> None:
        self.total = max(0, total)
        self.completed = 0
        self.note = note
        # Timed from here, not from construction: the dataset load that precedes
        # it is not generation and would drag the rate estimate down for minutes.
        self._started = time.monotonic()
        self._samples.clear()
        self._samples.append((self._started, 0))
        self._render(force=True)

    # Record finished generations and repaint.
    def advance(self, count: int = 1) -> None:
        self.completed += count
        self._sample()
        self._render(force=self.completed >= self.total)

    # Note the current count, and drop whatever has fallen out of the window.
    def _sample(self) -> None:
        now = time.monotonic()
        self._samples.append((now, self.completed))

        # Never below two samples: a benchmark whose generations take longer than
        # the window would otherwise have nothing left to measure a rate from.
        while len(self._samples) > 2 and now - self._samples[0][0] > RATE_WINDOW_S:
            self._samples.popleft()

    # Completions per second across the window the samples currently span.
    def _rate(self) -> float:
        if len(self._samples) < 2:
            return 0.0

        (oldest_at, oldest_done) = self._samples[0]
        (newest_at, newest_done) = self._samples[-1]
        span = newest_at - oldest_at
        return (newest_done - oldest_done) / span if span > 0 else 0.0

    # Erase the live row, so whatever prints next starts on a clean one.
    def clear(self) -> None:
        if self.enabled and self._line_len:
            self.stream.write("\r" + " " * self._line_len + "\r")
            self.stream.flush()
        self._line_len = 0

    # End the live row, leaving whatever it last showed on screen.
    def close(self) -> None:
        if self.enabled and self._line_len:
            self.stream.write("\n")
            self.stream.flush()
        self._line_len = 0

    # Paint the current counts, in place on a terminal and periodically otherwise.
    def _render(self, force: bool = False) -> None:
        if self.enabled:
            self._draw(self._bar_line(), force=force)
            return

        if self.completed % PROGRESS_EVERY == 0 or self.completed >= self.total:
            print(f"  {self._counts()}", file=self.stream, flush=True)

    # The counters shared by both output modes.
    def _counts(self) -> str:
        percent = 100.0 * self.completed / self.total if self.total else 100.0
        rate = self._rate()
        counted = f"{self.completed}/{self.total}"
        if self.note:
            counted += f" ({self.note})"

        # Once the cell is done, what it actually took replaces the estimate of
        # what was left, and the rate becomes the cell's average rather than the
        # window's — the window at that point covers only the final stragglers.
        if self.completed >= self.total:
            elapsed = time.monotonic() - self._started
            average = self.completed / elapsed if elapsed > 0 else 0.0
            return (
                f"{percent:3.0f}% | {counted} | "
                f"{average * 60:,.0f}/min | {_format_duration(elapsed)}"
            )

        # Before the first completion there is no rate to extrapolate from, and a
        # made-up estimate at that moment is the one most likely to be quoted.
        if rate <= 0:
            return f"{percent:3.0f}% | {counted} | starting..."

        remaining = (self.total - self.completed) / rate
        return (
            f"{percent:3.0f}% | {counted} | "
            f"{rate * 60:,.0f}/min | ~{_format_duration(remaining)} left"
        )

    # The bar plus its counters, sized to the terminal it is going to.
    def _bar_line(self) -> str:
        counts = self._counts()

        # What is left after the two leading spaces, the brackets, the space
        # before the counters, and the column _draw keeps clear of the edge.
        spare = shutil.get_terminal_size(fallback=(80, 24)).columns - len(counts) - 6
        width = max(MIN_BAR_WIDTH, min(BAR_WIDTH, spare))

        fraction = self.completed / self.total if self.total else 1.0
        filled = int(round(min(1.0, fraction) * width))
        bar = self.filled_glyph * filled + self.empty_glyph * (width - filled)
        return f"  [{bar}] {counts}"

    # Write one line to the live row.
    def _draw(self, line: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_draw < REDRAW_INTERVAL_S:
            return
        self._last_draw = now

        # Clipped one column short of the terminal, so a full-width line cannot
        # wrap and strand the previous redraw on the row above.
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        clipped = line[: max(0, width - 1)]
        padding = " " * max(0, self._line_len - len(clipped))
        self._line_len = len(clipped)

        self.stream.write("\r" + clipped + padding)
        self.stream.flush()
