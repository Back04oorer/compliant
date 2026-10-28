"""Shared terminal choices, editable fields and path browsing; no filesystem writes."""
import curses
import math
import os
from pathlib import Path
import sys


class Cancelled(Exception):
    """The operator returned from a form without executing it."""


MANUAL = object()
BROWSE = object()
CURRENT = object()
NEW = object()


def interactive():
    return sys.stdin.isatty() and sys.stdout.isatty()


def put(screen, row, text, attr=0):
    height, width = screen.getmaxyx()
    if 0 <= row < height:
        try:
            screen.addnstr(row, 1, str(text).replace('\n', ' '), max(0, width - 3), attr)
        except curses.error:
            pass


def prepare(screen):
    screen.keypad(True)
    # The live hardware menu uses timed reads; forms must wait for the operator.
    screen.timeout(-1)
    screen.clear()
    try:
        curses.curs_set(0)
    except curses.error:
        pass


def checked(value, validate):
    if not str(value).strip():
        raise ValueError('A value is required')
    return validate(value) if validate else value


def edit(screen, label, default='', validate=None):
    prepare(screen)
    value, cursor, replace = str(default), len(str(default)), True
    error = ''
    while True:
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 8 or width < 40:
            put(screen, 0, 'Resize terminal to at least 40x8; Esc returns')
        else:
            put(screen, 1, label, curses.A_BOLD)
            span = max(1, width - 5)
            first = max(0, cursor - span + 1)
            put(screen, 3, value[first:first + span] or ' ', curses.A_REVERSE if replace else 0)
            if not replace:
                put(screen, 4, ' ' * (cursor - first) + '^')
            put(screen, 5, error)
            put(screen, height - 2, 'Enter confirm | Esc back | Ctrl-U clear | Left/Right edit')
        screen.refresh()
        try:
            key = screen.get_wch()
        except curses.error:
            continue
        if key in ('\x1b', '\x03'):
            raise Cancelled()
        if height < 8 or width < 40:
            continue
        if key in ('\n', '\r', curses.KEY_ENTER):
            try:
                return checked(value.strip(), validate)
            except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
                error = str(exc)
                replace = False
        elif key == '\x15':
            value, cursor, replace = '', 0, False
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT, curses.KEY_HOME, curses.KEY_END):
            replace = False
            cursor = (max(0, cursor - 1) if key == curses.KEY_LEFT else
                      min(len(value), cursor + 1) if key == curses.KEY_RIGHT else
                      0 if key == curses.KEY_HOME else len(value))
        elif key in ('\b', '\x7f', curses.KEY_BACKSPACE):
            if replace:
                value, cursor = '', 0
            elif cursor:
                value, cursor = value[:cursor - 1] + value[cursor:], cursor - 1
            replace = False
        elif key == curses.KEY_DC:
            value, replace = value[:cursor] + value[cursor + 1:], False
        elif isinstance(key, str) and key.isprintable():
            if replace:
                value, cursor = '', 0
            value, cursor, replace = value[:cursor] + key + value[cursor:], cursor + 1, False


def pick(screen, label, entries, default=None, error='', paths=False):
    """Entries are (value, label); the same navigation serves enums and paths."""
    prepare(screen)
    selected = next((i for i, (value, _) in enumerate(entries) if value == default), 0)
    query, filtering = '', False
    while True:
        visible = [(value, text) for value, text in entries if query.casefold() in str(text).casefold()]
        selected = min(selected, max(0, len(visible) - 1))
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 8 or width < 40:
            put(screen, 0, 'Resize terminal to at least 40x8; Esc returns')
        else:
            put(screen, 0, label, curses.A_BOLD)
            put(screen, 1, 'Filter: ' + query + ('_' if filtering else ''))
            capacity = max(1, height - 6)
            first = max(0, selected - capacity + 1)
            for row, (_, text) in enumerate(visible[first:first + capacity], 2):
                put(screen, row, ('> ' if first + row - 2 == selected else '  ') + str(text),
                    curses.A_REVERSE if first + row - 2 == selected else 0)
            if not visible:
                put(screen, 2, 'No matches; change the filter' if entries else 'No items')
            put(screen, height - 3, error)
            put(screen, height - 2, 'Up/Down Enter | / filter | Esc back' + (' | Tab path' if paths else ''))
        screen.refresh()
        try:
            key = screen.get_wch()
        except curses.error:
            continue
        if key in ('\x1b', '\x03'):
            raise Cancelled()
        if height < 8 or width < 40:
            continue
        if key == '\t' and paths:
            return MANUAL
        if key == curses.KEY_UP and visible:
            selected = (selected - 1) % len(visible)
        elif key == curses.KEY_DOWN and visible:
            selected = (selected + 1) % len(visible)
        elif key == curses.KEY_PPAGE:
            selected = max(0, selected - capacity)
        elif key == curses.KEY_NPAGE:
            selected = min(max(0, len(visible) - 1), selected + capacity)
        elif key in ('\n', '\r', curses.KEY_ENTER) and visible:
            return visible[selected][0]
        elif key == '/' and not filtering:
            query, filtering, selected = '', True, 0
        elif filtering and key in ('\b', '\x7f', curses.KEY_BACKSPACE):
            query, selected = query[:-1], 0
        elif filtering and key == '\x15':
            query, selected = '', 0
        elif filtering and isinstance(key, str) and key.isprintable():
            query, selected = query + key, 0


def text(label, default='', validate=None):
    try:
        if interactive():
            return curses.wrapper(edit, label, default, validate)
        while True:
            raw = input(label + (f' [{default}]' if default != '' else '') + ': ').strip()
            if raw in ('\x1b', ':back'):
                raise Cancelled()
            try:
                return checked(raw or str(default), validate)
            except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
                print('Invalid: ' + str(exc) + '; edit this item, or :back to return.')
    except KeyboardInterrupt:
        raise Cancelled() from None


def choice(label, options, default):
    if interactive():
        try:
            return curses.wrapper(pick, label, [(v, v) for v in options], default)
        except KeyboardInterrupt:
            raise Cancelled() from None
    def validate(value):
        value = value.lower()
        if value not in options:
            raise ValueError('Choose ' + '/'.join(options))
        return value
    return text(label + ' (' + '/'.join(options) + ')', default, validate)


def number(label, default, minimum=None, maximum=None, integer=False, exclusive_minimum=False):
    def validate(value):
        result = int(value) if integer else float(value)
        if not math.isfinite(result) or (minimum is not None and (result <= minimum if exclusive_minimum else result < minimum)) or (maximum is not None and result > maximum):
            raise ValueError('Value must be finite' + (f', {">" if exclusive_minimum else ">="} {minimum}' if minimum is not None else '') +
                             (f', <= {maximum}' if maximum is not None else ''))
        return result
    return text(label, default, validate)


def image_size(label, default, even_width=False):
    def validate(value):
        parts = value.lower().split('x')
        if len(parts) != 2 or any(not p.isdigit() or int(p) <= 0 for p in parts):
            raise ValueError('Use positive WIDTHxHEIGHT, e.g. 680x480')
        width, height = map(int, parts)
        if even_width and width % 2:
            raise ValueError('Side-by-side width must be even')
        return width, height
    return text(label, default, validate)


def resolve_path(value, base):
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
    if not value:
        raise ValueError('A path is required')
    path = Path(os.path.expandvars(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def path(label, root, kind='file', default='', candidates=(), suffix='', new=False, allow_new=False, validate=None):
    root = Path(root).resolve()
    def accept(value, base=root):
        selected = resolve_path(value, base)
        if new:
            if selected.exists() or selected.is_symlink():
                raise ValueError('Output already exists; choose a new path')
        elif not (selected.is_file() if kind == 'file' else selected.is_dir()) and not (allow_new and not selected.exists()):
            raise ValueError('Choose an existing ' + kind)
        if (new or allow_new) and not selected.exists():
            parent = selected.parent
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            if not parent.is_dir():
                raise ValueError('The output parent must be a directory')
        if suffix and selected.suffix.lower() != suffix:
            raise ValueError('Choose a ' + suffix + ' file')
        if validate:
            validate(selected)
        return selected
    if not interactive():
        for candidate in candidates:
            print('  ' + str(candidate))
        # Pasted relative paths retain the command-line working-directory meaning.
        return text(label, default, lambda value: accept(value, Path.cwd()))

    def browse(screen):
        current, catalog, error = root, list(candidates), ''
        history = []
        selection = Path(default).resolve() if default else None
        while not current.is_dir() and current != current.parent:
            current = current.parent
        while True:
            entries = []
            if catalog:
                entries = [(Path(p).resolve(), os.path.relpath(p, root)) for p in catalog]
                entries += [(BROWSE, 'Browse directories...')]
            else:
                entries = [(current.parent, '../')]
                if kind == 'directory' and not new:
                    entries.append((CURRENT, 'Select this directory'))
                if new or allow_new:
                    entries.append((NEW, 'New directory name here...'))
                try:
                    children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
                    entries += [(p, p.name + ('/' if p.is_dir() else '')) for p in children
                                if p.is_dir() or (kind == 'file' and (not suffix or p.suffix.lower() == suffix))]
                except OSError as exc:
                    error = str(exc)
            entries.append((MANUAL, 'Enter a path... [Tab]'))
            try:
                selected = pick(screen, label + ' | ' + str(current), entries, selection, error, paths=True)
            except Cancelled:
                if not history:
                    raise
                current, catalog, selection = history.pop()
                error = ''
                continue
            error, selection = '', selected
            if selected is BROWSE:
                history.append((current, catalog, selection))
                catalog, selection = [], None
                continue
            if selected is MANUAL or selected is NEW:
                try:
                    return edit(screen, label if selected is MANUAL else 'New directory name',
                                str(default) if selected is MANUAL else '', lambda value: accept(value, current))
                except Cancelled:
                    continue
            if selected is CURRENT:
                selected = current
            elif not catalog and selected.is_dir():
                history.append((current, catalog, selection))
                current, selection = selected, None
                continue
            try:
                return accept(selected)
            except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
                error = str(exc)
    try:
        return curses.wrapper(browse)
    except KeyboardInterrupt:
        raise Cancelled() from None
