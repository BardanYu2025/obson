"""Witnessed session and split rules copied from the qualified month snapshot."""

import numpy as np


def ages(links, valid):
    """Number of consecutive usable observations ending at each row, not elapsed minutes."""
    i = np.arange(len(valid), dtype=np.int32)
    starts = np.maximum.accumulate(np.where(~links | ~valid, i, 0))
    return np.where(valid, i - starts + 1, 0).astype(np.int32)


def boundaries(times, main_clock, map_days, financial, good):
    n = len(times)
    kind = np.ones(n, dtype=np.uint8)
    kind[0] = 0
    gaps = np.flatnonzero(np.diff(times) != 1) + 1
    if len(gaps):
        left, right = times[gaps - 1], times[gaps]
        a, b = np.searchsorted(main_clock, left), np.searchsorted(main_clock, right)

        def matches(k, x):
            return (k < len(main_clock)) & (
                main_clock[np.minimum(k, len(main_clock) - 1)] == x
            )

        paired = matches(a, left) & matches(b, right)
        witness = paired & (b == a + 1)
        day0, hm0 = np.divmod(left, 1440)
        day1, hm1 = np.divmod(right, 1440)
        is_day0 = np.isin(day0, map_days)
        is_day1 = np.isin(day1, map_days)
        morning = 570 if financial else 540
        mids = (hm0 == 689) & (hm1 == (780 if financial else 810))
        if not financial:
            mids |= (hm0 == 614) & (hm1 == 630)
        recess = (day0 == day1) & is_day0 & mids
        # Official mapping trading dates pair with actual product-clock neighbours.
        # In particular, a weekend is never inferred from elapsed wall time alone.
        after0 = np.searchsorted(map_days, day0, side="right")
        nextday = (after0 < len(map_days)) & (
            map_days[np.minimum(after0, len(map_days) - 1)] == day1
        )
        day_to_night = (hm0 == 899) & (hm1 == 1260) & (day0 == day1) & is_day0
        day_to_day = (hm0 == 899) & (hm1 == morning) & nextday & is_day0
        eve_to_day = (
            (hm0 >= 1260)
            & np.isin(hm0, [1379, 1409, 1439])
            & (hm1 == morning)
            & nextday
            & is_day0
        )
        # A post-midnight night close belongs to the next mapped trading day
        # after the evening's civil date, including a Friday evening.
        evening_day = day0 - 1
        ni = np.searchsorted(map_days, evening_day, side="right")
        after_evening = (ni < len(map_days)) & (
            map_days[np.minimum(ni, len(map_days) - 1)] == day1
        )
        late_to_day = (
            np.isin(hm0, [59, 89, 149])
            & (hm1 == morning)
            & after_evening
            & np.isin(evening_day, map_days)
        )
        night = day_to_night | eve_to_day | late_to_day
        session = day_to_day | (night if not financial else False)
        kind[gaps] = np.select(
            [witness & recess, witness & is_day1 & session, paired & (b > a + 1)],
            [2, 3, 4],
            default=5,
        )
        # The evening date itself is mapped for a day-to-night transition.
    if n > 1:
        kind[1:][~(good[1:] & good[:-1])] = 6
    strict = (kind == 1) | (kind == 2)
    screened = strict | (kind == 3)
    return kind, strict, screened


def split_codes(times, hold, cutoff):
    # Local-clock minute starts. The old source cutoff is a minute START, hence <= cutoff is development.
    y24 = int(np.datetime64("2024-01-01", "m").astype(int))
    y25 = int(np.datetime64("2025-01-01", "m").astype(int))
    return np.select(
        [times > cutoff, times >= y25, times >= y24], [3, 2, 1], default=0
    ).astype(np.uint8) + 4 * int(hold)


def session_spans(times, financial):
    """Local minute grid, restarting each session. Night grid crosses civil midnight."""
    day, hm = np.divmod(times, 1440)
    spans = (
        [(570, 690), (780, 900)] if financial else [(540, 615), (630, 690), (810, 900)]
    )
    origin = np.full(len(times), -1, np.int64)
    end = np.full(len(times), -1, np.int64)
    for a, b in spans:
        match = (hm >= a) & (hm < b)
        origin[match] = day[match] * 1440 + a
        end[match] = day[match] * 1440 + b
    if not financial:
        night = (hm >= 1260) | (hm < 150)
        origin[night] = (day[night] - (hm[night] < 150)) * 1440 + 1260
        # Only full night bins. No present-day night close is backcast to an earlier year.
        end[night] = origin[night] + 600
    return origin, end
