#!/bin/bash
# Command-line version of examples/run_example.py (same Gulf of Mexico / 3-day example).
#   bash examples/run_example.sh            # all 72 hours
#   bash examples/run_example.sh --quick    # one hour
set -e
OUT=outputs/gulf_of_mexico

if [ "$1" = "--quick" ]; then
  coastdiff --timestamp "2026-10-09 12:00" --output-dir "$OUT"
else
  # These values are the defaults; --skip-missing keeps the run going if ERA5 lacks some hours.
  coastdiff --lat-min 24 --lat-max 31 --lon-min -98 --lon-max -80 \
            --start "2026-10-08 00:00" --end "2026-10-10 23:00" --interval 1h \
            --output-dir "$OUT" --skip-missing   # run the hours ERA5 has; missing hours are skipped and reported
fi
ls -1 "$OUT" | head
