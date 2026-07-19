"""The align tuning knobs: every magic threshold of the matching/position machinery in one
place, with the reasoning that set it. Modules import what they consult."""
from __future__ import annotations

# A cell claims a hint line only when at least this fraction of its tokens match the line.
_MATCH_THRESHOLD = 0.4

# A confident label may not sit further than this many hint lines from the position estimate.
_POSITION_GUARD = 3.0

# A tied cell (its token sits in several hint lines, e.g. a "dieren" shared by four sentences) is
# resolved by its line-neighbours: the nearest CONFIDENT cell touching it on the left/right of the
# same printed line. A neighbour counts when it overlaps vertically by this fraction of the shorter
# height (same line, tilt-tolerant) and sits within this gap (a word space, not a column gap — so a
# receipt's far label/amount never link).
_LINE_VOVERLAP_RATIO = 0.4
_LINE_GAP_RATIO = 1.2

# A hint block's column is trusted only when this fraction of its matching cells' score mass
# agrees on one column.
_HINT_COLUMN_MAJORITY = 0.6

# Only a WEAK cross-column match is dropped by the column filter: a match at/above this score
# means the cell genuinely carries the block's text (a header or byline the layout repeats in the
# other column, a fragment that reads verbatim) and is kept wherever it sits. The mislabel this
# guards is a partial match — a cell sharing one stray token ("in") with a far-column block —
# which scores well below it. Set between the two: the dropped-paragraph orphans measure ~0.5,
# genuine repeats ~1.0.
_COLUMN_FILTER_MAX_SCORE = 0.75
