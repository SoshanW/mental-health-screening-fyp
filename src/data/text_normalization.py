"""Text / orthography normalization — a first-class, separated concern.

Single responsibility: own the policy for cleaning text register so the
normalization rules live in exactly one place and can be applied uniformly to any
source (DAIC clinical transcripts today; Reddit posts in the future).

DAIC-WoZ transcripts follow the Artstein (2012) transcription conventions: all
lowercase, no punctuation, numbers spelled out, disfluencies kept (``i i i
think``, ``um``), non-speech events bracketed, and ``xxx`` for unintelligible
speech. This differs from raw Reddit text (mixed case, punctuated), and that
mismatch is part of the Reddit->clinical shift being studied. Loaders therefore
*depend on* this module instead of implementing cleaning inline.

DECISION (bracket style): the spec described angle-bracket non-speech tags
(``<laughter>``), but the real DAIC-WoZ (AVEC2017) files on disk use *square*
brackets (e.g. ``[laughter]``). To be robust we strip BOTH angle- and
square-bracketed spans by default. Confirm this matches your full corpus.

DECISION (truecasing default): re-punctuation / truecasing is OFF by default.
Normalized DAIC text is left lowercase and unpunctuated, preserving the genuine
register gap from Reddit. The toggle exists so the gap can be ablated later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches a non-speech event wrapped in angle or square brackets, e.g.
# "<laughter>", "[laughter]", "<sigh>", "[cough]". Non-greedy, no nested brackets.
_BRACKET_TAG_RE = re.compile(r"<[^<>]*>|\[[^\[\]]*\]")

# Matches the unintelligible-speech token "xxx" as a whole word (any casing).
_XXX_RE = re.compile(r"\bx{3,}\b", re.IGNORECASE)

# Collapses runs of whitespace left behind after removals.
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationPolicy:
    """Declarative, source-agnostic text-cleaning policy.

    Attributes:
        strip_bracket_tags: Remove ``<...>`` / ``[...]`` non-speech event tags.
        remove_xxx: Remove ``xxx`` unintelligible-speech tokens.
        truecase: If ``True``, apply :func:`truecase_and_punctuate`. Default OFF
            so normalized text keeps its lowercase, unpunctuated register.
        collapse_whitespace: Collapse and trim whitespace after removals.
    """

    strip_bracket_tags: bool = True
    remove_xxx: bool = True
    truecase: bool = False
    collapse_whitespace: bool = True


#: Default policy: strip non-speech tags and ``xxx``, leave casing/punctuation
#: untouched. This is the policy the DAIC loader uses unless overridden.
DEFAULT_POLICY: NormalizationPolicy = NormalizationPolicy()

#: An explicit no-op policy. Useful for the Reddit loaders, where normalization
#: is currently a pass-through but routed through this module so the policy still
#: lives in one place.
NOOP_POLICY: NormalizationPolicy = NormalizationPolicy(
    strip_bracket_tags=False,
    remove_xxx=False,
    truecase=False,
    collapse_whitespace=False,
)


def strip_bracket_tags(text: str) -> str:
    """Remove angle- and square-bracketed non-speech tags (``<laughter>``, ``[cough]``)."""
    return _BRACKET_TAG_RE.sub(" ", text)


def remove_xxx_tokens(text: str) -> str:
    """Remove ``xxx`` unintelligible-speech tokens (whole-word, any casing)."""
    return _XXX_RE.sub(" ", text)


def truecase_and_punctuate(text: str) -> str:
    """Optional truecasing / re-punctuation hook (default OFF).

    DECISION: this is intentionally a minimal placeholder. True truecasing would
    require a model (e.g. a seq2seq recaser); wiring that in is out of scope for
    the data layer. For now, enabling the toggle only capitalizes sentence-less
    text's first character, so flipping the flag has an observable but cheap
    effect. Replace with a real recaser when the ablation is run.
    """
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def normalize_text(text: str | None, policy: NormalizationPolicy = DEFAULT_POLICY) -> str:
    """Apply ``policy`` to ``text`` and return the cleaned string.

    ``None`` is treated as empty. With the default policy this strips non-speech
    tags and ``xxx`` and collapses whitespace, while leaving casing and
    punctuation untouched.
    """
    if text is None:
        return ""
    result = str(text)
    if policy.strip_bracket_tags:
        result = strip_bracket_tags(result)
    if policy.remove_xxx:
        result = remove_xxx_tokens(result)
    if policy.truecase:
        result = truecase_and_punctuate(result)
    if policy.collapse_whitespace:
        result = _WS_RE.sub(" ", result).strip()
    return result
