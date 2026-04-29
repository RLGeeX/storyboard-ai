"""Parse a structured markdown script file into a video_plan dict.

The expected file format is:

    # Title

    **Target length:** ~1:50
    **Style:** White hand-drawn chalk on black background. ...
    **Tagline at close:** *Climb Faster. Chain Mountain.*

    ---

    ## Scene 1 — Beat description (~Ns)

    **Narration**

    Spoken narration text, used VERBATIM in TTS.

    **On-screen text**

    Text to render on the chalkboard during this scene.

    **Visual cue**

    Description of what should be drawn — fed into the image prompter.

    ---

    ## Scene 2 — ...

The output dict matches the shape `director_tool_fn` returns, so the rest of
the pipeline (image_prompt_tool, image_gen, segmentation, animation, TTS,
merge, concat) consumes it without modification. The key difference: when
this parser produces the plan, the LLM Director never runs, so the narration
in each scene is exactly what the user wrote.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


# Regex parts kept readable.
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# Scene header looks like: "## Scene 1 — Pain: data sprawl, ... (~18s)"
# Accepts em-dash (—) and ASCII hyphen (-) as the separator.
_SCENE_RE = re.compile(
    r"^##\s+Scene\s+(?P<num>\d+)\s*[—\-]\s*(?P<beat>[^\n(]+?)(?:\s*\((?P<dur>~?[^)]+)\))?\s*$",
    re.MULTILINE,
)
# Frontmatter key/value: **Key:** value (single-line) or up to next **Key:** / --- / next ##.
_FRONT_RE = re.compile(
    r"^\*\*(?P<key>[^*:]+):\*\*\s*(?P<val>.+?)(?=\n\*\*[^*:]+:\*\*|\n---|\n##|\Z)",
    re.MULTILINE | re.DOTALL,
)


def parse_script_file(path: str) -> Dict[str, Any]:
    """Parse a structured markdown script file into a video_plan dict.

    Returns the same shape `director_tool_fn` returns:
      {
        "global_plan": {... metadata fields ..., "style_preamble": str},
        "scenes": [{"scene_number": int, "narration": str,
                    "description": str, "visual_setup": str,
                    "text_overlay": str, "emotional_beat": str, ...}, ...]
      }

    Raises:
      FileNotFoundError if the path doesn't exist.
      ValueError if no scenes can be parsed.
    """
    text = Path(path).read_text(encoding="utf-8")

    # Title (first H1).
    title_match = _TITLE_RE.search(text)
    title = title_match.group(1).strip() if title_match else "Untitled Video"

    # Frontmatter: **Key:** value entries that appear before the first scene header.
    first_scene = _SCENE_RE.search(text)
    front_text = text[: first_scene.start()] if first_scene else text
    frontmatter: Dict[str, str] = {}
    for m in _FRONT_RE.finditer(front_text):
        key = m.group("key").strip().lower()
        val = m.group("val").strip()
        # Strip surrounding *italics* and trailing punctuation noise.
        val = val.strip().strip("*").strip()
        frontmatter[key] = val

    style = frontmatter.get("style", "")
    target_length = frontmatter.get("target length", "")
    tagline = frontmatter.get("tagline at close") or frontmatter.get("tagline", "")

    # Scenes.
    scene_starts = list(_SCENE_RE.finditer(text))
    if not scene_starts:
        raise ValueError(
            f"No scenes found in {path}. Expected '## Scene N — beat (...)'"
        )

    scenes: List[Dict[str, Any]] = []
    for i, m in enumerate(scene_starts):
        block_start = m.end()
        block_end = scene_starts[i + 1].start() if i + 1 < len(scene_starts) else len(text)
        body = text[block_start:block_end]

        narration = _extract_labeled_block(body, "Narration")
        onscreen = _extract_labeled_block(body, "On-screen text")
        visual = _extract_labeled_block(body, "Visual cue")

        if not narration:
            raise ValueError(
                f"Scene {m.group('num')} in {path} has no '**Narration**' block"
            )
        if not visual:
            raise ValueError(
                f"Scene {m.group('num')} in {path} has no '**Visual cue**' block"
            )

        scene_num = int(m.group("num"))
        beat_summary = m.group("beat").strip()
        duration = m.group("dur").strip() if m.group("dur") else ""

        scenes.append({
            "scene_number": scene_num,
            "summary": beat_summary,
            "narration": narration,
            "description": visual,        # consumed by image_prompt_tool as scene_description
            "visual_setup": visual,       # composition hint; using same source by default
            "search_query": "",
            "text_overlay": onscreen,
            "key_information": "",
            "emotional_beat": _infer_emotional_beat(beat_summary),
            "duration_hint": duration,
        })

    return {
        "global_plan": {
            "title": title,
            "tone": "informative",
            "narrative_persona": "Professional Explainer",
            "visual_style": style or "Clean Whiteboard Animation",
            "pacing": "steady",
            "narrative_arc": "Author-controlled (parsed from script file)",
            "target_audience": "general business",
            "target_length": target_length,
            "tagline": tagline,
            "total_scenes": len(scenes),
            # Custom fields the script-file path adds:
            "style_preamble": style,      # injected into image prompts
            "source": "script_file",
        },
        "scenes": scenes,
    }


def _extract_labeled_block(body: str, label: str) -> str:
    """Extract a labeled block from a scene body.

    Format:
        **Label**

        Paragraph text, possibly multi-line.

    The block ends at the next **Label** of the same depth, or the next
    scene header (##), or the next ---, or end of body.
    """
    pattern = (
        rf"\*\*{re.escape(label)}\*\*\s*\n+"
        rf"(?P<body>.+?)"
        rf"(?=\n\s*\*\*[^*:]+\*\*|\n##\s|\n---|\Z)"
    )
    m = re.search(pattern, body, re.DOTALL)
    if not m:
        return ""
    return m.group("body").strip()


def _infer_emotional_beat(summary: str) -> str:
    """Best-effort mapping from a beat summary to a one-word emotional tone."""
    s = summary.lower()
    if any(w in s for w in ("pain", "problem", "fragment", "chaos", "sprawl", "wait")):
        return "frustrated"
    if any(w in s for w in ("promise", "imagine", "ideal", "what if")):
        return "hopeful"
    if any(w in s for w in ("intro", "introducing", "meet", "name the thing")):
        return "confident"
    if any(w in s for w in ("payoff", "scale", "summit", "tagline", "sign-off")):
        return "triumphant"
    if any(w in s for w in ("trust", "secure", "audit", "boundary")):
        return "reassuring"
    if any(w in s for w in ("differentiator", "factory", "ai")):
        return "transformative"
    return "neutral"
