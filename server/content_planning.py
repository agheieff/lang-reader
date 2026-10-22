"""Deterministic content blueprints that keep generated lessons structurally varied."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast, get_args

from server.schemas import (
    AgentBrief,
    ContentMove,
    FeedbackTag,
    GenerationContentPlan,
    SubjectDomain,
)

# A text may not share its subject domain with any of this many most recent texts. Counting
# texts, not domains, keeps a favourite domain returning every few texts instead of vanishing.
DOMAIN_WINDOW = 6
SUBJECT_DOMAINS = get_args(SubjectDomain)


@dataclass(frozen=True)
class _Archetype:
    key: str
    discourse_form: str
    perspective: str
    progression: str
    ending_shape: str


_ARCHETYPES = (
    _Archetype(
        "decision-dialogue",
        "A dialogue built from short, consequential turns rather than narrated group activity.",
        "Two speakers with different practical motives; neither is simply right.",
        "Introduce a concrete choice, reveal one cost of each option, and let the speakers revise "
        "their positions before choosing.",
        "A decision with a visible trade-off, not universal agreement or celebration.",
    ),
    _Archetype(
        "mistaken-assumption",
        "A first-person personal account focused on one tightly bounded incident.",
        "One narrator whose initial interpretation turns out to be incomplete.",
        "Open with a specific surprising detail, add two clues, then revise the narrator's "
        "understanding.",
        "A changed interpretation or remaining doubt, without a moral summary.",
    ),
    _Archetype(
        "micro-mystery",
        "A compact third-person scene with a small mystery grounded in ordinary details.",
        "Stay close to one person's observations and decisions.",
        "Present an unexplained detail, test at least two plausible explanations, and discover "
        "one concrete cause.",
        "A discovery that changes the next action; it need not make anyone happy.",
    ),
    _Archetype(
        "how-it-works",
        "A concise explanatory article using one concrete example rather than a broad overview.",
        "A neutral explainer addressing a curious reader.",
        "Start from a practical question, trace cause and effect through a real example, and "
        "contrast it with one common misconception.",
        "A qualified factual takeaway followed by one open question.",
    ),
    _Archetype(
        "message-chain",
        "A short chain of messages whose meaning changes as new information arrives.",
        "Two or three distinct voices with clear immediate goals.",
        "Begin with an ambiguous request, expose a misunderstanding through replies, and clarify "
        "the concrete next step.",
        "A specific plan or unanswered reply, not a narrated happy ending.",
    ),
    _Archetype(
        "contrast-report",
        "A descriptive report comparing two objects, methods, places, or moments.",
        "An attentive observer who distinguishes fact from opinion.",
        "Establish a useful comparison, give sensory or measurable details on both sides, and "
        "show why the difference matters.",
        "A nuanced preference or contrast, not a winner-takes-all conclusion.",
    ),
    _Archetype(
        "mini-interview",
        "A brief interview organized around specific questions and concrete answers.",
        "A curious interviewer and someone with firsthand experience.",
        "Move from what happened, to why, to one drawback or unresolved difficulty.",
        "An honest limitation or next question rather than generic praise.",
    ),
    _Archetype(
        "field-notes",
        "Diary or field notes covering a short interval through precise observations.",
        "One observer who notices change without organizing a group project.",
        "Accumulate several concrete details, notice a pattern, and offer a tentative explanation.",
        "A quiet image, prediction, or question; leave some uncertainty intact.",
    ),
    _Archetype(
        "speculative-consequence",
        "A near-future vignette built around one altered rule, tool, or capability.",
        "One person directly affected by both its benefit and its cost.",
        "Demonstrate the apparent solution, reveal an unintended consequence, and require a new "
        "choice.",
        "A complication or compromise, not effortless technological success.",
    ),
    _Archetype(
        "portrait",
        "A short profile of one person, place, or object through concrete details and one scene.",
        "An observer who has spent time with the subject and reports what stood out.",
        "Open on a characteristic detail, add two contrasting facts about its history or habits, "
        "then show it in use.",
        "An image or an unexpected fact that deepens the portrait; nothing needs to go wrong or "
        "be solved.",
    ),
    _Archetype(
        "practical-guide",
        "Advice from someone experienced, organized around a few concrete tips and their reasons.",
        "One knowledgeable voice speaking directly to a beginner.",
        "Give three specific tips, each with the reason behind it and a small example; no "
        "failure is required.",
        "A specific recommendation with a reason, not a summary of the tips.",
    ),
    _Archetype(
        "recollection",
        "A reflective memory anchored in one object or place and one short exchange.",
        "One narrator remembering, who also says how the memory looks now.",
        "Place the memory in time, recall a vivid detail and what was said, then say what is "
        "different today.",
        "A quiet comparison between then and now; no lesson is stated.",
    ),
    _Archetype(
        "process-interrupted",
        "A practical process shown through action, including a meaningful interruption.",
        "One learner or worker solving a specific problem without a cheering group.",
        "Follow two or three concrete steps, let one fail for an intelligible reason, and diagnose "
        "what must change.",
        "A corrected step or useful warning, not 'everyone was satisfied.'",
    ),
)

# Content moves: mostly what the learner likes, plus deliberate tests of variations and new
# subjects so preferences can evolve. The agent picks the subject; the host only picks the move.
MOVE_WEIGHTS = (("favourite", 0.60), ("variation", 0.25), ("new", 0.15))
_MOVE_INSTRUCTIONS = {
    "favourite": (
        "Choose a subject the learner clearly enjoys according to brief.reading_preferences, "
        "preferring one that brief.content_history shows was used least recently, in a "
        "subject_domain outside content_plan.avoid_domains. Set hypothesis to null."
    ),
    "variation": (
        "Take one interest from brief.reading_preferences and explore a different sub-area, genre, "
        "period, or angle of it than recent texts in brief.content_history did, in a "
        "subject_domain outside content_plan.avoid_domains. Set hypothesis to "
        "the preference question this text tests, addressed to the learner, e.g. 'Do you enjoy "
        "the engineering side of space travel more than the exploration stories?'."
    ),
    "new": (
        "Choose a subject that brief.reading_preferences does not list but that plausibly suits "
        "the learner: an adjacent field, a surprising connection, or something untried, in a "
        "subject_domain outside content_plan.avoid_domains. Never pick anything the notes say "
        "they dislike. Set hypothesis to the preference question this "
        "text tests, addressed to the learner, e.g. 'Would you like more texts about maritime "
        "history?'."
    ),
}
_CHOSEN_SEED = (
    "The agent chooses the subject according to move_instruction, then narrows it to one "
    "specific incident, object, question, or disagreement with a clear time and place."
)

_STORY_SHAPE_WARNING = (
    "A liked story mechanic is not a subject: do not build another text on a small malfunction, "
    "an unexplained reading, or a mystery solved from physical evidence merely because "
    "brief.reading_preferences or earlier liked texts feature one."
)

_GENERIC_PATTERN_WARNING = (
    "Do not default to the recurring template: 'last week' or 'one day,' a group gathers to "
    "learn or improve a shared place, the atmosphere and people are praised, a token difficulty "
    "is quickly overcome, and everyone ends happy."
)


def build_content_plan(
    brief: AgentBrief,
    *,
    task_id: int,
    requested_topic: str | None,
    feedback: Sequence[FeedbackTag] = (),
) -> GenerationContentPlan:
    """Choose a coherent blueprint while avoiding recently planned archetypes."""

    recent_archetypes = _recent_plan_values(brief, "archetype", limit=4)
    archetype = _choose_unused(_ARCHETYPES, recent_archetypes, task_id)
    move, move_instruction, concrete_seed = _content_move(
        brief, task_id=task_id, requested_topic=requested_topic, feedback=feedback
    )
    avoid_domains = (
        [] if _repeats_allowed(brief, requested_topic, feedback) else _recent_domains(brief)
    )
    avoid_patterns = [
        _GENERIC_PATTERN_WARNING,
        _STORY_SHAPE_WARNING,
        "Do not repeat the subject, central situation, or setting of any text in "
        "brief.content_history unless the learner explicitly asked for the same topic.",
    ]
    return GenerationContentPlan(
        archetype=archetype.key,
        discourse_form=archetype.discourse_form,
        perspective=archetype.perspective,
        concrete_seed=concrete_seed,
        progression=archetype.progression,
        ending_shape=archetype.ending_shape,
        avoid_patterns=avoid_patterns,
        move=move,
        move_instruction=move_instruction,
        avoid_domains=avoid_domains,
    )


class SubjectDomainError(ValueError):
    """A prose draft whose declared subject domains break the content plan."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        declared: Sequence[str | None],
        avoid: Sequence[str],
        allowed: Sequence[str],
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.declared = list(declared)
        self.avoid = list(avoid)
        self.allowed = list(allowed)


def subject_domain_violation(
    plan: GenerationContentPlan, domains: Sequence[str | None]
) -> SubjectDomainError | None:
    """How the drafts' declared subject domains break the plan, or None when they keep it."""

    allowed = [domain for domain in SUBJECT_DOMAINS if domain not in plan.avoid_domains]

    def violation(kind: str, message: str) -> SubjectDomainError:
        return SubjectDomainError(
            message, kind=kind, declared=domains, avoid=plan.avoid_domains, allowed=allowed
        )

    for index, domain in enumerate(domains, start=1):
        if domain is None:
            return violation("missing", f"prose lesson {index} did not report subject_domain")
        if domain in plan.avoid_domains:
            return violation(
                "repeat",
                f"prose lesson {index} repeats the recent subject domain {domain!r}; "
                f"choose a different subject_domain from: {', '.join(allowed)}",
            )
    if plan.move != "requested" and len(set(domains)) < len(domains):
        return violation(
            "duplicate", "the prose lessons of one request must use different subject domains"
        )
    return None


def content_plan_instructions() -> str:
    return (
        "Content variety is a required quality constraint. Follow content_plan as a concrete "
        "structural blueprint, not as text to quote or mention. Treat recent_lessons as negative "
        "examples for repeated setting, social setup, progression, and ending unless explicit "
        "same-topic feedback or requested_topic says otherwise. Even then, change the discourse "
        "form, immediate goal, progression, and ending. At lower proficiency, simplify vocabulary "
        "and syntax while preserving specific objects, motives, contrasts, and consequences; do "
        "not flatten the content into a generic cooperative success story. Do not add a final "
        "sentence merely to state that everyone felt happy or learned an important lesson. Choose "
        "the situation and structure before selecting SRS terms. Inspect beyond the first target "
        "candidate and use a lower candidate or none when the highest-ranked words would distort "
        "the text. Subject choice follows content_plan.move_instruction, guided by "
        "brief.reading_preferences (the learner's interests and dislikes) and "
        "brief.content_history (recent texts, what each tested, and how the learner reacted: lean "
        "toward what was liked or finished and away from what was disliked, skipped, or "
        "abandoned). Report the chosen subject in content_angle, its broad area in "
        "subject_domain (never one of content_plan.avoid_domains, the domains of the most recent "
        "texts; a draft that repeats one is rejected) and, for variation or new moves, the "
        "tested question in hypothesis. The preference notes name subjects, not plots: use them "
        "to choose what the text is about, and let content_plan alone decide how it is built."
    )


def _content_move(
    brief: AgentBrief,
    *,
    task_id: int,
    requested_topic: str | None,
    feedback: Sequence[FeedbackTag],
) -> tuple[ContentMove, str, str]:
    if requested_topic is not None:
        return (
            "requested",
            "Write about the learner's requested topic. Set hypothesis to null.",
            f"Narrow the requested topic {requested_topic!r} to one specific situation, question, "
            "or disagreement with concrete objects and a clear time and place.",
        )
    if "same_topic" in feedback and brief.recent_lessons:
        recent = brief.recent_lessons[0]
        label = recent.topic or recent.title
        return (
            "favourite",
            "The learner asked for more of the recent topic. Set hypothesis to null.",
            f"Take a genuinely different angle on the recent topic {label!r}; change the actors, "
            "immediate goal, setting details, and outcome.",
        )
    move = choose_move(task_id)
    return move, _MOVE_INSTRUCTIONS[move], _CHOSEN_SEED


def _repeats_allowed(
    brief: AgentBrief, requested_topic: str | None, feedback: Sequence[FeedbackTag]
) -> bool:
    """A requested topic or an explicit "more of this" may stay in a recent domain."""

    return requested_topic is not None or ("same_topic" in feedback and bool(brief.recent_lessons))


def _recent_domains(brief: AgentBrief) -> list[SubjectDomain]:
    domains: list[SubjectDomain] = []
    for item in brief.content_history[:DOMAIN_WINDOW]:
        if item.domain is not None and item.domain not in domains:
            domains.append(item.domain)
    return domains


def choose_move(task_id: int) -> ContentMove:
    """Deterministic per task, so a retried task keeps its move."""

    draw = random.Random(f"content-move:{task_id}").random()
    cumulative = 0.0
    for move, weight in MOVE_WEIGHTS:
        cumulative += weight
        if draw < cumulative:
            return cast(ContentMove, move)
    return "favourite"


def _recent_plan_values(brief: AgentBrief, key: str, *, limit: int) -> set[str]:
    values: list[str] = []
    for lesson in brief.recent_lessons:
        plan = lesson.metadata.get("content_plan")
        if not isinstance(plan, dict):
            continue
        value = plan.get(key)
        if isinstance(value, str):
            values.append(value)
        if len(values) >= limit:
            break
    return set(values)


def _choose_unused(choices: Sequence[_Archetype], recent_keys: set[str], seed: int) -> _Archetype:
    for offset in range(len(choices)):
        candidate = choices[(seed + offset) % len(choices)]
        if candidate.key not in recent_keys:
            return candidate
    return choices[seed % len(choices)]
