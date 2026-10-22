from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Connection, Engine
from sqlalchemy.orm import Session
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from server import settings
from server.access import Access
from server.audio_files import configured_suffix, media_type, require_encoder
from server.audio_worker_api import router as audio_worker_router
from server.base_path import BASE_PATH, app_path
from server.characters import get_characters_state
from server.db import init_all_databases, session_scope
from server.derived_state import ensure_derived_states
from server.generation_queue import (
    TextRequestConflictError,
    ensure_generation_task,
    request_topic_lesson,
)
from server.grammar import get_grammar_state
from server.han import character_tracking_available
from server.language_display import profile_option
from server.learning import EventConflictError, get_reader_state, record_events
from server.lesson_queue import (
    LessonQueueConflictError,
    record_lesson_queue_action,
    record_lesson_queue_move,
)
from server.profile_activation import (
    ProfileActivationUpdate,
    ProfileActivationView,
    activate_profile_settings,
    activation_view,
)
from server.profile_state import ProfileInactiveError, ensure_profile
from server.reading_preferences import (
    PreferencesConflictError,
    add_preference_message,
    preferences_view,
    save_user_preferences,
)
from server.schemas import (
    CharactersState,
    EventRecordResult,
    GrammarState,
    InteractionBatch,
    LessonQueueActionIn,
    LessonQueueActionView,
    LessonQueueMoveIn,
    LessonQueueMoveView,
    PreferenceMessageIn,
    ReaderState,
    ReadingPreferencesUpdate,
    ReadingPreferencesView,
    StatisticsSummary,
    TextDetail,
    TextRequestIn,
    TextRequestView,
    TextsState,
    WordsState,
    WorkspaceList,
    WorkspaceSummary,
)
from server.snapshot import SnapshotReader
from server.statistics import get_statistics_summary
from server.texts import get_text_detail, get_texts_state, text_request_view
from server.tts import (
    LessonAudioStatus,
    TtsSettingsUpdate,
    TtsSettingsView,
    audio_status,
    completed_audio_path,
    ensure_audio_backfill,
    get_tts_settings,
    set_tts_voice,
)
from server.words import get_words_state
from server.workspaces import Workspace, WorkspaceRegistry, registry

SNAPSHOT_DIR = settings.load().snapshot_dir
snapshot_reader = SnapshotReader() if SNAPSHOT_DIR else None
if SNAPSHOT_DIR:
    registry = WorkspaceRegistry(SNAPSHOT_DIR, read_only=True)

ROOT = Path(__file__).resolve().parent.parent
PROFILE_COOKIE = settings.load().profile_cookie


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if snapshot_reader is None:
        require_encoder(configured_suffix())
        init_all_databases()
        yield
    else:
        try:
            for workspace in registry.list():
                with snapshot_reader.session(workspace):
                    pass
            yield
        finally:
            snapshot_reader.close()


app = FastAPI(
    title="Lang Reader",
    description="Local language reader operated by a coding agent",
    version="0.2.0",
    lifespan=lifespan,
)
app.middleware("http")(Access(settings.load().auth))
app.include_router(audio_worker_router)
app.mount(
    "/assets",
    StaticFiles(directory=ROOT / "server" / "static", check_dir=False),
    name="assets",
)
templates = Jinja2Templates(directory=ROOT / "server" / "templates")


@app.middleware("http")
async def snapshot_read_only(request: Request, call_next: RequestResponseEndpoint) -> Response:
    if snapshot_reader is not None and request.method not in {"GET", "HEAD", "OPTIONS"}:
        return JSONResponse(status_code=403, content={"detail": "This snapshot is read-only"})
    return await call_next(request)


@app.exception_handler(LookupError)
def _lookup_error(_request: Request, error: LookupError) -> JSONResponse:
    # Exact type only: a stray KeyError/IndexError is a bug and must stay a 500.
    if type(error) is not LookupError:
        raise error
    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.exception_handler(EventConflictError)
@app.exception_handler(ProfileInactiveError)
@app.exception_handler(TextRequestConflictError)
@app.exception_handler(LessonQueueConflictError)
@app.exception_handler(PreferencesConflictError)
def _conflict_error(_request: Request, error: ValueError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(error)})


def _workspace(profile_id: str) -> Workspace:
    try:
        return registry.resolve(profile_id)
    except (LookupError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def get_db(profile_id: str) -> Iterator[Session]:
    scope = snapshot_reader.session if snapshot_reader is not None else session_scope
    with scope(_workspace(profile_id)) as db:
        yield db


Database = Annotated[Session, Depends(get_db)]


def get_fresh_db(db: Database) -> Session:
    """Refresh derived caches at the HTTP boundary so domain reads stay side-effect free."""

    ensure_derived_states(db)
    return db


FreshDatabase = Annotated[Session, Depends(get_fresh_db)]


def _refresh_derived_states(bind: Engine | Connection) -> None:
    with Session(bind=bind) as db:
        ensure_derived_states(db)


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
def index(request: Request) -> RedirectResponse:
    requested = request.cookies.get(PROFILE_COOKIE) or registry.selected_id()
    try:
        workspace = registry.resolve(requested)
    except (LookupError, ValueError):
        workspace = registry.resolve(registry.selected_id())
    return RedirectResponse(url=app_path(f"/p/{workspace.profile_id}"), status_code=307)


@app.get("/p/{profile_id}", response_class=HTMLResponse, include_in_schema=False)
def profile_page(request: Request, profile_id: str, view: str | None = None) -> HTMLResponse:
    workspace = _workspace(profile_id)
    show_characters = character_tracking_available(workspace.learning_language)
    available_views = {"texts", "words", "grammar", "statistics", "settings"}
    if show_characters:
        available_views.add("characters")
    initial_view = view if view in available_views else "reading"
    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "workspace": workspace,
            "initial_view": initial_view,
            "show_characters": show_characters,
            "base_path": BASE_PATH,
            "profile_options": [profile_option(item) for item in registry.list()],
            "read_only": snapshot_reader is not None,
        },
    )
    response.set_cookie(
        PROFILE_COOKIE,
        workspace.profile_id,
        httponly=True,
        samesite="lax",
        path=BASE_PATH or "/",
    )
    return response


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/api/profiles", response_model=WorkspaceList)
def profiles() -> WorkspaceList:
    return WorkspaceList(
        selected_profile_id=registry.selected_id(),
        profiles=[
            WorkspaceSummary(
                profile_id=workspace.profile_id,
                label=workspace.label,
                learning_language=workspace.learning_language,
                translation_language=workspace.translation_language,
            )
            for workspace in registry.list()
        ],
    )


@app.get(
    "/api/profiles/{profile_id}/activation",
    response_model=ProfileActivationView,
)
def profile_activation(db: Database) -> ProfileActivationView:
    return activation_view(ensure_profile(db))


@app.post(
    "/api/profiles/{profile_id}/activation",
    response_model=ProfileActivationView,
)
def activate_profile(
    update: ProfileActivationUpdate,
    db: Database,
) -> ProfileActivationView:
    view = activate_profile_settings(db, update)
    ensure_generation_task(db)
    ensure_audio_backfill(db)
    return view


def _events(batch: InteractionBatch, db: Session, background: BackgroundTasks) -> EventRecordResult:
    try:
        result = record_events(db, batch.events, rebuild_derived=False)
        # Replay after the response: a click must not wait for the whole history.
        background.add_task(_refresh_derived_states, db.get_bind())
        return result
    except (EventConflictError, LookupError):
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/profiles/{profile_id}/reader", response_model=ReaderState)
def reader(db: FreshDatabase, lesson_id: int | None = None, fresh: bool = False) -> ReaderState:
    return get_reader_state(db, lesson_id=lesson_id, fresh=fresh)


@app.get("/api/profiles/{profile_id}/texts", response_model=TextsState)
def texts(db: FreshDatabase) -> TextsState:
    return get_texts_state(db)


@app.get("/api/profiles/{profile_id}/texts/{lesson_id}", response_model=TextDetail)
def text_detail(lesson_id: int, db: Database) -> TextDetail:
    return get_text_detail(db, lesson_id)


def _lesson_queue_action(
    lesson_id: int,
    action: LessonQueueActionIn,
    db: Session,
    *,
    skipped: bool,
) -> LessonQueueActionView:
    result = record_lesson_queue_action(
        db,
        lesson_id=lesson_id,
        action_id=action.action_id,
        skipped=skipped,
    )
    if skipped:
        ensure_generation_task(db, trigger_queue_action_ids=[str(result.action_id)])
    return result


@app.post(
    "/api/profiles/{profile_id}/texts/{lesson_id}/skip",
    response_model=LessonQueueActionView,
)
def skip_text(lesson_id: int, action: LessonQueueActionIn, db: Database) -> LessonQueueActionView:
    return _lesson_queue_action(lesson_id, action, db, skipped=True)


@app.post(
    "/api/profiles/{profile_id}/texts/{lesson_id}/restore",
    response_model=LessonQueueActionView,
)
def restore_text(
    lesson_id: int, action: LessonQueueActionIn, db: Database
) -> LessonQueueActionView:
    return _lesson_queue_action(lesson_id, action, db, skipped=False)


@app.post(
    "/api/profiles/{profile_id}/texts/{lesson_id}/move",
    response_model=LessonQueueMoveView,
)
def move_text(lesson_id: int, move: LessonQueueMoveIn, db: Database) -> LessonQueueMoveView:
    return record_lesson_queue_move(
        db,
        lesson_id=lesson_id,
        action_id=move.action_id,
        direction=move.direction,
        neighbor_lesson_id=move.neighbor_lesson_id,
    )


@app.post(
    "/api/profiles/{profile_id}/text-requests",
    response_model=TextRequestView,
    status_code=202,
)
def request_text(request: TextRequestIn, db: Database) -> TextRequestView:
    return text_request_view(db, request_topic_lesson(db, request))


@app.get("/api/profiles/{profile_id}/words", response_model=WordsState)
def words(db: FreshDatabase) -> WordsState:
    return get_words_state(db)


@app.get("/api/profiles/{profile_id}/characters", response_model=CharactersState)
def characters(db: FreshDatabase) -> CharactersState:
    return get_characters_state(db)


@app.get("/api/profiles/{profile_id}/statistics", response_model=StatisticsSummary)
def statistics(db: FreshDatabase) -> StatisticsSummary:
    return get_statistics_summary(db)


@app.get("/api/profiles/{profile_id}/grammar", response_model=GrammarState)
def grammar(db: FreshDatabase) -> GrammarState:
    return get_grammar_state(db)


@app.get("/api/profiles/{profile_id}/reading-preferences", response_model=ReadingPreferencesView)
def reading_preferences(db: Database) -> ReadingPreferencesView:
    return preferences_view(db)


@app.put("/api/profiles/{profile_id}/reading-preferences", response_model=ReadingPreferencesView)
def update_reading_preferences(
    update: ReadingPreferencesUpdate, db: Database
) -> ReadingPreferencesView:
    return save_user_preferences(db, update.text, expected_revision_id=update.expected_revision_id)


@app.post(
    "/api/profiles/{profile_id}/reading-preferences/messages",
    response_model=ReadingPreferencesView,
)
def send_preference_message(message: PreferenceMessageIn, db: Database) -> ReadingPreferencesView:
    return add_preference_message(db, str(message.message_id), message.text)


@app.get("/api/profiles/{profile_id}/tts", response_model=TtsSettingsView)
def tts_settings(db: Database) -> TtsSettingsView:
    return get_tts_settings(ensure_profile(db))


@app.put("/api/profiles/{profile_id}/tts", response_model=TtsSettingsView)
def update_tts_settings(update: TtsSettingsUpdate, db: Database) -> TtsSettingsView:
    try:
        return get_tts_settings(set_tts_voice(db, update.voice_id))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get(
    "/api/profiles/{profile_id}/lessons/{lesson_id}/audio/status",
    response_model=LessonAudioStatus,
)
def lesson_audio_status(profile_id: str, lesson_id: int, db: Database) -> LessonAudioStatus:
    status = audio_status(db, _workspace(profile_id), lesson_id)
    if snapshot_reader is not None and status.state == "preparing":
        return status.model_copy(
            update={
                "state": "unavailable",
                "reason": "provider_unavailable",
                "message": "This snapshot has no cached audio; no audio worker runs here.",
            }
        )
    return status


@app.get("/api/profiles/{profile_id}/lessons/{lesson_id}/audio", response_class=FileResponse)
def lesson_audio(profile_id: str, lesson_id: int, db: Database) -> FileResponse:
    path = completed_audio_path(db, _workspace(profile_id), lesson_id)
    return FileResponse(
        path,
        media_type=media_type(path),
        headers={"Cache-Control": "private, no-store"},
    )


@app.post("/api/profiles/{profile_id}/events", response_model=EventRecordResult)
def events(batch: InteractionBatch, db: Database, background: BackgroundTasks) -> EventRecordResult:
    return _events(batch, db, background)
