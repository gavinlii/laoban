from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from policy_loader import load_policy
from webgame import SessionStore

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", str(BASE_DIR / "policy.pt"))

policy   = load_policy(CHECKPOINT_PATH)
sessions = SessionStore(policy)

app = FastAPI(title="laoban.cards API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewGameRequest(BaseModel):
    bot_first: bool = False
    seed: Optional[int] = None

class ActionRequest(BaseModel):
    session_id: str
    action_index: int

class SessionRequest(BaseModel):
    session_id: str

class ResetRequest(BaseModel):
    session_id: str
    bot_first: Optional[bool] = None
    seed: Optional[int] = None


@app.get("/")
def root():
    return {"ok": True, "message": "laoban.cards backend", "health": "/health"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "checkpoint": policy.path,
        "encoder": "history_belief_v3",
        "state_dim": policy.state_dim,
        "action_dim": policy.action_dim,
        "episode": policy.episode,
    }


@app.post("/api/new-game")
def new_game(req: NewGameRequest):
    ctrl = sessions.create(bot_first=req.bot_first, seed=req.seed)
    return ctrl.state_payload()


@app.post("/api/action")
def action(req: ActionRequest):
    try:
        ctrl = sessions.get(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")
    try:
        ctrl.human_play_by_index(req.action_index)
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ctrl.state_payload()


@app.post("/api/bot-turn")
def bot_turn(req: SessionRequest):
    try:
        ctrl = sessions.get(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")
    ctrl.bot_play_if_needed()
    return ctrl.state_payload()


@app.post("/api/reset")
def reset(req: ResetRequest):
    try:
        ctrl = sessions.get(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")
    if req.bot_first is not None:
        ctrl.bot_first = req.bot_first
    if req.seed is not None:
        ctrl.seed = req.seed
    ctrl.reset(initial=False)
    return ctrl.state_payload()


@app.get("/api/state/{session_id}")
def state(session_id: str):
    try:
        ctrl = sessions.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")
    return ctrl.state_payload()
