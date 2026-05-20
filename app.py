import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import base64
import hashlib
import hmac
import io
import html
import re
import secrets
import sqlite3
import urllib.parse
import urllib.request
import uuid
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

from train_model import train_and_save_model


MODEL_PATH = "model/mental_health_model.pkl"
DB_PATH = "history.db"
IMAGE_DIR = "history_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

st.set_page_config(
    page_title="MindCare AI",
    page_icon="MC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    r"""
<style>
:root {
  --ink: #24122e;
  --muted: #74637b;
  --paper: #fbf8ff;
  --panel: rgba(255,255,255,.82);
  --line: rgba(42,14,60,.13);
  --navy: #2a0e3c;
  --deep: #170420;
  --teal: #5e2b6d;
  --mint: #e6e6fa;
  --gold: #d8b96a;
  --rose: #a87ca0;
  --violet: #5e2b6d;
  --lavender: #e6e6fa;
  --plum: #a87ca0;
  --aubergine: #2a0e3c;
}

html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
  background:
    radial-gradient(circle at 8% 4%, rgba(230,230,250,.72), transparent 24%),
    radial-gradient(circle at 92% 8%, rgba(168,124,160,.23), transparent 28%),
    linear-gradient(135deg, #fbf8ff 0%, #f2edf7 45%, #fffaf2 100%);
}

.block-container {
  max-width: 1220px;
  padding: .75rem 1.1rem 2.5rem;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #2a0e3c 0%, #24102f 58%, #170420 100%);
  border-right: 1px solid rgba(255,255,255,.12);
}

[data-testid="stSidebar"] * { color: #f8fafc !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  border-radius: 12px;
  padding: .35rem .55rem;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(255,255,255,.08);
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 999;
  margin: 0 0 12px;
  padding: 12px 16px;
  background:
    linear-gradient(135deg, rgba(42,14,60,.98), rgba(94,43,109,.94)),
    linear-gradient(90deg, rgba(216,185,106,.22), transparent);
  border: 1px solid rgba(255,255,255,.13);
  border-radius: 22px;
  box-shadow: 0 22px 55px rgba(42,14,60,.18);
  backdrop-filter: blur(18px);
}

.brand {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
}

.brand-left {
  display: flex;
  align-items: center;
  gap: 14px;
}

.logo {
  width: 46px;
  height: 46px;
  border-radius: 15px;
  display: grid;
  place-items: center;
  background: linear-gradient(145deg, #fff8e8, #d8b96a);
  color: #fff8e8;
  color: #2a0e3c;
  font-size: 16px;
  font-weight: 900;
  letter-spacing: .06em;
  box-shadow: 0 16px 36px rgba(0,0,0,.26);
}

.brand h2 {
  margin: 0;
  font-size: 23px;
  font-weight: 900;
  color: #fffaf0;
  letter-spacing: 0;
}

.brand p {
  margin: 3px 0 0;
  color: rgba(236,244,241,.72);
  font-size: 14px;
}

.brand-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 1px solid rgba(216,185,106,.36);
  color: #fff5d6;
  background: rgba(216,185,106,.16);
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 800;
  white-space: nowrap;
}

.stButton > button {
  width: 100%;
  min-height: 42px;
  border-radius: 13px !important;
  border: 1px solid rgba(17,29,47,.12) !important;
  background: rgba(255,255,255,.70) !important;
  color: var(--navy) !important;
  font-weight: 800 !important;
  box-shadow: 0 10px 26px rgba(17,29,47,.06) !important;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}

.stButton > button:hover {
  transform: translateY(-1px);
  border-color: rgba(94,43,109,.42) !important;
  box-shadow: 0 16px 34px rgba(94,43,109,.13) !important;
}

[data-testid="column"] .stButton > button {
  padding-left: .75rem !important;
  padding-right: .75rem !important;
}

.nav-note {
  margin: 0 0 18px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(185,139,47,.32), transparent);
}

.hero {
  min-height: 330px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(360px, .78fr);
  align-items: center;
  gap: 24px;
  padding: 34px;
  border-radius: 24px;
  border: 1px solid rgba(255,255,255,.86);
  background:
    linear-gradient(135deg, rgba(255,255,255,.94), rgba(241,248,245,.80)),
    linear-gradient(45deg, rgba(230,230,250,.36), rgba(168,124,160,.18));
  box-shadow: 0 24px 70px rgba(17,29,47,.12);
  overflow: hidden;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  color: #285f5f;
  background: rgba(230,230,250,.72);
  border: 1px solid rgba(94,43,109,.14);
  font-weight: 900;
  font-size: 13px;
  margin-bottom: 18px;
}

.hero h1 {
  max-width: 760px;
  margin: 0;
  color: var(--navy);
  font-size: clamp(34px, 4.4vw, 56px);
  line-height: 1.06;
  letter-spacing: 0;
  font-weight: 950;
}

.hero h1 span { color: var(--teal); }

.hero p {
  max-width: 680px;
  margin: 16px 0 0;
  color: #46576d;
  font-size: 16px;
  line-height: 1.65;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 22px;
}

.hero-chip {
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid rgba(19,36,58,.12);
  background: rgba(255,255,255,.72);
  color: var(--navy);
  font-weight: 800;
  font-size: 14px;
}

.hero-visual {
  border-radius: 22px;
  background:
    linear-gradient(145deg, rgba(255,255,255,.06), transparent),
    #2a0e3c;
  color: #f8fafc;
  padding: 22px;
  border: 1px solid rgba(255,255,255,.18);
  box-shadow: 0 30px 70px rgba(19,36,58,.23);
}

.pulse-line {
  height: 82px;
  margin: 18px 0;
  border-radius: 18px;
  background:
    linear-gradient(90deg, transparent 0 6%, #9bd8c3 6% 8%, transparent 8% 18%, #c59b3d 18% 22%, transparent 22% 38%, #bd5b66 38% 42%, transparent 42% 54%, #9bd8c3 54% 60%, transparent 60% 100%),
    linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.10);
}

.empty-signal-line {
  height: 82px;
  margin: 18px 0;
  border-radius: 18px;
  position: relative;
  overflow: hidden;
  background:
    repeating-linear-gradient(90deg, rgba(255,250,242,.10) 0 1px, transparent 1px 64px),
    linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.02));
  border: 1px dashed rgba(255,255,255,.18);
}

.empty-signal-line::after {
  content: "";
  position: absolute;
  left: 22px;
  right: 22px;
  top: 50%;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(216,185,106,.62), transparent);
}

.risk-strip {
  margin: 18px 0;
  padding: 16px;
  border-radius: 18px;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  background:
    linear-gradient(135deg, rgba(255,250,242,.14), rgba(255,255,255,.05));
  border: 1px solid rgba(255,255,255,.16);
}

.risk-pill {
  min-height: 84px;
  border-radius: 15px;
  padding: 13px 14px;
  display: grid;
  align-content: center;
  gap: 5px;
  background: rgba(255,255,255,.09);
  border: 1px solid rgba(255,255,255,.13);
}

.risk-pill span {
  color: #d9e5ef !important;
  font-size: 12px;
  font-weight: 850;
  text-transform: uppercase;
}

.risk-pill b {
  color: #fff8e8 !important;
  font-size: clamp(16px, 2vw, 22px);
  line-height: 1.18;
  overflow-wrap: anywhere;
}

.visual-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  align-items: center;
  padding: 14px 0;
  border-top: 1px solid rgba(255,255,255,.12);
}

.visual-row b { color: #fff8e8; }
.hero-visual .visual-row,
.hero-visual .visual-row b {
  color: #fff8e8 !important;
}

.hero-visual .visual-row span,
.hero-visual .visual-row div,
.hero-visual .visual-row br + span {
  color: #d9e5ef !important;
  font-size: 13px;
}

.hero-visual .visual-row > span {
  color: #fff5d6 !important;
  font-weight: 900;
}

.score-ring {
  width: 58px;
  height: 58px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  color: #2a0e3c;
  font-weight: 950;
  background: conic-gradient(var(--mint) 0 72%, rgba(255,255,255,.16) 72% 100%);
  box-shadow: inset 0 0 0 7px rgba(19,36,58,.65);
}

.score-ring.empty {
  color: #fff8e8;
  background:
    radial-gradient(circle, rgba(255,250,242,.12) 0 48%, transparent 49%),
    conic-gradient(rgba(216,185,106,.34) 0 100%);
}

.quick-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0 10px;
}

.feature-card, .info-card, .history-card, .metric-tile {
  border: 1px solid rgba(19,36,58,.10);
  background: rgba(255,255,255,.78);
  box-shadow: 0 16px 42px rgba(17,29,47,.075);
}

.feature-link {
  display: block;
  color: inherit !important;
  text-decoration: none !important;
}

.feature-card {
  min-height: 218px;
  border-radius: 20px;
  padding: 24px;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  cursor: pointer;
  position: relative;
  overflow: hidden;
}

.feature-card::after {
  content: "";
  position: absolute;
  inset: auto 20px 18px 20px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(94,43,109,.28), transparent);
}

.feature-link:hover .feature-card {
  transform: translateY(-4px);
  border-color: rgba(94,43,109,.26);
  box-shadow: 0 24px 54px rgba(94,43,109,.14);
}

.feature-card .ico {
  width: 46px;
  height: 46px;
  display: grid;
  place-items: center;
  border-radius: 14px;
  margin-bottom: 18px;
  color: #fff;
  background: var(--navy);
  font-weight: 900;
}

.feature-card h3,
.feature-title {
  margin: 0 0 10px;
  color: var(--navy);
  font-size: 19px;
  line-height: 1.3;
  font-weight: 900;
}

.feature-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.65;
  font-size: 14px;
}

.tone-teal .ico { background: #5e2b6d; }
.tone-gold .ico { background: #d8b96a; color: #2a0e3c; }
.tone-violet .ico { background: #7a4c86; }
.tone-rose .ico { background: #a87ca0; }

.section {
  margin-top: 18px;
  border-radius: 22px;
  padding: 26px;
  background: rgba(255,255,255,.80);
  border: 1px solid rgba(255,255,255,.82);
  box-shadow: 0 18px 52px rgba(17,29,47,.09);
}

.section h1, .section h2, .section h3 {
  color: var(--navy);
  letter-spacing: 0;
}

.section h1 { margin: 0 0 12px; font-size: clamp(30px, 4vw, 48px); line-height: 1.08; }
.section h2 { margin: 0 0 14px; font-size: 26px; }
.section h3 { margin-bottom: 8px; }
.section > p:first-of-type { margin-top: 0; }

.section p, .section li {
  color: #4d5e73;
  line-height: 1.75;
}

.grid2 {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.grid3 {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.lux-title {
  display: grid;
  grid-template-columns: minmax(0,1fr) auto;
  gap: 18px;
  align-items: end;
  margin-bottom: 18px;
}

.lux-title .kicker {
  display: inline-flex;
  width: max-content;
  padding: 7px 11px;
  border-radius: 999px;
  background: rgba(185,139,47,.12);
  border: 1px solid rgba(185,139,47,.22);
  color: #765b1f;
  font-size: 12px;
  font-weight: 950;
  letter-spacing: .04em;
  text-transform: uppercase;
  margin-bottom: 10px;
}

.lux-title h1, .lux-title h2 {
  margin: 0;
}

.lux-title p {
  max-width: 680px;
  margin: 8px 0 0;
}

.metric-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 18px 0 4px;
}

.metric-tile {
  border-radius: 16px;
  padding: 18px;
}

.metric-tile b {
  display: block;
  color: var(--navy);
  font-size: 25px;
  line-height: 1;
  margin-bottom: 6px;
}

.metric-tile span {
  color: var(--muted);
  font-size: 13px;
  font-weight: 800;
}

.info-card {
  border-radius: 16px;
  padding: 22px;
}

.info-card h3 { margin-top: 0; }
.info-card h3,
.info-card p,
.info-card b,
.metric-tile h3,
.metric-tile p {
  color: var(--navy) !important;
}

.info-card p {
  font-weight: 650;
}

.badge {
  display: inline-flex;
  align-items: center;
  padding: 7px 12px;
  margin: 5px 6px 5px 0;
  border-radius: 999px;
  background: rgba(31,138,138,.10);
  border: 1px solid rgba(31,138,138,.18);
  color: #226b6b;
  font-weight: 900;
  font-size: 13px;
}

.step {
  display: flex;
  gap: 16px;
  padding: 18px 0;
  border-bottom: 1px solid rgba(19,36,58,.10);
}

.step:last-child { border-bottom: 0; }

.step-num {
  min-width: 42px;
  height: 42px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  color: #fff;
  font-weight: 950;
  background: linear-gradient(145deg, #13243a, #1f8a8a);
}

.result {
  border-radius: 20px;
  padding: 26px;
  margin-top: 18px;
  border: 1px solid transparent;
  box-shadow: 0 18px 42px rgba(17,32,51,.09);
}

.result-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.emotion-mark {
  width: 58px;
  height: 58px;
  border-radius: 18px;
  display: grid;
  place-items: center;
  background: rgba(255,255,255,.74);
  border: 1px solid rgba(42,14,60,.10);
  color: var(--navy);
  font-size: 30px;
  font-weight: 950;
  font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
  line-height: 1;
  box-shadow: 0 12px 28px rgba(42,14,60,.08);
}

.result .label {
  display: inline-flex;
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 950;
  margin-bottom: 10px;
}

.result h2,
.result-title {
  margin: 4px 0 12px;
  color: var(--navy);
  font-size: clamp(28px, 4vw, 42px);
  line-height: 1.14;
  font-weight: 950;
}

.result p {
  margin: 6px 0;
  color: #2a0e3c !important;
  font-weight: 850;
}
.result b { color: #2a0e3c !important; }
.low { background: rgba(230,230,250,.82); border-color: rgba(94,43,109,.22); }
.low .label { color: #166534; background: rgba(22,101,52,.10); }
.medium { background: rgba(255,244,218,.86); border-color: rgba(197,155,61,.34); }
.medium .label { color: #854d0e; background: rgba(197,155,61,.16); }
.high { background: rgba(255,232,235,.88); border-color: rgba(189,91,102,.34); }
.high .label { color: #9f1239; background: rgba(189,91,102,.13); }

.insight-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-top: 16px;
}

.insight-card {
  border-radius: 18px;
  padding: 18px;
  border: 1px solid rgba(17,29,47,.10);
  background: rgba(255,255,255,.76);
  box-shadow: 0 14px 34px rgba(17,29,47,.07);
}

.insight-card h3 {
  margin: 0 0 10px;
  color: var(--navy);
  font-size: 18px;
}

.insight-card ul {
  margin: 0;
  padding-left: 18px;
}

.insight-card li {
  color: #4f6075;
  line-height: 1.62;
  margin: 6px 0;
}

.care-note {
  margin-top: 14px;
  border-radius: 16px;
  padding: 13px 15px;
  color: #5f4a19;
  background: rgba(255,248,229,.78);
  border: 1px solid rgba(185,139,47,.24);
  font-weight: 750;
}

.mood-graph {
  width: 100%;
  margin: 12px 0 24px;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(216,185,106,.28);
  background:
    radial-gradient(circle at 88% 14%, rgba(216,185,106,.18), transparent 24%),
    linear-gradient(135deg, #190520 0%, #2a0e3c 58%, #371348 100%);
  box-shadow: 0 22px 54px rgba(42,14,60,.18);
}

.mood-graph-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 14px;
  padding: 18px 20px 6px;
}

.mood-graph .mood-graph-head h3,
.mood-graph .mood-graph-head h3 *,
.mood-graph .mood-graph-head h3 a {
  margin: 0;
  color: #fff8e8 !important;
  -webkit-text-fill-color: #fff8e8 !important;
  font-size: clamp(20px, 2.6vw, 30px);
  text-shadow: 0 2px 10px rgba(0,0,0,.34);
}

.mood-graph .mood-graph-head p,
.mood-graph .mood-graph-head p * {
  margin: 5px 0 0;
  color: #f5ecff !important;
  -webkit-text-fill-color: #f5ecff !important;
  font-weight: 700;
  text-shadow: 0 2px 10px rgba(0,0,0,.28);
}

.mood-graph-pill {
  min-width: 86px;
  padding: 9px 12px;
  border-radius: 999px;
  text-align: center;
  color: #2a0e3c;
  background: #fff4c7;
  font-weight: 950;
  box-shadow: 0 12px 28px rgba(0,0,0,.18);
}

.mood-svg-wrap {
  padding: 0 10px 14px;
}

.mood-svg-wrap svg {
  display: block;
  width: 100%;
  height: auto;
  min-height: 260px;
}

.mood-empty-note {
  margin: 0 20px 18px;
  padding: 12px 14px;
  border-radius: 14px;
  color: #fff8e8;
  background: rgba(255,255,255,.10);
  border: 1px solid rgba(255,255,255,.12);
  font-weight: 800;
}

.hist {
  width: 100%;
  overflow-x: auto;
  border-radius: 16px;
  border: 1px solid rgba(19,36,58,.10);
  -webkit-overflow-scrolling: touch;
}

.row {
  min-width: 900px;
  display: grid;
  grid-template-columns: 130px 1.9fr 190px 130px 190px;
  align-items: center;
  gap: 16px;
  padding: 15px 18px;
  border-bottom: 1px solid rgba(19,36,58,.08);
  background: rgba(255,255,255,.55);
  color: #405267;
}

.row span {
  min-width: 0;
  overflow-wrap: anywhere;
}

.row:last-child { border-bottom: 0; }
.head {
  background: rgba(19,36,58,.94);
  color: #f8fafc;
  font-weight: 900;
}

.hist .head span {
  color: #fffaf2 !important;
}

.happy, .sad, .neutral {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 13px;
  font-weight: 900;
}

.happy { color: #166534; background: rgba(22,101,52,.10); }
.sad { color: #9f1239; background: rgba(159,18,57,.10); }
.neutral { color: #854d0e; background: rgba(197,155,61,.14); }

.history-list {
  display: grid;
  gap: 12px;
}

.history-card {
  border-radius: 18px;
  padding: 18px;
  display: grid;
  grid-template-columns: 124px minmax(0,1fr) 170px;
  gap: 18px;
  align-items: center;
}

.history-thumb {
  width: 112px;
  height: 82px;
  border-radius: 15px;
  display: grid;
  place-items: center;
  overflow: hidden;
  color: #fff8e8;
  background: linear-gradient(145deg, #111d2f, #227d78);
  font-weight: 950;
  letter-spacing: .06em;
}

.history-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.history-main h3 {
  margin: 0 0 7px;
  color: var(--navy);
  font-size: 19px;
}

.history-main p {
  margin: 0;
  color: #53647a;
  line-height: 1.55;
}

.history-meta {
  display: grid;
  gap: 8px;
  justify-items: end;
  text-align: right;
}

.meta-pill {
  display: inline-flex;
  width: max-content;
  border-radius: 999px;
  padding: 7px 10px;
  color: #45556b;
  background: rgba(17,29,47,.06);
  font-size: 12px;
  font-weight: 900;
}

.history-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-top: 16px;
}

.journal-entry {
  border-radius: 18px;
  padding: 18px;
  border: 1px solid rgba(17,29,47,.10);
  background: rgba(255,255,255,.78);
  box-shadow: 0 14px 34px rgba(17,29,47,.07);
  margin-bottom: 12px;
}

.journal-entry h3 {
  margin: 0 0 8px;
  color: var(--navy);
}

.journal-entry p {
  margin: 4px 0;
}

.resource-panel {
  border-radius: 20px;
  padding: 22px;
  border: 1px solid rgba(185,139,47,.24);
  background:
    linear-gradient(135deg, rgba(255,248,229,.88), rgba(255,255,255,.78));
  box-shadow: 0 16px 42px rgba(17,29,47,.08);
}

.resource-panel strong {
  color: var(--navy);
}

.privacy-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.palette-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.swatch {
  min-height: 94px;
  border-radius: 18px;
  padding: 14px;
  display: grid;
  align-content: end;
  border: 1px solid rgba(42,14,60,.10);
  box-shadow: 0 14px 32px rgba(42,14,60,.08);
  font-weight: 900;
}

.swatch span {
  display: block;
  font-size: 12px;
  opacity: .82;
  margin-top: 3px;
}

.social-preview {
  border-radius: 20px;
  padding: 18px;
  border: 1px solid rgba(42,14,60,.12);
  background: rgba(255,255,255,.76);
  box-shadow: 0 16px 42px rgba(42,14,60,.08);
  margin: 16px 0;
}

.social-preview h3 {
  margin: 0 0 10px;
  color: var(--navy);
}

textarea, input {
  border-radius: 14px !important;
}

[data-testid="stMain"] [data-testid="stWidgetLabel"],
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] label p {
  color: var(--navy) !important;
  font-weight: 850 !important;
}

[data-testid="stMain"] div[data-baseweb="input"],
[data-testid="stMain"] div[data-baseweb="textarea"],
[data-testid="stMain"] textarea,
[data-testid="stMain"] input {
  background: rgba(255,255,255,.92) !important;
  color: var(--navy) !important;
  border-color: rgba(42,14,60,.18) !important;
}

[data-testid="stMain"] textarea::placeholder,
[data-testid="stMain"] input::placeholder {
  color: rgba(42,14,60,.52) !important;
}

[data-testid="stMain"] textarea:focus,
[data-testid="stMain"] input:focus,
[data-testid="stMain"] div[data-baseweb="input"]:focus-within,
[data-testid="stMain"] div[data-baseweb="textarea"]:focus-within {
  border-color: #5e2b6d !important;
  box-shadow: 0 0 0 3px rgba(94,43,109,.18) !important;
  caret-color: #5e2b6d !important;
}

[data-testid="stMain"] [data-testid="stFileUploader"] {
  color: var(--navy) !important;
  margin-top: 8px;
}

[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] {
  min-height: 124px;
  border-radius: 24px !important;
  border: 1px dashed rgba(94,43,109,.38) !important;
  background:
    radial-gradient(circle at 92% 18%, rgba(216,185,106,.20), transparent 22%),
    linear-gradient(135deg, rgba(255,255,255,.96), rgba(249,245,255,.90)) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.86),
    0 22px 54px rgba(42,14,60,.10) !important;
  transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}

[data-testid="stMain"] [data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(94,43,109,.62) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.92),
    0 28px 66px rgba(94,43,109,.16) !important;
  transform: translateY(-1px);
}

[data-testid="stMain"] [data-testid="stFileUploader"] small,
[data-testid="stMain"] [data-testid="stFileUploader"] p,
[data-testid="stMain"] [data-testid="stFileUploader"] span {
  color: #4f3a5e !important;
}

[data-testid="stMain"] [data-testid="stFileUploaderDropzoneInstructions"] span {
  color: #6f5878 !important;
  font-size: 14px !important;
  font-weight: 700 !important;
}

[data-testid="stMain"] [data-testid="stFileUploader"] button {
  min-height: 50px !important;
  border-radius: 14px !important;
  color: #fffaf2 !important;
  background:
    linear-gradient(135deg, #2a0e3c 0%, #5e2b6d 58%, #8c638f 100%) !important;
  border: 1px solid rgba(255,255,255,.22) !important;
  box-shadow: 0 14px 28px rgba(42,14,60,.24) !important;
}

[data-testid="stMain"] [data-testid="stFileUploader"] button *,
[data-testid="stMain"] [data-testid="stFileUploader"] button p,
[data-testid="stMain"] [data-testid="stFileUploader"] button span {
  color: #fffaf2 !important;
  font-weight: 900 !important;
}

button[kind="secondary"]:has(div p:where(:not(:empty))) {
  border-color: rgba(94,43,109,.22) !important;
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stMain"] .stButton > button:has(div p:is(:where(*))) {
  transition: transform .18s ease, box-shadow .18s ease !important;
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stMain"] .stButton > button:has(div p) {
  color: var(--navy) !important;
}

[data-testid="stSidebar"] .stButton > button:has(p),
[data-testid="stMain"] .stButton > button:has(p) {
  color: var(--navy) !important;
}

[data-testid="stSidebar"] .stButton > button:has(p:only-child),
[data-testid="stMain"] .stButton > button:has(p:only-child) {
  color: var(--navy) !important;
}

[data-testid="stSidebar"] .stButton > button:has(p),
[data-testid="stMain"] .stButton > button:has(p) {
  color: var(--navy) !important;
}

[data-testid="stSidebar"] .stButton > button:has(p:is(:where(*))) {
  background: #2a0e3c !important;
  border-color: #2a0e3c !important;
  color: #fffaf2 !important;
  box-shadow: 0 14px 30px rgba(42,14,60,.30) !important;
}

[data-testid="stSidebar"] .stButton > button p {
  color: #fffaf2 !important;
}

.upload-empty-state {
  margin-top: 20px;
  border-radius: 22px;
  padding: 18px 20px;
  color: #2a0e3c;
  background:
    linear-gradient(135deg, rgba(255,250,242,.96), rgba(246,239,250,.92));
  border: 1px solid rgba(216,185,106,.30);
  box-shadow: 0 18px 46px rgba(42,14,60,.08), inset 0 1px 0 rgba(255,255,255,.84);
  font-size: 16px;
  font-weight: 750;
  display: flex;
  align-items: center;
  gap: 12px;
}

.upload-empty-state::before {
  content: "";
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #d8b96a;
  box-shadow: 0 0 0 6px rgba(216,185,106,.18);
}

.home-session-card {
  min-height: 76px;
  margin: 22px 0 24px;
  padding: 18px 24px;
  border-radius: 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  background:
    linear-gradient(135deg, rgba(255,255,255,.88), rgba(255,250,242,.74));
  border: 1px solid rgba(216,185,106,.28);
  box-shadow: 0 18px 46px rgba(42,14,60,.08), inset 0 1px 0 rgba(255,255,255,.82);
}

.home-session-card b {
  display: block;
  color: var(--navy);
  font-size: 16px;
  line-height: 1.25;
  margin-bottom: 8px;
}

.home-session-card span {
  color: #74637b !important;
  font-size: 14px;
  line-height: 1.35;
  font-weight: 750;
}

.home-logout-link {
  min-height: 64px;
  margin: 22px 0 24px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  text-decoration: none !important;
  color: #fffaf2 !important;
  background:
    linear-gradient(135deg, #2a0e3c 0%, #24102f 58%, #170420 100%);
  border: 1px solid rgba(255,255,255,.14);
  box-shadow: 0 18px 38px rgba(42,14,60,.30);
  font-weight: 950;
  transition: transform .18s ease, box-shadow .18s ease;
}

.home-logout-link:hover {
  transform: translateY(-1px);
  box-shadow: 0 24px 46px rgba(42,14,60,.36);
}

.home-auth-actions {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin: 22px 0 24px;
}

.home-auth-link {
  min-height: 48px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  text-decoration: none !important;
  font-weight: 950;
  transition: transform .18s ease, box-shadow .18s ease;
}

.home-auth-link.primary {
  color: #fffaf2 !important;
  background: linear-gradient(135deg, #2a0e3c 0%, #5e2b6d 100%);
  border: 1px solid rgba(255,255,255,.14);
  box-shadow: 0 18px 38px rgba(42,14,60,.24);
}

.home-auth-link.secondary {
  color: #2a0e3c !important;
  background: rgba(255,255,255,.80);
  border: 1px solid rgba(42,14,60,.14);
  box-shadow: 0 12px 26px rgba(42,14,60,.08);
}

.home-auth-link:hover {
  transform: translateY(-1px);
}

.auth-shell {
  height: 1px;
  overflow: hidden;
}

.auth-lux-grid [data-testid="stHorizontalBlock"],
[data-testid="stHorizontalBlock"]:has(.auth-panel) {
  width: min(960px, 100%);
  min-height: 520px;
  margin: 3vh auto 0;
  gap: 0 !important;
  overflow: hidden;
  border-radius: 30px;
  border: 1px solid rgba(255,255,255,.58);
  background:
    linear-gradient(135deg, rgba(255,255,255,.96), rgba(255,251,246,.90));
  box-shadow: 0 34px 90px rgba(42,14,60,.20), inset 0 1px 0 rgba(255,255,255,.84);
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="column"] {
  padding: 0 !important;
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="column"]:has(.auth-form-wrap) {
  min-height: 520px;
  padding: 58px 48px 42px !important;
  background:
    radial-gradient(circle at 92% 12%, rgba(216,185,106,.13), transparent 24%),
    rgba(255,255,255,.72);
}

.auth-panel {
  min-height: 520px;
  padding: 40px 52px;
  background:
    radial-gradient(circle at 84% 24%, rgba(216,185,106,.28), transparent 25%),
    radial-gradient(circle at 12% 88%, rgba(255,250,242,.12), transparent 30%),
    linear-gradient(145deg, #2a0e3c 0%, #4e205f 50%, #986c94 100%);
  color: #fffaf2;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.auth-panel * {
  color: #fffaf2 !important;
}

.auth-panel .logo {
  margin-bottom: 28px;
}

.auth-panel h1 {
  margin: 0 0 14px;
  font-size: clamp(34px, 4vw, 54px);
  line-height: 1.02;
}

.auth-panel p {
  max-width: 420px;
  line-height: 1.7;
  color: rgba(255,250,242,.82) !important;
}

.auth-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 24px;
}

.auth-badges span {
  border-radius: 999px;
  padding: 9px 12px;
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.20);
  font-size: 13px;
  font-weight: 850;
}

.auth-form-wrap {
  padding: 0 34px;
}

.auth-form-wrap h2 {
  margin: 0 0 8px;
  color: var(--navy);
}

.auth-form-wrap > p {
  margin: 0 0 20px;
  color: var(--muted);
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stTabs"] [role="tablist"] {
  gap: 8px;
  padding: 5px;
  border-radius: 16px;
  background: rgba(42,14,60,.06);
  border: 1px solid rgba(42,14,60,.08);
  margin: 18px 0 18px;
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stTabs"],
[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stForm"] {
  padding-left: 34px;
  padding-right: 34px;
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stTabs"] [role="tab"] {
  min-height: 42px;
  border-radius: 12px;
  color: #5b4965 !important;
  font-weight: 900;
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stTabs"] [aria-selected="true"] {
  background:
    linear-gradient(135deg, #2a0e3c, #6e347a) !important;
  color: #fffaf2 !important;
  box-shadow: 0 12px 26px rgba(42,14,60,.18);
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stTabs"] [aria-selected="true"] p,
[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stTabs"] [aria-selected="true"] span {
  color: #fffaf2 !important;
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stForm"] {
  border: 0;
  padding: 0;
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stTextInput"] {
  margin-bottom: 12px;
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) input {
  min-height: 48px;
  border-radius: 14px !important;
  box-shadow: 0 10px 28px rgba(42,14,60,.06);
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) .stFormSubmitButton button {
  min-height: 50px;
  border-radius: 15px !important;
  margin-top: 4px;
  color: #fffaf2 !important;
  background:
    linear-gradient(135deg, #2a0e3c 0%, #5e2b6d 58%, #a87ca0 100%) !important;
  border: 1px solid rgba(255,255,255,.24) !important;
  box-shadow: 0 16px 36px rgba(42,14,60,.22) !important;
}

[data-testid="stHorizontalBlock"]:has(.auth-panel) .stFormSubmitButton button p {
  color: #fffaf2 !important;
  font-weight: 950 !important;
}

[data-testid="stMain"] div[data-baseweb="select"],
[data-testid="stMain"] div[data-baseweb="select"] * {
  background: rgba(255,255,255,.94) !important;
  color: var(--navy) !important;
}

[data-testid="stMain"] div[role="slider"] {
  color: var(--navy) !important;
}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] p,
[data-testid="stMain"] span {
  color: var(--navy);
}

.topbar,
.topbar * {
  color: #fffaf0 !important;
}

.topbar .logo {
  color: #2a0e3c !important;
}

.topbar .brand-pill {
  color: #fff5d6 !important;
}

[data-testid="stMain"] [data-testid="stDataFrame"] div,
[data-testid="stMain"] [data-testid="stDataFrame"] span {
  color: inherit;
}

[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(19,36,58,.10);
}

@media (max-width: 1100px) {
  .hero { grid-template-columns: 1fr; padding: 34px; }
  .quick-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .grid2, .grid3 { grid-template-columns: 1fr; }
  .brand { align-items: flex-start; }
  .brand-pill { display: none; }
  .history-card { grid-template-columns: 96px minmax(0,1fr); }
  .history-meta { grid-column: 1 / -1; justify-items: start; text-align: left; grid-template-columns: repeat(3, max-content); }
  .history-thumb { width: 90px; height: 72px; }
  .lux-title { grid-template-columns: 1fr; }
  .metric-row { grid-template-columns: 1fr; }
  .insight-grid { grid-template-columns: 1fr; }
  .privacy-list { grid-template-columns: 1fr; }
  .palette-grid { grid-template-columns: 1fr; }
  [data-testid="stHorizontalBlock"]:has(.auth-panel) {
    min-height: auto;
    margin-top: 1vh;
  }
  [data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="column"]:has(.auth-form-wrap) {
    min-height: auto;
    padding: 28px !important;
  }
  .auth-panel {
    min-height: 320px;
    padding: 30px 34px;
    justify-content: flex-start;
  }
  .auth-panel h1 { font-size: 34px; }
  .auth-panel .logo { margin-bottom: 18px; }
  .auth-badges { margin-top: 16px; }
}

@media (max-width: 680px) {
  .block-container { padding-left: .8rem; padding-right: .8rem; }
  .hero { padding: 26px; min-height: auto; }
  .hero h1 { font-size: 40px; }
  .quick-grid { grid-template-columns: 1fr; }
  .section { padding: 22px; border-radius: 18px; }
  [data-testid="stHorizontalBlock"]:has(.auth-panel) {
    border-radius: 24px;
  }
  .auth-panel {
    min-height: auto;
    padding: 24px 28px;
  }
  .auth-panel h1 {
    font-size: 30px;
    line-height: 1.08;
  }
  .auth-panel p {
    font-size: 14px;
    line-height: 1.48;
  }
  .auth-badges {
    display: none;
  }
  .auth-form-wrap {
    padding: 0 24px;
  }
  [data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="column"]:has(.auth-form-wrap) {
    padding: 24px 0 28px !important;
  }
  [data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stTabs"],
  [data-testid="stHorizontalBlock"]:has(.auth-panel) [data-testid="stForm"] {
    padding-left: 24px;
    padding-right: 24px;
  }
  .history-card { grid-template-columns: 1fr; }
  .history-meta { grid-template-columns: 1fr; }
  .history-thumb { width: 100%; height: 150px; }
  .mood-graph-head {
    display: grid;
  }
  .mood-graph-pill {
    width: max-content;
  }
  .mood-svg-wrap svg {
    min-height: 220px;
  }
  .risk-strip {
    grid-template-columns: 1fr;
  }
  .hist {
    border: 0;
    overflow: visible;
  }
  .row,
  .row.head {
    min-width: 0;
    grid-template-columns: 1fr;
    gap: 8px;
    border-radius: 16px;
    margin-bottom: 12px;
    border: 1px solid rgba(19,36,58,.10);
  }
  .row.head {
    display: none;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            mobile TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            remember_token TEXT,
            created_at TEXT
        )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS detection_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_mobile TEXT,
            detection_type TEXT,
            input_data TEXT,
            image_path TEXT,
            result TEXT,
            score TEXT,
            emotion TEXT,
            created_at TEXT
        )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS mood_journal(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_mobile TEXT,
            mood_score INTEGER,
            mood_label TEXT,
            mood_kind TEXT,
            note TEXT,
            gratitude TEXT,
            created_at TEXT
        )"""
    )
    ensure_column(con, "users", "name", "TEXT")
    ensure_column(con, "users", "remember_token", "TEXT")
    ensure_column(con, "detection_history", "user_mobile", "TEXT")
    ensure_column(con, "mood_journal", "user_mobile", "TEXT")
    ensure_column(con, "mood_journal", "mood_kind", "TEXT")
    con.commit()
    con.close()


def ensure_column(con, table, column, column_type):
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def current_mobile():
    return st.session_state.get("user_mobile", "")


def current_user_name():
    return st.session_state.get("user_name", "")


def current_auth_token():
    return st.session_state.get("auth_token", "") or st.query_params.get("auth", "")


def normalize_mobile(value):
    return re.sub(r"\D", "", str(value or ""))


def is_valid_mobile(value):
    return bool(re.fullmatch(r"\d{10}", str(value or "").strip()))


def hash_password(password, salt):
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return digest.hex()


def create_user(name, mobile, password):
    name = " ".join(str(name or "").split())
    mobile = str(mobile or "").strip()
    if len(name) < 2:
        return False, "Enter the person's name."
    if not is_valid_mobile(mobile):
        return False, "Mobile number must be exactly 10 digits. Letters and symbols are not allowed."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not re.search(r"\d", password):
        return False, "Password must include at least one number."

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    con = None
    try:
        con = sqlite3.connect(DB_PATH)
        user_count = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        con.execute(
            "INSERT INTO users(name, mobile, password_hash, salt, created_at) VALUES(?,?,?,?,?)",
            (name, mobile, password_hash, salt, datetime.now().strftime("%d %b %Y, %I:%M %p")),
        )
        if user_count == 0:
            con.execute("UPDATE detection_history SET user_mobile = ? WHERE user_mobile IS NULL OR user_mobile = ''", (mobile,))
            con.execute("UPDATE mood_journal SET user_mobile = ? WHERE user_mobile IS NULL OR user_mobile = ''", (mobile,))
        con.commit()
        return True, "Account created. You are signed in."
    except sqlite3.IntegrityError:
        return False, "This mobile number already has an account."
    finally:
        if con:
            con.close()


def verify_user(mobile, password):
    mobile = str(mobile or "").strip()
    if not is_valid_mobile(mobile):
        return None
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT name, password_hash, salt FROM users WHERE mobile = ?", (mobile,)).fetchone()
    con.close()
    if not row:
        return None
    name, expected_hash, salt = row
    if hmac.compare_digest(hash_password(password, salt), expected_hash):
        return name or mobile
    return None


def save_remember_token(mobile):
    token = secrets.token_urlsafe(32)
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE users SET remember_token = ? WHERE mobile = ?", (token, mobile))
    con.commit()
    con.close()
    return token


def user_from_remember_token(token):
    token = str(token or "").strip()
    if not token:
        return None
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT mobile, name FROM users WHERE remember_token = ?", (token,)).fetchone()
    con.close()
    if not row:
        return None
    return {"mobile": row[0], "name": row[1] or row[0]}


def clear_remember_token(mobile):
    if not mobile:
        return
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE users SET remember_token = NULL WHERE mobile = ?", (mobile,))
    con.commit()
    con.close()


def clear_auth_query():
    if "auth" in st.query_params:
        del st.query_params["auth"]


def sign_in_user(mobile, name="", remember_token="", set_home=True):
    st.session_state.authenticated = True
    st.session_state.user_mobile = str(mobile or "").strip()
    st.session_state.user_name = name or st.session_state.user_mobile
    token = remember_token or save_remember_token(st.session_state.user_mobile)
    st.session_state.auth_token = token
    if set_home:
        target_page = get_after_login_page("Home")
        st.session_state.page = target_page
        st.session_state.after_login_page = ""
        st.query_params.from_dict({"page": target_page, "auth": token})
    elif st.query_params.get("auth", "") != token:
        st.query_params["auth"] = token


def logout_user():
    clear_remember_token(current_mobile())
    st.session_state.authenticated = False
    st.session_state.user_mobile = ""
    st.session_state.user_name = ""
    st.session_state.auth_token = ""
    st.session_state.page = "Home"
    clear_auth_query()
    st.query_params["page"] = "Home"
    st.rerun()


def render_auth_page():
    st.markdown('<div class="auth-shell"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, .95], gap="small")

    with left:
        st.markdown(
            """
            <div class="auth-panel">
              <div class="logo">MC</div>
              <h1>Welcome to MindCare AI</h1>
              <p>Sign in with your mobile number to keep your analysis history, uploaded image records, and mood journal private on this device.</p>
              <div class="auth-badges">
                <span>Mobile login</span>
                <span>Password protected</span>
                <span>Local storage</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="auth-form-wrap">
              <h2>Secure Access</h2>
              <p>Use your mobile number and password to continue.</p>
            """,
            unsafe_allow_html=True,
        )

        sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])

        with sign_in_tab:
            with st.form("signin_form"):
                mobile = st.text_input("Mobile number", placeholder="10 digit mobile number", max_chars=10, key="signin_mobile")
                password = st.text_input("Password", type="password", placeholder="Enter password", key="signin_password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)
                if submitted:
                    user_name = verify_user(mobile, password)
                    if user_name:
                        sign_in_user(mobile, user_name)
                        st.success("Signed in successfully.")
                        st.rerun()
                    else:
                        st.error("Mobile number must be 10 digits, and the password must match.")

        with sign_up_tab:
            with st.form("signup_form"):
                name = st.text_input("Name", placeholder="Person's name", key="signup_name")
                mobile = st.text_input("Mobile number", placeholder="10 digit mobile number", max_chars=10, key="signup_mobile")
                password = st.text_input("Create password", type="password", placeholder="Minimum 6 characters with 1 number", key="signup_password")
                confirm = st.text_input("Confirm password", type="password", placeholder="Re-enter password", key="signup_confirm")
                submitted = st.form_submit_button("Create account", use_container_width=True)
                if submitted:
                    if password != confirm:
                        st.error("Passwords do not match.")
                    else:
                        ok, message = create_user(name, mobile, password)
                        if ok:
                            sign_in_user(mobile, name)
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

        st.markdown("</div>", unsafe_allow_html=True)


def save_history(detection_type, input_data, image_path, result, score, emotion):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO detection_history(
            user_mobile, detection_type, input_data, image_path, result, score, emotion, created_at
        ) VALUES(?,?,?,?,?,?,?,?)""",
        (
            current_mobile(),
            detection_type,
            input_data,
            image_path,
            result,
            score,
            emotion,
            datetime.now().strftime("%d %b %Y, %I:%M %p"),
        ),
    )
    con.commit()
    con.close()


def read_history(limit=None):
    con = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM detection_history WHERE user_mobile = ? ORDER BY id DESC"
    params = [current_mobile()]
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, con, params=params)
    con.close()
    return df


def clear_history():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM detection_history WHERE user_mobile = ?", (current_mobile(),))
    con.commit()
    con.close()


def save_mood_entry(mood_score, mood_label, mood_kind, note, gratitude):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO mood_journal(user_mobile, mood_score, mood_label, mood_kind, note, gratitude, created_at)
        VALUES(?,?,?,?,?,?,?)""",
        (
            current_mobile(),
            int(mood_score),
            mood_label,
            mood_kind,
            note,
            gratitude,
            datetime.now().strftime("%d %b %Y, %I:%M %p"),
        ),
    )
    con.commit()
    con.close()


def read_mood_entries(limit=None):
    con = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM mood_journal WHERE user_mobile = ? ORDER BY id DESC"
    params = [current_mobile()]
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, con, params=params)
    con.close()
    return df


def clear_mood_entries():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM mood_journal WHERE user_mobile = ?", (current_mobile(),))
    con.commit()
    con.close()


def save_image(image_bytes, ext=".jpg"):
    path = os.path.join(IMAGE_DIR, f"{uuid.uuid4().hex}{ext}")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((650, 650))
    img.save(path, quality=90)
    return path


def fetch_url_bytes(url, timeout=8):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 MindCareAI/1.0",
            "Accept": "text/html,image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        data = response.read(5_000_000)
    return data, content_type


def find_preview_image_url(page_html, base_url):
    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, page_html, re.IGNORECASE)
        if match:
            return urllib.parse.urljoin(base_url, html.unescape(match.group(1)))
    return ""


def fetch_preview_image_from_link(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "", "Please enter a valid http or https link."
    try:
        data, content_type = fetch_url_bytes(url)
        if content_type.startswith("image/"):
            ext = ".png" if "png" in content_type else ".jpg"
            return save_image(data, ext), ""

        page_html = data.decode("utf-8", errors="ignore")
        image_url = find_preview_image_url(page_html, url)
        if not image_url:
            return "", "No preview image was found on this link. Paste caption text for text analysis."

        image_data, image_type = fetch_url_bytes(image_url)
        if not image_type.startswith("image/"):
            return "", "A preview link was found, but it was not an image."
        ext = ".png" if "png" in image_type else ".jpg"
        return save_image(image_data, ext), ""
    except Exception as exc:
        return "", f"Could not fetch a preview image from this link. Some sites block automated previews. Details: {exc}"


def extract_text_from_image_optional(path):
    try:
        import easyocr

        reader = easyocr.Reader(["en", "hi"], gpu=False, verbose=False)
        rows = reader.readtext(path, detail=0, paragraph=True)
        text = " ".join(str(row) for row in rows)
        return " ".join(text.split())
    except Exception:
        pass

    try:
        import pytesseract

        image = Image.open(path)
        text = pytesseract.image_to_string(image, lang="eng+hin")
        return " ".join(text.split())
    except Exception:
        return ""


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_text_emotion(text, score):
    lowered = text.lower()
    devanagari_count = len(re.findall(r"[\u0900-\u097F]", str(text)))
    non_ascii_count = sum(1 for char in str(text) if ord(char) > 127)
    sad_hinglish = [
        "rona", "rone", "roya", "rula", "dukh", "udas", "tanha", "akela", "akele", "yaad",
        "bhul", "bhool", "hissa", "reh gaye", "dil toot", "dil tut", "heart break", "heartbreak",
    ]
    sad_devanagari_hints = ["रो", "दुख", "उदास", "तन्ह", "अकेल", "याद", "भूल", "हिस", "रह", "आंस", "आँस"]
    if "💔" in str(text) or any(signal in lowered for signal in sad_hinglish) or any(signal in str(text) for signal in sad_devanagari_hints):
        return "Sad"
    if any(word in lowered for word in ["anxiety", "anxious", "panic", "fear", "scared", "nervous", "gabhraman", "dar", "darr"]):
        return "Anxiety / Fear"
    if any(word in lowered for word in ["stress", "pressure", "tension", "tired", "exhausted", "overthinking", "worried", "pareshan"]):
        return "Stress"
    if any(word in lowered for word in ["sad", "lonely", "hopeless", "cry", "empty", "worthless", "broken", "depressed", "heartbroken", "akele", "akela", "dhokha", "toot", "tut", "hissa", "reh gaye"]):
        return "Sad"
    if (devanagari_count >= 12 or non_ascii_count >= 12) and score >= 35:
        return "Sad"
    if any(word in lowered for word in ["happy", "excited", "joy", "great", "good", "proud", "confident", "positive", "saro"]):
        return "Happy / Positive"
    return "Negative" if score >= 60 else "Neutral"


def adjusted_text_score(text, base_score):
    emotion = get_text_emotion(text, base_score)
    if emotion in ["Sad", "Anxiety / Fear"]:
        return max(base_score, 70)
    if emotion == "Stress":
        return max(base_score, 55)
    if emotion == "Happy / Positive":
        return min(base_score, 25)
    return base_score


def risk_result(score):
    if score >= 60:
        return "Possible Mental Health Risk", "high", "Priority"
    if score >= 35:
        return "Medium Risk / Stress Pattern", "medium", "Watch"
    return "Low Risk / Normal", "low", "Steady"


def result_explanation(score, emotion, input_text=""):
    lowered = str(input_text or "").lower()
    keyword_groups = {
        "stress or pressure words": ["stress", "pressure", "tension", "tired", "exhausted", "overthinking", "worried"],
        "sadness words": ["sad", "lonely", "hopeless", "cry", "empty", "worthless", "broken", "depressed"],
        "anxiety or fear words": ["anxiety", "anxious", "panic", "fear", "scared", "nervous", "gabhraman"],
        "positive mood words": ["happy", "excited", "joy", "great", "good", "proud", "confident", "positive", "saro"],
    }
    matched = []
    for label, words in keyword_groups.items():
        found = [word for word in words if word in lowered]
        if found:
            matched.append(f"{label}: {', '.join(found[:4])}")

    if matched:
        signals = [
            f"Detected {matched[0]}.",
            f"The model confidence is {score:.2f}%, so the app classified this as a {risk_result(score)[0].lower()} signal.",
            "This is an awareness-based interpretation, not a clinical conclusion.",
        ]
        if len(matched) > 1:
            signals.insert(1, f"Also noticed {matched[1]}.")
        return signals

    emotion_text = str(emotion or "Neutral")
    if score >= 60:
        first = f"The visible or predicted emotion is {emotion_text}, which maps to a stronger risk signal."
    elif score >= 35:
        first = f"The visible or predicted emotion is {emotion_text}, giving a moderate watch signal."
    else:
        first = f"The visible or predicted emotion is {emotion_text}, which currently maps to a lower risk signal."
    return [
        first,
        f"The confidence score is {score:.2f}%, based on the app's trained model and emotion rules.",
        "Use this as a reflection aid, not as a diagnosis.",
    ]


def wellness_suggestions(score):
    if score >= 60:
        return [
            "Pause and do one grounding activity: slow breathing, water, or stepping away from the screen.",
            "Talk to a trusted friend, family member, teacher, or mentor if the feeling continues.",
            "If someone may be in immediate danger, contact local emergency support or a qualified professional.",
        ]
    if score >= 35:
        return [
            "Take a short break and reduce stimulation for a few minutes.",
            "Write down what triggered the feeling and one small action you can take next.",
            "Check sleep, food, hydration, and workload because these can increase stress signals.",
        ]
    return [
        "Keep the routine that is helping: rest, hydration, movement, and social connection.",
        "Use the history page to notice changes over time instead of judging one result alone.",
        "If your mood changes suddenly, run another check and reflect on what changed.",
    ]


def score_number(value):
    try:
        return float(str(value).replace("%", "").strip())
    except ValueError:
        return 0.0


def dashboard_stats(df):
    if df.empty:
        return {
            "total": 0,
            "average": 0,
            "priority": 0,
            "common_emotion": "None",
            "latest": "No saved checks",
        }
    scores = df["score"].map(score_number)
    priority = int((scores >= 60).sum())
    common_emotion = str(df["emotion"].mode().iloc[0]) if not df["emotion"].mode().empty else "None"
    return {
        "total": len(df),
        "average": round(float(scores.mean()), 2),
        "priority": priority,
        "common_emotion": common_emotion,
        "latest": str(df.iloc[0]["created_at"]),
    }


def trend_label(df):
    if len(df) < 2:
        return "Need more saved checks"
    recent = df.head(3)["score"].map(score_number).mean()
    older = df.iloc[3:6]["score"].map(score_number).mean() if len(df) > 3 else df.tail(3)["score"].map(score_number).mean()
    if recent > older + 5:
        return "Recent risk signals are increasing"
    if recent < older - 5:
        return "Recent risk signals are reducing"
    return "Recent signals look stable"


def parse_created_at(series):
    return pd.to_datetime(series, format="%d %b %Y, %I:%M %p", errors="coerce")


def prepare_detection_chart(df):
    if df.empty:
        return pd.DataFrame()
    chart_df = df.copy()
    chart_df["date"] = parse_created_at(chart_df["created_at"])
    chart_df["Risk confidence"] = chart_df["score"].map(score_number)
    chart_df = chart_df.dropna(subset=["date"]).sort_values("date")
    if chart_df.empty:
        return pd.DataFrame()
    return chart_df.set_index("date")[["Risk confidence"]]


def prepare_mood_chart(df):
    if df.empty:
        return pd.DataFrame()
    chart_df = df.copy()
    chart_df["date"] = parse_created_at(chart_df["created_at"])
    chart_df["Mood score"] = pd.to_numeric(chart_df["mood_score"], errors="coerce")
    chart_df["Stress load"] = 11 - chart_df["Mood score"]
    chart_df = chart_df.dropna(subset=["date", "Mood score"]).sort_values("date")
    if chart_df.empty:
        return pd.DataFrame()
    return chart_df.set_index("date")[["Mood score", "Stress load"]]


def mood_stats(df):
    if df.empty:
        return {"total": 0, "average": 0, "latest": "No journal entries", "common": "None"}
    scores = pd.to_numeric(df["mood_score"], errors="coerce")
    common = str(df["mood_label"].mode().iloc[0]) if not df["mood_label"].mode().empty else "None"
    return {
        "total": len(df),
        "average": round(float(scores.mean()), 2) if not scores.dropna().empty else 0,
        "latest": str(df.iloc[0]["created_at"]),
        "common": common,
    }


def render_mood_health_graph(df, title="Daily Mood and Mental Health Graph"):
    chart_df = df.copy()
    if chart_df.empty:
        st.info("No mood entries yet. Save daily mood entries to build this graph.")
        return

    chart_df["date"] = parse_created_at(chart_df["created_at"])
    chart_df["mood_score_num"] = pd.to_numeric(chart_df["mood_score"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date", "mood_score_num"]).sort_values("date").tail(10)
    if chart_df.empty:
        st.info("Mood entries need valid dates and mood scores to draw the graph.")
        return

    scores = [max(1, min(10, float(value))) for value in chart_df["mood_score_num"].tolist()]
    labels = chart_df["date"].dt.strftime("%d %b").tolist()
    kinds = [str(value or "Mood") for value in chart_df.get("mood_kind", pd.Series(["Mood"] * len(chart_df))).tolist()]

    width, height = 900, 320
    left, right, top, bottom = 58, 28, 36, 64
    graph_w = width - left - right
    graph_h = height - top - bottom
    point_count = len(scores)

    def xy(index, score):
        x = left + (graph_w / max(point_count - 1, 1)) * index if point_count > 1 else left + graph_w / 2
        y = top + ((10 - score) / 9) * graph_h
        return round(x, 2), round(y, 2)

    points = [xy(index, score) for index, score in enumerate(scores)]
    if point_count == 1:
        x, y = points[0]
        mood_line = f"{left},{y} {right + graph_w},{y}"
        note = "Add more days to turn this into a full trend line."
    else:
        mood_line = " ".join(f"{x},{y}" for x, y in points)
        note = "Gold line shows mood score. Higher is better."

    stress_points = [xy(index, 11 - score) for index, score in enumerate(scores)]
    stress_line = " ".join(f"{x},{y}" for x, y in stress_points)
    avg_score = round(sum(scores) / len(scores), 1)
    latest_label = html.escape(str(chart_df.iloc[-1].get("mood_label", "Mood")))

    grid = []
    for score in range(1, 11):
        y = top + ((10 - score) / 9) * graph_h
        opacity = ".28" if score in [1, 5, 10] else ".12"
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="rgba(255,255,255,{opacity})" stroke-width="1"/>')
        if score in [1, 5, 10]:
            grid.append(f'<text x="18" y="{y + 5:.2f}" fill="#fff8e8" font-size="14" font-weight="800">{score}</text>')

    dots = []
    date_marks = []
    for index, ((x, y), score, label, kind) in enumerate(zip(points, scores, labels, kinds)):
        dots.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#fff4c7" stroke="#d8b96a" stroke-width="4"/>')
        dots.append(f'<text x="{x}" y="{y - 16}" text-anchor="middle" fill="#fff8e8" font-size="14" font-weight="900">{score:g}</text>')
        if point_count <= 6 or index in [0, point_count - 1]:
            date_marks.append(f'<text x="{x}" y="{height - 26}" text-anchor="middle" fill="#d9e5ef" font-size="13" font-weight="800">{html.escape(label)}</text>')
            date_marks.append(f'<text x="{x}" y="{height - 10}" text-anchor="middle" fill="#fff4c7" font-size="12" font-weight="800">{html.escape(kind[:14])}</text>')

    st.markdown(
        f"""
        <div class="mood-graph">
          <div class="mood-graph-head">
            <div>
              <h3>{html.escape(title)}</h3>
              <p>{html.escape(note)}</p>
            </div>
            <div class="mood-graph-pill">Avg {avg_score}/10<br>{latest_label}</div>
          </div>
          <div class="mood-svg-wrap">
            <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
              <rect x="0" y="0" width="{width}" height="{height}" fill="transparent"/>
              {''.join(grid)}
              <polyline points="{stress_line}" fill="none" stroke="rgba(155,216,195,.50)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="8 10"/>
              <polyline points="{mood_line}" fill="none" stroke="#d8b96a" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
              {''.join(dots)}
              {''.join(date_marks)}
              <text x="{left}" y="24" fill="#fff8e8" font-size="14" font-weight="900">Mood score</text>
              <text x="{left + 126}" y="24" fill="#9bd8c3" font-size="14" font-weight="900">Stress load</text>
            </svg>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def analyze_image_cached(image_bytes, ext):
    try:
        import tempfile
        from deepface import DeepFace

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(image_bytes)
            path = tmp.name
        result = DeepFace.analyze(img_path=path, actions=["emotion"], enforce_detection=False)
        data = result[0] if isinstance(result, list) else result
        emotion = data.get("dominant_emotion", "neutral")
    except Exception:
        emotion = "neutral"

    emotion_names = {
        "happy": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "fear": "Fear",
        "surprise": "Surprise",
        "neutral": "Neutral",
        "disgust": "Disgust",
    }
    scores = {
        "happy": 10,
        "neutral": 25,
        "surprise": 30,
        "sad": 70,
        "fear": 75,
        "angry": 65,
        "disgust": 60,
    }
    return emotion_names.get(emotion, emotion.title()), scores.get(emotion, 25)


def list_items(items):
    return "".join(f"<li>{html.escape(str(item))}</li>" for item in items)


def emotion_symbol(emotion, score):
    value = str(emotion or "").lower()
    if "happy" in value or "positive" in value:
        return "&#128522;"
    if "sad" in value:
        return "&#128542;"
    if "angry" in value:
        return "&#128544;"
    if "fear" in value or "anxiety" in value:
        return "&#128552;"
    if "stress" in value or "worried" in value:
        return "&#128531;"
    if "surprise" in value:
        return "&#128558;"
    if "disgust" in value:
        return "&#129314;"
    if score >= 60:
        return "&#9888;&#65039;"
    return "&#128522;"


def show_result(title, score, emotion, input_text=""):
    result, css_class, label = risk_result(score)
    explanations = result_explanation(score, emotion, input_text)
    suggestions = wellness_suggestions(score)
    symbol = emotion_symbol(emotion, score)
    st.markdown(
        f"""
        <div class="result {css_class}">
          <div class="result-top">
            <div>
              <span class="label">{label} signal</span>
              <div class="result-title">{title}</div>
            </div>
            <div class="emotion-mark">{symbol}</div>
          </div>
          <p><b>Result:</b> {result}</p>
          <p><b>Confidence:</b> {score:.2f}%</p>
          <p><b>Emotion:</b> {emotion}</p>
        </div>
        <div class="insight-grid">
          <div class="insight-card">
            <h3>Why this result?</h3>
            <ul>{list_items(explanations)}</ul>
          </div>
          <div class="insight-card">
            <h3>Wellness suggestions</h3>
            <ul>{list_items(suggestions)}</ul>
          </div>
        </div>
        <div class="care-note">MindCare AI is for awareness and reflection only. It does not replace help from a qualified mental health professional.</div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(int(score), 100))
    return result


def go(page):
    st.session_state.page = page
    params = {"page": page}
    token = current_auth_token()
    if token:
        params["auth"] = token
    st.query_params.from_dict(params)


ACCOUNT_PAGE = "Sign In / Create Account"

PROTECTED_PAGES = {
    "Text Analysis",
    "Social Media Analyzer",
    "Camera Detection",
    "Image Upload",
    "Results Dashboard",
    "Mood Journal",
    "History",
}


def page_url(page):
    params = {"page": page}
    token = current_auth_token()
    if token:
        params["auth"] = token
    return "?" + urllib.parse.urlencode(params)


def account_page_url(next_page=""):
    params = {"page": ACCOUNT_PAGE}
    token = current_auth_token()
    if token:
        params["auth"] = token
    if next_page:
        params["next"] = next_page
    return "?" + urllib.parse.urlencode(params)


def logout_url():
    params = {"logout": "1"}
    token = current_auth_token()
    if token:
        params["auth"] = token
    return "?" + urllib.parse.urlencode(params)


def protected_page_url(page):
    if st.session_state.get("authenticated", False):
        return page_url(page)
    return account_page_url(page)


def remember_after_login(page):
    if page in PROTECTED_PAGES:
        st.session_state.after_login_page = page


def get_after_login_page(default="Home"):
    page = st.session_state.get("after_login_page", "")
    if page in PROTECTED_PAGES:
        return page
    return default


def emotion_class(value):
    value = str(value)
    if "Happy" in value:
        return "happy"
    if any(word in value for word in ["Sad", "Angry", "Fear", "Stress", "Negative", "Anxiety"]):
        return "sad"
    return "neutral"


def render_recent_history(limit=4):
    df = read_history(limit)
    if df.empty:
        st.info("No history yet. Try text, camera, or image detection.")
        return

    st.markdown(
        '<div class="hist"><div class="row head"><span>Type</span><span>Input</span><span>Signal</span><span>Confidence</span><span>Date & Time</span></div>',
        unsafe_allow_html=True,
    )
    for _, row in df.iterrows():
        input_text = str(row["input_data"])[:92]
        css_class = emotion_class(row["emotion"])
        st.markdown(
            f"""
            <div class="row">
              <span>{row["detection_type"]}</span>
              <span>{input_text}</span>
              <span class="{css_class}">{row["emotion"]}</span>
              <span>{row["score"]}</span>
              <span>{row["created_at"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def trim_text(value, length=150):
    value = " ".join(str(value or "").split())
    if len(value) <= length:
        return value
    return value[: length - 1].rstrip() + "..."


def image_data_uri(path):
    try:
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "jpeg"
        mime = "jpeg" if ext == "jpg" else ext
        return f"data:image/{mime};base64,{encoded}"
    except OSError:
        return ""


def render_history_cards(df):
    st.markdown('<div class="history-list">', unsafe_allow_html=True)
    for _, row in df.iterrows():
        img = str(row.get("image_path", "") or "")
        if img and os.path.exists(img):
            data_uri = image_data_uri(img)
            thumb = f'<img src="{data_uri}" alt="Detection image">' if data_uri else html.escape(str(row["detection_type"])[:2].upper())
        else:
            thumb = html.escape(str(row["detection_type"])[:2].upper())

        signal_class = emotion_class(row["emotion"])
        st.markdown(
            f"""
            <div class="history-card">
              <div class="history-thumb">{thumb}</div>
              <div class="history-main">
                <h3>{html.escape(str(row["result"]))}</h3>
                <p>{html.escape(trim_text(row["input_data"]))}</p>
              </div>
              <div class="history-meta">
                <span class="{signal_class}">{html.escape(str(row["emotion"]))}</span>
                <span class="meta-pill">{html.escape(str(row["score"]))}</span>
                <span class="meta-pill">{html.escape(str(row["created_at"]))}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


init_db()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_mobile" not in st.session_state:
    st.session_state.user_mobile = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "auth_token" not in st.session_state:
    st.session_state.auth_token = st.query_params.get("auth", "")
if "after_login_page" not in st.session_state:
    st.session_state.after_login_page = ""

if st.query_params.get("logout", "") == "1":
    token_user = user_from_remember_token(st.query_params.get("auth", ""))
    clear_remember_token(current_mobile() or (token_user["mobile"] if token_user else ""))
    st.session_state.authenticated = False
    st.session_state.user_mobile = ""
    st.session_state.user_name = ""
    st.session_state.auth_token = ""
    st.session_state.page = "Home"
    clear_auth_query()
    if "logout" in st.query_params:
        del st.query_params["logout"]
    st.query_params["page"] = "Home"
    st.rerun()

if not st.session_state.authenticated:
    remembered_user = user_from_remember_token(st.query_params.get("auth", ""))
    if remembered_user:
        sign_in_user(
            remembered_user["mobile"],
            remembered_user["name"],
            remember_token=st.query_params.get("auth", ""),
            set_home=False,
        )
    else:
        clear_auth_query()

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Detection":
    st.session_state.page = "Results Dashboard"

if "social_link_to_fetch" not in st.session_state:
    st.session_state.social_link_to_fetch = ""
if "text_to_analyze" not in st.session_state:
    st.session_state.text_to_analyze = ""

pages = [
    "Home",
    "Project Details",
    ACCOUNT_PAGE,
    "Text Analysis",
    "Social Media Analyzer",
    "Camera Detection",
    "Image Upload",
    "Results Dashboard",
    "Mood Journal",
    "History",
    "Help",
    "Privacy",
    "Features",
    "How It Works",
    "About Us",
]

requested_page = st.query_params.get("page", "")
requested_next_page = st.query_params.get("next", "")
if requested_next_page in PROTECTED_PAGES:
    remember_after_login(requested_next_page)

if requested_page in pages and requested_page != st.session_state.page:
    st.session_state.page = requested_page

if st.session_state.authenticated and st.session_state.page == ACCOUNT_PAGE:
    target_page = get_after_login_page("Home")
    st.session_state.after_login_page = ""
    st.session_state.page = target_page
    params = {"page": target_page}
    auth_token = st.query_params.get("auth", "")
    if auth_token:
        params["auth"] = auth_token
    st.query_params.from_dict(params)
    st.rerun()

if not st.session_state.authenticated and st.session_state.page in PROTECTED_PAGES:
    remember_after_login(st.session_state.page)
    st.session_state.page = ACCOUNT_PAGE
    st.query_params.from_dict({"page": ACCOUNT_PAGE, "next": get_after_login_page()})
    st.warning("Please sign in or create an account before using this feature.")

sidebar_pages = [page for page in pages if page != ACCOUNT_PAGE] if st.session_state.authenticated else [
    "Home",
    "Project Details",
    ACCOUNT_PAGE,
    "Help",
    "Privacy",
    "Features",
    "How It Works",
    "About Us",
]

if st.session_state.authenticated:
    display_name = html.escape(current_user_name() or current_mobile())
    st.sidebar.markdown(f"Signed in as **{display_name}**")
    st.sidebar.caption(current_mobile())
    if st.sidebar.button("Logout", use_container_width=True):
        logout_user()
else:
    st.sidebar.markdown("**Welcome to MindCare AI**")
    st.sidebar.caption("Sign in to use analysis tools, journal, dashboard, and history.")
    if st.sidebar.button("Sign in / Create account", use_container_width=True):
        st.session_state.after_login_page = ""
        go(ACCOUNT_PAGE)
        st.rerun()

side = st.sidebar.radio(
    "Menu",
    sidebar_pages,
    index=sidebar_pages.index(st.session_state.page) if st.session_state.page in sidebar_pages else 0,
)
if side != st.session_state.page:
    go(side)
    st.rerun()

st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="brand-left">
          <div class="logo">MC</div>
          <div>
            <h2>MindCare AI</h2>
            <p>Premium AI workspace for mental health pattern awareness</p>
          </div>
        </div>
        <div class="brand-pill">Private local analysis</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_links = ["Home", "Social Media Analyzer", "Camera Detection", "Image Upload", "Results Dashboard", "Mood Journal", "History"]
cols = st.columns([.75, 1.65, 1.35, 1.05, 1.42, 1.05, .82], gap="small")
for col, item in zip(cols, top_links):
    with col:
        if st.button(item, key="top_" + item):
            if st.session_state.authenticated or item not in PROTECTED_PAGES:
                go(item)
            else:
                remember_after_login(item)
                go(ACCOUNT_PAGE)
                st.query_params["next"] = item
            st.rerun()
st.markdown('<div class="nav-note"></div>', unsafe_allow_html=True)

page = st.session_state.page
model = load_model() if page in {"Text Analysis", "Social Media Analyzer"} else None

if page == "Home":
    latest_checks = read_history(1) if st.session_state.authenticated else pd.DataFrame()
    if latest_checks.empty:
        hero_title = "No checks yet"
        hero_subtitle = "Run an analysis to create your first saved signal"
        hero_score = "--"
        hero_status = "Waiting"
        hero_emotion = "Private"
    else:
        latest_check = latest_checks.iloc[0]
        hero_title = f"Latest {latest_check['detection_type']} check"
        hero_subtitle = trim_text(latest_check["input_data"], 78)
        hero_score = str(latest_check["score"])
        hero_status = str(latest_check["emotion"])
        hero_emotion = str(latest_check["result"])
    st.markdown(
        f"""
        <section class="hero">
          <div>
            <div class="eyebrow">AI assisted wellbeing intelligence</div>
            <h1>MindCare AI for <span>clearer emotional signals.</span></h1>
            <p>
              Analyze social media-style text, camera captures, and uploaded images
              with a polished dashboard built for awareness, review, and fast follow-up.
            </p>
          <div class="hero-actions">
              <span class="hero-chip">Text mood scan</span>
              <span class="hero-chip">Social post analyzer</span>
              <span class="hero-chip">Face emotion review</span>
              <span class="hero-chip">Mood journal</span>
              <span class="hero-chip">Result history</span>
            </div>
          </div>
          <div class="hero-visual">
            <div class="visual-row"><div><b>{html.escape(hero_title)}</b><br><span>{html.escape(hero_subtitle)}</span></div><div class="score-ring empty">{html.escape(hero_score)}</div></div>
            <div class="risk-strip">
              <div class="risk-pill"><span>Latest score</span><b>{html.escape(hero_score)}</b></div>
              <div class="risk-pill"><span>Emotion</span><b>{html.escape(hero_status)}</b></div>
              <div class="risk-pill"><span>Risk level</span><b>{html.escape(hero_emotion)}</b></div>
            </div>
            <div class="visual-row"><div><b>Signal status</b><br><span>Text, camera, and image results appear here after use</span></div><span>{html.escape(hero_status)}</span></div>
            <div class="visual-row"><div><b>Care note</b><br><span>Awareness tool, not diagnosis</span></div><span>{html.escape(hero_emotion)}</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    session_cols = st.columns([3, 1], gap="small")
    if st.session_state.authenticated:
        with session_cols[0]:
            st.markdown(
                f"""
                <div class="home-session-card">
                  <div>
                    <b>Welcome back, {html.escape(current_user_name() or current_mobile())}</b>
                    <span>Your session stays active after refresh on this device.</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with session_cols[1]:
            st.markdown(f'<a class="home-logout-link" href="{logout_url()}">Logout</a>', unsafe_allow_html=True)
    else:
        with session_cols[0]:
            st.markdown(
                """
                <div class="home-session-card">
                  <div>
                    <b>Start with a secure account</b>
                    <span>Sign in or create an account to unlock analysis tools, journal, dashboard, and private history.</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with session_cols[1]:
            st.markdown(
                f"""
                <div class="home-auth-actions">
                  <a class="home-auth-link primary" target="_self" href="{account_page_url()}">Sign in</a>
                  <a class="home-auth-link secondary" target="_self" href="{account_page_url()}">Create account</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <section class="quick-grid">
          <a class="feature-link" target="_self" href="{protected_page_url("Text Analysis")}"><div class="feature-card tone-teal"><div class="ico">TX</div><div class="feature-title">Text Analysis</div><p>Review posts, thoughts, and WhatsApp-style text for emotional risk patterns.</p></div></a>
          <a class="feature-link" target="_self" href="{protected_page_url("Social Media Analyzer")}"><div class="feature-card tone-gold"><div class="ico">SM</div><div class="feature-title">Social Media Analyzer</div><p>Analyze social post links, captions, and preview-image text safely.</p></div></a>
          <a class="feature-link" target="_self" href="{protected_page_url("Camera Detection")}"><div class="feature-card"><div class="ico">CA</div><div class="feature-title">Camera Detection</div><p>Capture an image and estimate the visible emotion with a focused result panel.</p></div></a>
          <a class="feature-link" target="_self" href="{protected_page_url("Image Upload")}"><div class="feature-card tone-violet"><div class="ico">UP</div><div class="feature-title">Image Upload</div><p>Upload a gallery image, preview it, and save the detected emotion automatically.</p></div></a>
          <a class="feature-link" target="_self" href="{protected_page_url("Results Dashboard")}"><div class="feature-card tone-rose"><div class="ico">DB</div><div class="feature-title">Results Dashboard</div><p>See total checks, average confidence, common emotion, and trend summary.</p></div></a>
          <a class="feature-link" target="_self" href="{protected_page_url("Mood Journal")}"><div class="feature-card tone-gold"><div class="ico">MJ</div><div class="feature-title">Mood Journal</div><p>Save daily mood notes, gratitude, and personal reflections privately.</p></div></a>
          <a class="feature-link" target="_self" href="{page_url("Privacy")}"><div class="feature-card"><div class="ico">PR</div><div class="feature-title">Privacy</div><p>Understand local storage, sensitive data handling, and project limitations.</p></div></a>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Latest checks</span>
              <h2>Recent History</h2>
              <p>Your latest saved signals are kept clean and easy to scan.</p>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )
    render_recent_history(4)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == ACCOUNT_PAGE:
    if st.session_state.authenticated:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.success("You are already signed in.")
        if st.button("Go to dashboard", use_container_width=True):
            go("Results Dashboard")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        render_auth_page()

elif page == "Project Details":
    st.markdown(
        r"""
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Project overview</span>
              <h1>AI for Detecting Mental Health Patterns from Social Media</h1>
              <p>Artificial Intelligence can analyze social media posts, text, images, and emotions to identify possible patterns such as stress, anxiety, sadness, or emotional changes.</p>
            </div>
          </div>
          <div class="metric-row">
            <div class="metric-tile"><b>NLP</b><span>Text signal processing</span></div>
            <div class="metric-tile"><b>Vision</b><span>Face emotion review</span></div>
            <div class="metric-tile"><b>SQLite</b><span>Local result history</span></div>
          </div>
          <p>This project uses NLP, Machine Learning, Emotion Detection, and Image Analysis to support mental health awareness from social media-style text and images.</p>
          <h2>Objectives</h2>
          <div class="grid2">
            <div class="info-card"><h3>Main Goals</h3><ul><li>Detect mental health risk using AI</li><li>Analyze text from social media posts</li><li>Detect emotions from face images</li></ul></div>
            <div class="info-card"><h3>Website Goals</h3><ul><li>Maintain user history and mood journal</li><li>Create a modern responsive website</li><li>Show confidence score, result explanation, and wellness suggestions</li></ul></div>
          </div>
          <h2>Technologies Used</h2>
          <p><span class="badge">Python</span><span class="badge">Streamlit</span><span class="badge">Machine Learning</span><span class="badge">NLP</span><span class="badge">DeepFace</span><span class="badge">SQLite</span><span class="badge">Pandas</span><span class="badge">Scikit-learn</span><span class="badge">OpenCV</span></p>
        </div>

        <div class="section">
          <div class="lux-title"><div><span class="kicker">System modules</span><h2>Modules</h2><p>Each module has one job, so the experience stays simple and focused.</p></div></div>
          <div class="grid3">
            <div class="info-card"><h3>1. Text Analysis</h3><p>User enters social media-style text. The system cleans text, converts it using TF-IDF, and predicts risk score using Logistic Regression.</p><p><b>Example:</b> Aaje mood saro nathi, khub tension lage chhe</p></div>
            <div class="info-card"><h3>2. Social Media Analyzer</h3><p>User pastes a caption or post text manually. The app analyzes only the text pattern and does not diagnose the real person behind the post.</p></div>
            <div class="info-card"><h3>3. Camera Detection</h3><p>User captures an image using camera. AI detects emotions like Happy, Sad, Angry, Fear, and Neutral.</p></div>
            <div class="info-card"><h3>4. Image Upload</h3><p>User uploads an image from gallery. Emotion detection is performed and image preview is shown.</p></div>
            <div class="info-card"><h3>5. Detection Result</h3><p>Shows risk level, emotion, confidence score, result explanation, and wellness suggestions.</p><p><b>Example:</b> Emotion: Sad, Confidence: 89%, Result: Possible Mental Health Risk</p></div>
            <div class="info-card"><h3>6. Results Dashboard</h3><p>Summarizes total checks, average confidence, priority signals, common emotion, and recent trend.</p></div>
            <div class="info-card"><h3>7. Mood Journal</h3><p>Lets the user save mood score, feeling label, reflection, gratitude, and date.</p></div>
            <div class="info-card"><h3>8. Privacy and Help</h3><p>Explains local data storage, project limitations, and mental health support resources.</p></div>
          </div>
        </div>

        <div class="section">
          <div class="lux-title"><div><span class="kicker">Flow</span><h2>Working Process</h2><p>A clean six-step pipeline from input to stored history.</p></div></div>
          <div class="step"><div class="step-num">1</div><div><h3>Input</h3><p>User enters text, camera image, or uploaded image.</p></div></div>
          <div class="step"><div class="step-num">2</div><div><h3>Preprocessing</h3><p>Text cleaning removes symbols and URLs, then converts text to lowercase.</p></div></div>
          <div class="step"><div class="step-num">3</div><div><h3>Feature Extraction</h3><p>TF-IDF converts text into numeric values.</p></div></div>
          <div class="step"><div class="step-num">4</div><div><h3>Prediction</h3><p>Machine Learning predicts Low Risk, Medium Risk, or High Risk.</p></div></div>
          <div class="step"><div class="step-num">5</div><div><h3>Emotion Detection</h3><p>DeepFace detects facial emotion.</p></div></div>
          <div class="step"><div class="step-num">6</div><div><h3>Store History</h3><p>All results are saved in SQLite database.</p></div></div>
          <div class="step"><div class="step-num">7</div><div><h3>Review and Reflect</h3><p>The dashboard, mood journal, privacy page, and help page support safer awareness and reflection.</p></div></div>
        </div>

        <div class="section">
          <div class="grid3">
            <div class="info-card"><h3>Advantages</h3><ul><li>Fast detection</li><li>Easy to use</li><li>Modern UI</li><li>Supports social media-style text</li><li>Detects emotion from images</li><li>Includes journal and dashboard reflection</li></ul></div>
            <div class="info-card"><h3>Limitations</h3><ul><li>Not a medical diagnosis system</li><li>Accuracy depends on data</li><li>Internet/camera permissions may be required</li></ul></div>
            <div class="info-card"><h3>Future Enhancements</h3><ul><li>Real-time live camera detection</li><li>Voice emotion analysis</li><li>Chatbot support</li><li>Multi-language support</li><li>Cloud deployment</li></ul></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Text Analysis":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Text signal</span>
              <h1>Text Analysis</h1>
              <p>You can type English or WhatsApp-style Gujarati-English text.</p>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )
    text = st.text_area(
        "Enter your text",
        height=170,
        placeholder="Example: Aaje mood saro nathi, khub tension lage chhe",
    )
    if st.button("Analyze Text"):
        if text.strip():
            st.session_state.text_to_analyze = text.strip()
        else:
            st.warning("Please enter text first.")

    analyzed_text = st.session_state.text_to_analyze
    if analyzed_text:
        base_score = model.predict_proba([clean_text(analyzed_text)])[0][1] * 100
        score = adjusted_text_score(analyzed_text, base_score)
        emotion = get_text_emotion(analyzed_text, score)
        result = show_result("Text Analysis Result", score, emotion, analyzed_text)
        if st.button("Save Text Result"):
            save_history("Text", analyzed_text, "", result, f"{score:.2f}%", emotion)
            st.success("Saved to history.")
    else:
        st.info("Enter text and click Analyze Text to see detection.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Social Media Analyzer":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Social post signal</span>
              <h1>Social Media Analyzer</h1>
              <p>Add an Instagram/post/image link as a reference and paste caption text. The app will try to fetch the preview image, read text inside that image with OCR when available, and analyze content signals safely.</p>
            </div>
          </div>
          <div class="care-note">Safe use: some social sites block previews. If a preview image is found, MindCare AI reads visible quote/text automatically. This is still an awareness signal, not a diagnosis of a real person.</div>
        """,
        unsafe_allow_html=True,
    )
    instagram_link = st.text_input(
        "Instagram/post/image link (optional)",
        placeholder="Example: https://www.instagram.com/p/...",
    )
    if st.button("Analyze Link Preview"):
        if instagram_link.strip():
            st.session_state.social_link_to_fetch = instagram_link.strip()
        else:
            st.warning("Please paste a link first.")

    post_text = st.text_area(
        "Paste social media post/caption text",
        height=190,
        placeholder="Example: I am tired of pretending everything is okay. Too much pressure lately.",
    )
    source_note = st.text_input(
        "Optional source note",
        placeholder="Example: Instagram caption / public post / my own post",
    )

    social_results = []
    social_image_path = ""
    extracted_text = ""
    visual_emotion = ""
    visual_score = 0
    link_to_fetch = st.session_state.social_link_to_fetch

    if link_to_fetch:
        with st.spinner("Trying to fetch preview image from link..."):
            social_image_path, fetch_error = fetch_preview_image_from_link(link_to_fetch)
        st.markdown(
            f"""
            <div class="social-preview">
              <h3>Reference link</h3>
              <p>{html.escape(link_to_fetch)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if social_image_path:
            st.markdown('<div class="social-preview"><h3>Preview image from link</h3>', unsafe_allow_html=True)
            st.image(social_image_path, caption="Fetched preview image", width=340)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.spinner("Reading text inside preview image..."):
                extracted_text = extract_text_from_image_optional(social_image_path)
            if extracted_text:
                st.markdown(
                    f"""
                    <div class="social-preview">
                      <h3>Text detected in image</h3>
                      <p>{html.escape(extracted_text)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No text was detected inside this preview image. The app will use visual emotion only if no caption text is available.")

            with open(social_image_path, "rb") as fetched_image:
                image_bytes = fetched_image.read()
            ext = os.path.splitext(social_image_path)[1] or ".jpg"
            with st.spinner("Analyzing preview image..."):
                visual_emotion, visual_score = analyze_image_cached(image_bytes, ext)
        elif fetch_error:
            st.warning(fetch_error)

    combined_social_text = " ".join(part for part in [post_text.strip(), extracted_text.strip()] if part)
    if combined_social_text:
        base_score = model.predict_proba([clean_text(combined_social_text)])[0][1] * 100
        score = adjusted_text_score(combined_social_text, base_score)
        emotion = get_text_emotion(combined_social_text, score)
        result = show_result("Social Media Link Content Result", score, emotion, combined_social_text)
        social_results.append(("Content Text", result, score, emotion))
    elif social_image_path:
        result = show_result("Social Media Visual Result", visual_score, visual_emotion, link_to_fetch)
        social_results.append(("Visual", result, visual_score, visual_emotion))

    if social_results:
        if st.button("Save Social Media Result"):
            latest_type, latest_result, latest_score, latest_emotion = social_results[-1]
            label_parts = []
            if source_note.strip():
                label_parts.append(source_note.strip())
            if link_to_fetch:
                label_parts.append(link_to_fetch)
            if post_text.strip():
                label_parts.append(post_text.strip())
            if extracted_text.strip():
                label_parts.append(f"Detected image text: {extracted_text.strip()}")
            if not label_parts:
                label_parts.append("Fetched social media preview image")
            save_history(
                f"Social Media {latest_type}",
                " | ".join(label_parts),
                social_image_path,
                latest_result,
                f"{latest_score:.2f}%",
                latest_emotion,
            )
            st.success("Saved to history as Social Media.")
    else:
        st.info("Paste a caption/post text or add a link that has a public preview image to start analysis.")
    st.markdown("</div>", unsafe_allow_html=True)
elif page == "Camera Detection":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Camera signal</span>
              <h1>Camera Detection</h1>
              <p>Capture a photo and review the estimated facial emotion in a focused panel.</p>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )
    img = st.camera_input("Take a photo")
    if img:
        image_bytes = img.getvalue()
        path = save_image(image_bytes, ".jpg")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(path, caption="Captured Image", width=280)
        with col2:
            with st.spinner("Analyzing image..."):
                emotion, score = analyze_image_cached(image_bytes, ".jpg")
            result = show_result("Camera Emotion Result", score, emotion, "Captured image")
            key = "cam_" + path
            if key not in st.session_state:
                save_history("Camera", "Captured image", path, result, f"{score:.2f}%", emotion)
                st.session_state[key] = True
                st.success("Saved automatically.")
    else:
        st.info("Capture a photo to start detection.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Image Upload":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Image signal</span>
              <h1>Image Upload</h1>
              <p>Upload a photo, preview it, and save the emotion result automatically.</p>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image_bytes = uploaded.getvalue()
        ext = ".png" if uploaded.name.lower().endswith(".png") else ".jpg"
        path = save_image(image_bytes, ext)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(path, caption="Uploaded Image", width=280)
        with col2:
            with st.spinner("Analyzing uploaded image..."):
                emotion, score = analyze_image_cached(image_bytes, ext)
            result = show_result("Uploaded Image Result", score, emotion, uploaded.name)
            key = f"up_{uploaded.name}_{len(image_bytes)}"
            if key not in st.session_state:
                save_history("Image", uploaded.name, path, result, f"{score:.2f}%", emotion)
                st.session_state[key] = True
                st.success("Saved automatically.")
    else:
        st.markdown('<div class="upload-empty-state">Upload an image to start detection.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Results Dashboard":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Analytics</span>
              <h1>Results Dashboard</h1>
              <p>A polished summary of saved checks, average signal strength, common emotion, and recent trend.</p>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )
    df = read_history()
    mood_df = read_mood_entries()
    if df.empty and mood_df.empty:
        st.info("No saved data found yet. Run an analysis or save a daily Mood Journal entry first.")
    if not df.empty:
        stats = dashboard_stats(df)
        st.markdown(
            f"""
            <div class="metric-row">
              <div class="metric-tile"><b>{stats['total']}</b><span>Total saved checks</span></div>
              <div class="metric-tile"><b>{stats['average']}%</b><span>Average confidence</span></div>
              <div class="metric-tile"><b>{stats['priority']}</b><span>Priority signals</span></div>
            </div>
            <div class="grid2">
              <div class="info-card"><h3>Most common emotion</h3><p>{html.escape(stats['common_emotion'])}</p></div>
              <div class="info-card"><h3>Recent trend</h3><p>{html.escape(trend_label(df))}</p><p><b>Latest:</b> {html.escape(stats['latest'])}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        detection_chart = prepare_detection_chart(df)
        if not detection_chart.empty:
            st.markdown("<h3>Mental Health Risk Graph</h3>", unsafe_allow_html=True)
            st.line_chart(detection_chart, use_container_width=True)
        st.markdown("<h3>Detailed Results</h3>", unsafe_allow_html=True)
        st.dataframe(
            df[["detection_type", "input_data", "result", "score", "emotion", "created_at"]],
            use_container_width=True,
        )
    else:
        st.info("No detection results found. Run Text Analysis, Camera Detection, or Image Upload first.")

    if not mood_df.empty:
        mood_summary = mood_stats(mood_df)
        st.markdown(
            f"""
            <div class="metric-row">
              <div class="metric-tile"><b>{mood_summary['total']}</b><span>Journal entries</span></div>
              <div class="metric-tile"><b>{mood_summary['average']}/10</b><span>Average mood score</span></div>
              <div class="metric-tile"><b>{html.escape(mood_summary['common'])}</b><span>Common mood</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        mood_chart = prepare_mood_chart(mood_df)
        if not mood_chart.empty:
            render_mood_health_graph(mood_df, "Daily Mood and Mental Health Graph")
        st.markdown("<h3>Recent Mood Journal</h3>", unsafe_allow_html=True)
        show_cols = ["mood_score", "mood_label", "mood_kind", "note", "gratitude", "created_at"]
        st.dataframe(mood_df[[col for col in show_cols if col in mood_df.columns]], use_container_width=True)
    else:
        st.info("No Mood Journal entries yet. Add your daily mood to see the mental health graph.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Mood Journal":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Reflection</span>
              <h1>Mood Journal</h1>
              <p>Save a private daily check-in beside your AI detection history.</p>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )
    mood_score = st.slider("Mood score", 1, 10, 6, help="1 means very low, 10 means very positive.")
    mood_label = st.selectbox("Today I feel", ["Calm", "Happy", "Neutral", "Stressed", "Sad", "Anxious", "Tired"])
    mood_kind = st.selectbox(
        "Daily mood kind",
        ["Balanced", "Positive", "Low mood", "Stress", "Anxiety", "Tired / low energy", "Angry / irritated"],
    )
    note = st.text_area("What happened today?", height=130, placeholder="Write a short reflection...")
    gratitude = st.text_input("One good thing or gratitude", placeholder="Example: I completed my work / talked to a friend")
    if st.button("Save Journal Entry"):
        save_mood_entry(mood_score, mood_label, mood_kind, note, gratitude)
        st.success("Mood journal entry saved.")

    all_entries = read_mood_entries()
    entries = all_entries.head(6) if not all_entries.empty else all_entries
    if not entries.empty:
        mood_chart = prepare_mood_chart(all_entries)
        if not mood_chart.empty:
            render_mood_health_graph(all_entries, "Daily Mood Graph")
        st.markdown("<h3>Recent Journal Entries</h3>", unsafe_allow_html=True)
        for _, entry in entries.iterrows():
            kind = str(entry.get("mood_kind", "") or "Not added")
            st.markdown(
                f"""
                <div class="journal-entry">
                  <h3>{html.escape(str(entry['mood_label']))} &middot; {int(entry['mood_score'])}/10</h3>
                  <p><b>Mood kind:</b> {html.escape(kind)}</p>
                  <p>{html.escape(trim_text(entry['note'], 220))}</p>
                  <p><b>Gratitude:</b> {html.escape(str(entry['gratitude'] or 'Not added'))}</p>
                  <p><b>Date:</b> {html.escape(str(entry['created_at']))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.download_button(
            "Download Journal CSV",
            entries.to_csv(index=False).encode("utf-8"),
            "mood_journal.csv",
            "text/csv",
        )
    else:
        st.info("No mood journal entries yet. Save today's mood to build your daily mental health graph.")
    if st.button("Clear Mood Journal"):
        clear_mood_entries()
        st.success("Mood journal cleared. Refresh page.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "History":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Timeline</span>
              <h1>History</h1>
              <p>Every saved analysis is organized as a clean timeline card with signal, score, and date.</p>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )
    df = read_history()
    if df.empty:
        st.info("No history found yet.")
    else:
        st.markdown(
            f"""
            <div class="metric-row">
              <div class="metric-tile"><b>{len(df)}</b><span>Total saved checks</span></div>
              <div class="metric-tile"><b>{df['detection_type'].nunique()}</b><span>Detection types used</span></div>
              <div class="metric-tile"><b>{html.escape(str(df.iloc[0]['score']))}</b><span>Latest confidence</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_history_cards(df)
        st.download_button(
            "Download History CSV",
            df.to_csv(index=False).encode("utf-8"),
            "history.csv",
            "text/csv",
        )
    if st.button("Clear History"):
        clear_history()
        st.success("History cleared. Refresh page.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Help":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Support</span>
              <h1>Need Help Now?</h1>
              <p>MindCare AI can show awareness signals, but urgent support should come from trusted people, emergency services, or trained crisis professionals.</p>
            </div>
          </div>
          <div class="resource-panel">
            <p><strong>If there is immediate danger:</strong> call your local emergency number or go to the nearest emergency department.</p>
            <p><strong>United States:</strong> call or text <b>988</b> for the 988 Suicide & Crisis Lifeline.</p>
            <p><strong>India:</strong> Tele-MANAS provides mental health support at <b>14416</b> or <b>1800-89-14416</b>.</p>
            <p><strong>If it is not urgent:</strong> consider talking to a trusted friend, family member, teacher, mentor, counselor, or qualified mental health professional.</p>
          </div>
        </div>
        <div class="section">
          <div class="grid3">
            <div class="info-card"><h3>Grounding</h3><p>Name five things you can see, four you can feel, three you can hear, two you can smell, and one you can taste.</p></div>
            <div class="info-card"><h3>Breathing</h3><p>Try slow breathing for two minutes: inhale for four counts, hold briefly, and exhale slowly.</p></div>
            <div class="info-card"><h3>Connection</h3><p>Send one simple message to someone safe: â€œI am having a hard moment. Can you stay with me?â€</p></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Privacy":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Privacy and limits</span>
              <h1>Privacy</h1>
              <p>Mental health data is sensitive, so this project is designed with clear local-storage expectations.</p>
            </div>
          </div>
          <div class="privacy-list">
            <div class="info-card"><h3>Local Storage</h3><p>Detection history and mood journal entries are saved in the local SQLite database file named <b>history.db</b>.</p></div>
            <div class="info-card"><h3>Image Storage</h3><p>Uploaded and captured images are saved locally in the <b>history_images</b> folder for history preview.</p></div>
            <div class="info-card"><h3>Awareness Only</h3><p>MindCare AI is not a medical diagnosis system and should not replace care from qualified professionals.</p></div>
            <div class="info-card"><h3>User Control</h3><p>You can clear detection history from the History page and clear mood notes from Mood Journal.</p></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Features":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Capabilities</span>
              <h1>Features</h1>
              <p>Focused AI tools for text, image, camera, and saved result review.</p>
            </div>
          </div>
          <div class="grid2">
            <div class="info-card"><h3>Text Analysis</h3><p>Analyze English and WhatsApp-style Gujarati-English text for possible emotional patterns.</p></div>
            <div class="info-card"><h3>Social Media Analyzer</h3><p>Paste social media captions, posts, or comments manually and analyze text signals safely without diagnosing a real person.</p></div>
            <div class="info-card"><h3>Camera Detection</h3><p>Use the camera to capture a face image and estimate the visible emotion.</p></div>
            <div class="info-card"><h3>Image Upload Detection</h3><p>Upload an image from your gallery, preview it, analyze it, and save the result.</p></div>
            <div class="info-card"><h3>Result Explanation</h3><p>See why a result appeared, including detected text signals or image emotion confidence.</p></div>
            <div class="info-card"><h3>Wellness Suggestions</h3><p>Get gentle next-step suggestions based on low, medium, or priority risk signals.</p></div>
            <div class="info-card"><h3>Results Dashboard</h3><p>Review average confidence, priority signals, common emotion, and recent trend.</p></div>
            <div class="info-card"><h3>Mood Journal</h3><p>Save daily mood score, feeling label, reflection, and gratitude notes.</p></div>
            <div class="info-card"><h3>History</h3><p>Review recent and full detection history, then export it as a CSV when needed.</p></div>
            <div class="info-card"><h3>Privacy and Help</h3><p>Understand local storage and view support resources for difficult moments.</p></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "How It Works":
    st.session_state.page = "Project Details"
    st.rerun()

elif page == "About Us":
    st.markdown(
        """
        <div class="section">
          <div class="lux-title">
            <div>
              <span class="kicker">Purpose</span>
              <h1>About Us</h1>
            </div>
          </div>
          <p>MindCare AI is an educational project for mental health awareness using AI.</p>
          <p>This project is not a medical diagnosis system. It is designed to support awareness and learning through simple AI-powered signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
