#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_app.py - Single-file Flask RAG over PDF (Thai/English) ผ่าน OpenRouter

ใช้ OpenRouter คีย์เดียวทั้ง embedding และ chat
API ของ OpenRouter เป็น OpenAI-compatible จึงเรียกด้วย requests ตรง ๆ ได้ ไม่ต้องพึ่ง SDK

ติดตั้ง:
    pip install flask pypdf numpy requests
    # ตัวเลือก: ถ้าจะ embed บนเครื่องแทนการยิง API
    pip install sentence-transformers

ตั้งค่า: สร้างไฟล์ .env ไว้ข้าง ๆ สคริปต์นี้

    OPENROUTER_API_KEY=sk-or-v1-...
    CHAT_MODEL=anthropic/claude-sonnet-5        # ดู slug ที่ openrouter.ai/models
    EMBED_BACKEND=openrouter                    # openrouter | local
    EMBED_MODEL=openai/text-embedding-3-large
    RAG_DB=rag.db

    อย่าลืมใส่ .env ใน .gitignore
    ตัวแปรที่ตั้งไว้ใน shell อยู่แล้วจะชนะค่าใน .env เสมอ

รัน:
    python rag_app.py                       # http://127.0.0.1:5000
    python rag_app.py --env /path/to/.env   # ระบุไฟล์ .env เอง
    python rag_app.py --list-embed-models   # ดูรายชื่อ embedding model ที่ใช้ได้จริงตอนนี้
"""

import io
import json
import os
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime

import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template_string, request
from pypdf import PdfReader

# ---------------------------------------------------------------- .env

def load_dotenv(path=None):
    """
    อ่าน .env เองแบบง่าย ๆ ไม่ต้องลง python-dotenv เพิ่ม
    รองรับ: คอมเมนต์ #, บรรทัดว่าง, คำนำหน้า export, ค่าที่ครอบด้วย " หรือ '

    ใช้ setdefault ตั้งใจ - ตัวแปรที่ตั้งไว้ใน shell หรือใน docker
    ต้องชนะค่าใน .env เสมอ ไม่งั้นเวลา deploy จะงงว่าทำไมคีย์ไม่เปลี่ยน
    """
    if path is None:
        for cand in (os.path.join(os.getcwd(), ".env"),
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
            if os.path.isfile(cand):
                path = cand
                break
    if not path or not os.path.isfile(path):
        return None

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]                  # ค่าที่ครอบ quote เก็บทั้งก้อน
            elif " #" in val:
                val = val.split(" #", 1)[0].rstrip()   # ตัดคอมเมนต์ท้ายบรรทัด
            if key:
                os.environ.setdefault(key, val)
    return path


_env_arg = None
if "--env" in sys.argv:
    i = sys.argv.index("--env")
    if i + 1 < len(sys.argv):
        _env_arg = sys.argv[i + 1]

ENV_FILE = load_dotenv(_env_arg)


def mask(secret):
    if not secret:
        return "(ไม่มี)"
    return f"{secret[:10]}…{secret[-4:]}" if len(secret) > 18 else "(สั้นผิดปกติ)"


# ---------------------------------------------------------------- config

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

APP_TITLE = "PDF RAG"                       # โผล่ใน leaderboard ของ OpenRouter
APP_URL = os.environ.get("APP_URL", "http://localhost:5000")

DB_PATH = os.environ.get("RAG_DB", "rag.db")
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "openrouter").lower()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "openai/text-embedding-3-large")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "anthropic/claude-sonnet-5")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 900))       # ตัวอักษร
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 180))
TOP_K = int(os.environ.get("TOP_K", 6))
MIN_SCORE = float(os.environ.get("MIN_SCORE", 0.25))      # cosine ต่ำกว่านี้ = ตัดทิ้ง
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 2048))
MAX_UPLOAD_MB = 50

SYSTEM_PROMPT = """คุณเป็นผู้ช่วยตอบคำถามจากเอกสารที่ผู้ใช้อัปโหลด

กติกา:
- ตอบโดยอ้างอิงจาก <context> ที่ให้มาเท่านั้น ห้ามเดาหรือเติมความรู้ทั่วไปโดยไม่บอก
- ถ้าข้อมูลใน context ไม่พอ ให้บอกตรง ๆ ว่าเอกสารไม่ได้ระบุ แล้วบอกว่าต้องการข้อมูลอะไรเพิ่ม
- ใส่เลขอ้างอิงท้ายประโยคที่ดึงข้อมูลมา เช่น [1] [3] ตรงกับหมายเลขบล็อกใน context
- ตอบเป็นภาษาเดียวกับคำถามของผู้ใช้
- ถ้าเป็นสเปกทางเทคนิค ตัวเลข หรือค่าพิกัด ให้คัดมาตรง ๆ อย่าปัดเศษเอง"""


def or_headers():
    if not API_KEY:
        raise RuntimeError("ยังไม่ได้ตั้ง OPENROUTER_API_KEY")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_URL,
        "X-Title": APP_TITLE,
    }


def or_error(resp):
    """OpenRouter ส่ง error เป็น JSON {"error":{"message":...}} ดึงข้อความจริงออกมา"""
    try:
        return resp.json()["error"]["message"]
    except Exception:
        return resp.text[:300]


# ---------------------------------------------------------------- embedder

RETRY_CODES = {408, 429, 500, 502, 503, 520, 524, 529}


class OpenRouterEmbedder:
    """POST /embeddings รูปแบบเดียวกับ OpenAI"""

    BATCH = 64          # ส่วนใหญ่รับได้ถึง 96 ต่อ request เผื่อไว้หน่อย

    def __init__(self, model):
        self.model = model
        self.name = model

    def embed(self, texts, is_query=False):
        vecs = []
        for i in range(0, len(texts), self.BATCH):
            vecs.extend(self._call(texts[i:i + self.BATCH]))
        v = np.asarray(vecs, dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v

    def _call(self, batch, tries=4):
        last = ""
        for attempt in range(tries):
            r = requests.post(
                f"{OPENROUTER_BASE}/embeddings",
                headers=or_headers(),
                json={"model": self.model, "input": batch, "encoding_format": "float"},
                timeout=120,
            )
            if r.status_code == 200:
                data = r.json()["data"]
                # ลำดับที่คืนมาไม่การันตี ต้องเรียงตาม index เอง ไม่งั้น chunk สลับเวกเตอร์
                data.sort(key=lambda d: d.get("index", 0))
                return [d["embedding"] for d in data]
            last = f"{r.status_code}: {or_error(r)}"
            if r.status_code in RETRY_CODES:
                time.sleep(2 ** attempt)
                continue
            break
        raise RuntimeError(f"embeddings ล้มเหลว {last}")


class LocalEmbedder:
    """sentence-transformers รันบนเครื่อง ฟรี ใช้เมื่อไม่อยากส่งเอกสารออกนอกบริษัท"""

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        if "/" in model_name and model_name.startswith("openai/"):
            model_name = "intfloat/multilingual-e5-small"
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        # ตระกูล e5 ต้องมี prefix ไม่งั้นคุณภาพตกอย่างชัดเจน
        self.use_prefix = "e5" in model_name.lower()

    def embed(self, texts, is_query=False):
        if self.use_prefix:
            p = "query: " if is_query else "passage: "
            texts = [p + t for t in texts]
        v = self.model.encode(texts, batch_size=16, normalize_embeddings=True,
                              show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


_embedder = None
_embedder_lock = threading.Lock()


def get_embedder():
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = (LocalEmbedder(EMBED_MODEL) if EMBED_BACKEND == "local"
                         else OpenRouterEmbedder(EMBED_MODEL))
        return _embedder


def list_embed_models():
    r = requests.get(f"{OPENROUTER_BASE}/embeddings/models",
                     headers=or_headers(), timeout=30)
    if r.status_code != 200:
        raise RuntimeError(or_error(r))
    return r.json().get("data", [])


# ---------------------------------------------------------------- chat

def stream_chat(messages):
    """yield ข้อความทีละชิ้นจาก /chat/completions (SSE)"""
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.2,
        "stream": True,
    }
    with requests.post(f"{OPENROUTER_BASE}/chat/completions", headers=or_headers(),
                       json=payload, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {or_error(r)}")
        # OpenRouter ไม่ระบุ charset มาใน Content-Type ของ text/event-stream
        # requests เลยถอยไปใช้ ISO-8859-1 ตามสเปก HTTP เก่า แล้วภาษาไทยกลายเป็น mojibake
        # ต้องบังคับเองก่อนเรียก iter_lines(decode_unicode=True)
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith(":"):
                continue                      # keepalive comment ของ OpenRouter
            if not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if body == "[DONE]":
                return
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if "error" in chunk:              # error กลางสตรีมก็มาแบบ 200 ได้
                raise RuntimeError(chunk["error"].get("message", "unknown error"))
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text = (choices[0].get("delta") or {}).get("content")
            if text:
                yield text


# ---------------------------------------------------------------- pdf + chunking

def clean_text(t):
    t = t.replace("\u00a0", " ").replace("\ufeff", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_pages(file_bytes):
    """คืน (list ของ (page_no, text) เฉพาะหน้าที่มีข้อความ, จำนวนหน้าทั้งหมด)"""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            text = ""
        if text:
            pages.append((i, text))
    return pages, len(reader.pages)


def chunk_pages(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    ต่อทุกหน้าเป็นข้อความเดียว แล้วตัดเป็น window ตามจำนวนตัวอักษร
    เก็บ offset ของแต่ละหน้าไว้ เพื่อ map กลับว่า chunk นี้มาจากหน้าไหน

    ใช้ char window เพราะภาษาไทยไม่มีเว้นวรรคระหว่างคำ
    ถ้าตัดด้วย whitespace แบบภาษาอังกฤษ ทั้งย่อหน้าจะกลายเป็นก้อนเดียว
    """
    parts, spans, cursor = [], [], 0
    for page_no, text in pages:
        parts.append(text)
        spans.append((cursor, cursor + len(text), page_no))
        cursor += len(text) + 1
    doc = "\n".join(parts)

    def page_of(pos):
        for s, e, p in spans:
            if s <= pos < e:
                return p
        return spans[-1][2] if spans else 1

    chunks, i, n = [], 0, len(doc)
    while i < n:
        end = min(i + size, n)
        if end < n:
            # ถอยหาจุดตัดที่สวยกว่าภายใน 25% ท้าย window
            window_start = max(i + int(size * 0.75), i + 1)
            cut = -1
            for sep in ("\n\n", "\n", ". ", " "):
                found = doc.rfind(sep, window_start, end)
                if found > cut:
                    cut = found + len(sep)
            if cut > window_start:
                end = cut
        piece = doc[i:end].strip()
        if len(piece) >= 40:
            chunks.append({"text": piece,
                           "page_start": page_of(i),
                           "page_end": page_of(max(i, end - 1))})
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


# ---------------------------------------------------------------- store

class Store:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self._init_db()
        self.matrix = None      # (N, dim) float32 normalized
        self.meta = []
        self.reload()

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value TEXT
            );
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                n_pages INTEGER, n_chunks INTEGER, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                page_start INTEGER, page_end INTEGER,
                text TEXT NOT NULL, embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """)

    # --- index ต้องมาจาก embedding model ตัวเดียวกันทั้งหมด ---
    def index_model(self):
        with self._conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key='embed_model'").fetchone()
        return row["value"] if row else None

    def check_model(self, model_name):
        current = self.index_model()
        if current and current != model_name:
            raise RuntimeError(
                f"index นี้สร้างด้วย '{current}' แต่ตอนนี้ตั้งเป็น '{model_name}' "
                f"เวกเตอร์คนละสเปซเทียบกันไม่ได้ ให้ลบ {self.path} แล้วอัปโหลดใหม่ "
                f"หรือกลับไปใช้โมเดลเดิม"
            )

    def add_document(self, filename, chunks, vectors, n_pages, model_name):
        with self.lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO settings VALUES ('embed_model', ?)", (model_name,))
            cur = c.execute(
                "INSERT INTO documents (filename, n_pages, n_chunks, created_at) VALUES (?,?,?,?)",
                (filename, n_pages, len(chunks),
                 datetime.now().isoformat(timespec="seconds")))
            doc_id = cur.lastrowid
            c.executemany(
                "INSERT INTO chunks (doc_id, page_start, page_end, text, embedding) VALUES (?,?,?,?,?)",
                [(doc_id, ch["page_start"], ch["page_end"], ch["text"],
                  vectors[i].astype(np.float32).tobytes())
                 for i, ch in enumerate(chunks)])
        self.reload()
        return doc_id

    def delete_document(self, doc_id):
        with self.lock, self._conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            if not c.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
                c.execute("DELETE FROM settings WHERE key='embed_model'")
        self.reload()

    def list_documents(self):
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT id, filename, n_pages, n_chunks, created_at "
                "FROM documents ORDER BY id DESC")]

    def reload(self):
        """โหลด embedding ทั้งหมดขึ้น RAM เป็น matrix เดียว - พอสำหรับหลักหมื่น chunk"""
        with self._conn() as c:
            rows = list(c.execute("""
                SELECT ch.id, ch.doc_id, ch.page_start, ch.page_end, ch.text,
                       ch.embedding, d.filename
                FROM chunks ch JOIN documents d ON d.id = ch.doc_id ORDER BY ch.id"""))
        if not rows:
            self.matrix, self.meta = None, []
            return
        self.matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32)
                                 for r in rows])
        self.meta = [{"id": r["id"], "doc_id": r["doc_id"], "filename": r["filename"],
                      "page_start": r["page_start"], "page_end": r["page_end"],
                      "text": r["text"]} for r in rows]

    def search(self, qvec, k=TOP_K, doc_ids=None):
        if self.matrix is None:
            return []
        scores = self.matrix @ qvec          # normalize แล้ว = cosine
        idx = np.arange(len(scores))
        if doc_ids:
            mask = np.array([m["doc_id"] in doc_ids for m in self.meta])
            idx, scores = idx[mask], scores[mask]
        if len(idx) == 0:
            return []
        out = []
        for j in np.argsort(-scores)[:k]:
            score = float(scores[j])
            if score < MIN_SCORE:
                continue
            m = dict(self.meta[idx[j]])
            m["score"] = round(score, 4)
            out.append(m)
        return out


store = Store(DB_PATH)


# ---------------------------------------------------------------- prompt

def build_context(hits):
    blocks = []
    for n, h in enumerate(hits, start=1):
        pages = (str(h["page_start"]) if h["page_start"] == h["page_end"]
                 else f'{h["page_start"]}-{h["page_end"]}')
        blocks.append(f'[{n}] ไฟล์: {h["filename"]} | หน้า {pages}\n{h["text"]}')
    return "\n\n---\n\n".join(blocks)


def build_messages(question, hits, history):
    msgs = []
    for turn in history[-6:]:
        content = (turn.get("content") or "").strip()
        if content:
            msgs.append({"role": "user" if turn.get("role") == "user" else "assistant",
                         "content": content})
    msgs.append({"role": "user",
                 "content": f"<context>\n{build_context(hits)}\n</context>\n\n"
                            f"คำถาม: {question}"})
    return msgs


# ---------------------------------------------------------------- flask

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/api/status")
def api_status():
    return jsonify({
        "documents": store.list_documents(),
        "chunks": len(store.meta),
        "backend": EMBED_BACKEND,
        "embed_model": store.index_model() or EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "has_key": bool(API_KEY),
    })


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "ต้องเป็นไฟล์ .pdf"}), 400

    t0 = time.time()
    try:
        pages, total_pages = extract_pages(f.read())
    except Exception as e:
        return jsonify({"error": f"อ่าน PDF ไม่ได้: {e}"}), 400

    if not pages:
        return jsonify({"error": "ไม่พบข้อความในไฟล์นี้ ถ้าเป็น PDF สแกน ต้องทำ OCR ก่อน "
                                 "เช่น ocrmypdf -l tha+eng input.pdf output.pdf"}), 400

    chunks = chunk_pages(pages)
    if not chunks:
        return jsonify({"error": "ตัด chunk ไม่ได้ เอกสารสั้นเกินไป"}), 400

    try:
        emb = get_embedder()
        store.check_model(emb.name)
        vectors = emb.embed([c["text"] for c in chunks], is_query=False)
    except Exception as e:
        return jsonify({"error": f"สร้าง embedding ไม่สำเร็จ - {e}"}), 500

    doc_id = store.add_document(f.filename, chunks, vectors, total_pages, emb.name)
    return jsonify({"id": doc_id, "filename": f.filename, "pages": total_pages,
                    "chunks": len(chunks), "seconds": round(time.time() - t0, 1)})


@app.delete("/api/documents/<int:doc_id>")
def api_delete(doc_id):
    store.delete_document(doc_id)
    return jsonify({"ok": True})


@app.post("/api/search")
def api_search():
    """ดู retrieval ดิบ ๆ โดยไม่เรียก LLM - ใช้ debug คุณภาพ chunk"""
    body = request.get_json(force=True)
    q = (body.get("question") or "").strip()
    if not q:
        return jsonify({"error": "ไม่มีคำถาม"}), 400
    try:
        qvec = get_embedder().embed([q], is_query=True)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"hits": store.search(qvec, k=int(body.get("k", TOP_K)))})


@app.post("/api/ask")
def api_ask():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    history = body.get("history") or []
    doc_ids = body.get("doc_ids") or None
    if not question:
        return jsonify({"error": "ไม่มีคำถาม"}), 400
    if store.matrix is None:
        return jsonify({"error": "ยังไม่มีเอกสารในระบบ อัปโหลด PDF ก่อน"}), 400

    try:
        qvec = get_embedder().embed([question], is_query=True)[0]
    except Exception as e:
        return jsonify({"error": f"embed คำถามไม่สำเร็จ - {e}"}), 500
    hits = store.search(qvec, k=TOP_K, doc_ids=set(doc_ids) if doc_ids else None)

    def stream():
        if not hits:
            yield sse({"type": "delta",
                       "text": "ไม่พบเนื้อหาที่เกี่ยวข้องกับคำถามนี้ในเอกสารที่มีอยู่"})
            yield sse({"type": "done"})
            return
        yield sse({"type": "sources", "sources": [
            {"n": i, "filename": h["filename"], "page_start": h["page_start"],
             "page_end": h["page_end"], "score": h["score"], "preview": h["text"][:300]}
            for i, h in enumerate(hits, start=1)]})
        try:
            for piece in stream_chat(build_messages(question, hits, history)):
                yield sse({"type": "delta", "text": piece})
        except Exception as e:
            yield sse({"type": "error", "text": f"เรียกโมเดลไม่สำเร็จ - {e}"})
        yield sse({"type": "done"})

    return Response(stream(), content_type="text/event-stream; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- ui

PAGE = r"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ถาม-ตอบจากเอกสาร PDF</title>
<style>
  :root{
    --paper:#f7f6f3; --ink:#1b1b1a; --muted:#6f6d67;
    --rule:#d8d5cd; --field:#ffffff; --mark:#1f5f4a;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);
       font-family:"IBM Plex Sans Thai","Noto Sans Thai",-apple-system,Segoe UI,sans-serif;
       font-size:15px;line-height:1.65}
  .wrap{display:grid;grid-template-columns:300px 1fr;min-height:100vh}
  aside{border-right:1px solid var(--rule);padding:24px 20px;background:#fbfaf8}
  main{display:flex;flex-direction:column;max-height:100vh}
  h1{font-size:17px;margin:0 0 4px;letter-spacing:-.01em}
  .sub{color:var(--muted);font-size:12.5px;margin-bottom:6px}
  .warn{color:#a33;font-size:12.5px;margin-bottom:16px}
  .drop{border:1.5px dashed var(--rule);border-radius:3px;padding:20px 14px;
        text-align:center;cursor:pointer;font-size:13.5px;color:var(--muted);
        transition:border-color .15s,background .15s;margin-top:16px}
  .drop:hover,.drop.on{border-color:var(--mark);background:#fff;color:var(--ink)}
  .doc{border-top:1px solid var(--rule);padding:11px 0;font-size:13px;
       display:flex;justify-content:space-between;gap:8px;align-items:flex-start}
  .doc b{font-weight:600;word-break:break-all;display:block}
  .doc span{color:var(--muted);font-size:12px}
  .doc button{border:0;background:none;color:var(--muted);cursor:pointer;
              font-size:16px;line-height:1;padding:2px 4px}
  .doc button:hover{color:#a33}
  #log{flex:1;overflow-y:auto;padding:34px 40px}
  .turn{max-width:66ch;margin:0 auto 30px}
  .q{font-weight:600;border-left:2px solid var(--mark);padding-left:14px}
  .a{margin-top:14px}
  .a p{margin:0 0 10px}
  .a h3{font-size:15px;font-weight:600;margin:20px 0 8px}
  .a ul{margin:0 0 10px;padding-left:22px}
  .a li{margin:3px 0}
  .a code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;
          background:#ece9e1;padding:1px 5px;border-radius:3px}
  .a pre{background:#f2efe8;border:1px solid var(--rule);border-radius:3px;
         padding:12px 14px;overflow-x:auto;margin:0 0 12px}
  .a pre code{background:none;padding:0;font-size:13px;line-height:1.5}
  .cite{color:var(--mark);font-weight:600}
  .src{margin-top:16px;font-size:12.5px}
  .src details{margin:5px 0;color:var(--muted)}
  .src summary{cursor:pointer}
  .src p{margin:6px 0 10px;padding-left:14px;border-left:1px solid var(--rule);
         color:var(--ink);opacity:.8}
  footer{border-top:1px solid var(--rule);padding:16px 40px;background:#fbfaf8}
  .bar{max-width:66ch;margin:0 auto;display:flex;gap:10px}
  textarea{flex:1;resize:none;border:1px solid var(--rule);border-radius:3px;
           padding:11px 13px;font:inherit;background:var(--field);color:inherit}
  textarea:focus{outline:2px solid var(--mark);outline-offset:-1px;border-color:transparent}
  button.send{border:0;background:var(--mark);color:#fff;border-radius:3px;
              padding:0 22px;font:inherit;font-weight:600;cursor:pointer}
  button.send:disabled{opacity:.45;cursor:default}
  .note{color:var(--muted);font-size:12.5px;max-width:66ch;margin:60px auto}
  @media(max-width:820px){.wrap{grid-template-columns:1fr}aside{border-right:0;
    border-bottom:1px solid var(--rule)}#log,footer{padding:20px}}
</style>
</head>
<body>
<div class="wrap">
  <aside>
    <h1>ถาม-ตอบจากเอกสาร</h1>
    <div class="sub" id="cfg">กำลังโหลด…</div>
    <div class="warn" id="warn"></div>
    <div class="drop" id="drop">ลากไฟล์ PDF มาวาง<br>หรือคลิกเพื่อเลือกไฟล์</div>
    <input type="file" id="file" accept="application/pdf" hidden>
    <div id="docs"></div>
  </aside>
  <main>
    <div id="log">
      <p class="note">อัปโหลด PDF แล้วถามได้เลย ระบบจะค้นเฉพาะส่วนที่เกี่ยวข้อง
      แล้วส่งให้โมเดลตอบพร้อมเลขหน้าอ้างอิง</p>
    </div>
    <footer>
      <div class="bar">
        <textarea id="q" rows="2" placeholder="พิมพ์คำถาม แล้วกด Enter"></textarea>
        <button class="send" id="send">ถาม</button>
      </div>
    </footer>
  </main>
</div>
<script>
const $ = s => document.querySelector(s);
const log = $("#log"), drop = $("#drop");
let history = [], busy = false;

const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function md(src){
  const codes = [];
  let s = src.replace(/```[\w+-]*\n?([\s\S]*?)(?:```|$)/g, (m, body) => {
    codes.push(esc(body.replace(/\s+$/, "")));
    return "\u0000" + (codes.length - 1) + "\u0000";
  });
  const inline = t => t
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>');
  const out = []; let inList = false;
  const closeList = () => { if(inList){ out.push("</ul>"); inList = false; } };
  for(const ln of esc(s).split("\n")){
    const ph = ln.match(/^\u0000(\d+)\u0000$/);
    if(ph){ closeList(); out.push("<pre><code>" + codes[+ph[1]] + "</code></pre>"); continue; }
    const h = ln.match(/^#{1,6}\s+(.+)$/);
    if(h){ closeList(); out.push("<h3>" + inline(h[1]) + "</h3>"); continue; }
    const li = ln.match(/^\s*(?:[-*\u2022]|\d+[.)])\s+(.+)$/);
    if(li){ if(!inList){ out.push("<ul>"); inList = true; }
            out.push("<li>" + inline(li[1]) + "</li>"); continue; }
    closeList();
    if(ln.trim()) out.push("<p>" + inline(ln) + "</p>");
  }
  closeList();
  return out.join("");
}
const DROP_LABEL = "ลากไฟล์ PDF มาวาง<br>หรือคลิกเพื่อเลือกไฟล์";

async function refresh(){
  const d = await (await fetch("/api/status")).json();
  $("#cfg").textContent = `${d.documents.length} ไฟล์ · ${d.chunks} chunk
    · ${d.chat_model} · ${d.embed_model}`;
  $("#warn").textContent = d.has_key ? "" : "ยังไม่ได้ตั้ง OPENROUTER_API_KEY";
  $("#docs").innerHTML = d.documents.map(x => `
    <div class="doc">
      <div><b>${esc(x.filename)}</b><span>${x.n_pages} หน้า · ${x.n_chunks} chunk</span></div>
      <button data-id="${x.id}" title="ลบไฟล์นี้">×</button>
    </div>`).join("");
  $("#docs").querySelectorAll("button").forEach(b => b.onclick = async () => {
    await fetch("/api/documents/" + b.dataset.id, {method:"DELETE"});
    refresh();
  });
}

drop.onclick = () => $("#file").click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add("on"); };
drop.ondragleave = () => drop.classList.remove("on");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("on");
                     if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); };
$("#file").onchange = e => e.target.files[0] && upload(e.target.files[0]);

async function upload(file){
  drop.textContent = "กำลังอ่านและสร้าง index…";
  const fd = new FormData(); fd.append("file", file);
  try{
    const r = await fetch("/api/upload", {method:"POST", body:fd});
    const j = await r.json();
    drop.textContent = r.ok ? `เพิ่ม ${j.filename} แล้ว (${j.seconds} วิ)` : j.error;
  }catch(e){ drop.textContent = "อัปโหลดไม่สำเร็จ: " + e.message; }
  setTimeout(() => drop.innerHTML = DROP_LABEL, 6000);
  refresh();
}

async function ask(){
  const q = $("#q").value.trim();
  if(!q || busy) return;
  busy = true; $("#send").disabled = true; $("#q").value = "";
  if(log.querySelector(".note")) log.innerHTML = "";

  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `<div class="q">${esc(q)}</div><div class="a">…</div><div class="src"></div>`;
  log.appendChild(turn); log.scrollTop = log.scrollHeight;
  const ansEl = turn.querySelector(".a"), srcEl = turn.querySelector(".src");
  let answer = "";

  const res = await fetch("/api/ask", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({question:q, history})
  });
  if(!res.ok){
    ansEl.textContent = (await res.json()).error;
    busy = false; $("#send").disabled = false; return;
  }

  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = "";
  while(true){
    const {value, done} = await reader.read();
    if(done) break;
    buf += dec.decode(value, {stream:true});
    const parts = buf.split("\n\n"); buf = parts.pop();
    for(const p of parts){
      if(!p.startsWith("data: ")) continue;
      const ev = JSON.parse(p.slice(6));
      if(ev.type === "sources"){
        srcEl.innerHTML = ev.sources.map(s => `
          <details><summary>[${s.n}] ${esc(s.filename)} — หน้า ${s.page_start}${s.page_start!==s.page_end?"–"+s.page_end:""} · ${s.score}</summary>
          <p>${esc(s.preview)}…</p></details>`).join("");
      } else if(ev.type === "delta"){
        answer += ev.text; ansEl.innerHTML = md(answer);
        log.scrollTop = log.scrollHeight;
      } else if(ev.type === "error"){
        answer += "\n\n" + ev.text; ansEl.innerHTML = md(answer);
      }
    }
  }
  history.push({role:"user", content:q}, {role:"assistant", content:answer});
  busy = false; $("#send").disabled = false; $("#q").focus();
}

$("#send").onclick = ask;
$("#q").onkeydown = e => { if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); ask(); } };
refresh();
</script>
</body>
</html>"""


if __name__ == "__main__":
    if "--list-embed-models" in sys.argv:
        for m in list_embed_models():
            print(f'{m["id"]:<50} context={m.get("context_length", "?")}')
        sys.exit(0)

    print(f"env file: {ENV_FILE or '(ไม่พบ .env - ใช้ค่าจาก shell)'}")
    print(f"key: {mask(API_KEY)}")
    print(f"DB={DB_PATH}  embed={EMBED_BACKEND}:{EMBED_MODEL}  chat={CHAT_MODEL}")
    print(f"chunks in index: {len(store.meta)}  (index model: {store.index_model()})")
    if not API_KEY:
        print("!! ยังไม่ได้ตั้ง OPENROUTER_API_KEY - สร้างไฟล์ .env ก่อน")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
