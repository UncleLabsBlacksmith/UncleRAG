# -*- coding: utf-8 -*-
"""
==============================================================================
  RAG Chat  ·  ถาม-ตอบจากไฟล์ PDF   (ไฟล์เดียวจบ: Python + HTML)
  จากสไลด์ EP.3 — RAG for Beginners  ·  uncle-engineer.com
==============================================================================

  Pipeline ตามสไลด์:
     PDF → Chunk → Embedding → Vector DB        (ตอนอัปโหลด)
     Question → Search → Context → LLM          (ตอนถาม)

  วิธีใช้:
     pip install flask pypdf numpy requests
     python app.py
     เปิด http://127.0.0.1:5000

  อยากได้ Semantic Search เต็มรูปแบบ (แนะนำ):
     pip install sentence-transformers
     -> ครั้งแรกจะโหลดโมเดลราว 500MB แล้วใช้ออฟไลน์ได้ตลอด

  ถ้าไม่ลง ก็ยังใช้ได้ทันที ระบบจะสลับไปใช้ตัวค้นหาแบบ built-in ให้เอง
==============================================================================
"""

import os
import re
import io
import json
import time
import pickle
import hashlib
import threading
import traceback

import numpy as np
import requests
from flask import Flask, request, jsonify, Response, render_template_string, stream_with_context

# ============================================================================
#  1) ตั้งค่า  —  แก้ตรงนี้ที่เดียวพอ
# ============================================================================

# ---- การตัด Chunk (สไลด์ 7-9) -------------------------------------------
CHUNK_SIZE    = 900      # ความยาว 1 chunk (ตัวอักษร) ~ 300-400 tokens ไทย
CHUNK_OVERLAP = 150      # ให้ chunk เหลื่อมกัน กันข้อมูลตรงรอยต่อหาย
TOP_K         = 4        # หยิบ chunk ใกล้เคียงที่สุดกี่ชิ้นส่งให้ LLM

# ---- Embedding: "auto" | "st" | "ollama" | "simple" ----------------------
EMBED_PROVIDER = "auto"
ST_MODEL       = "intfloat/multilingual-e5-small"   # โมเดล sentence-transformers (รองรับไทย)
OLLAMA_EMBED   = "bge-m3"                           # ถ้าใช้ Ollama ทำ embedding

# ---- LLM: "auto" | "anthropic" | "openai" | "ollama" | "none" ------------
LLM_PROVIDER   = "auto"
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_LLM     = "qwen2.5:7b"        # ถ้าไม่มีรุ่นนี้ ระบบจะเลือกรุ่นที่ติดตั้งไว้ให้เอง
OPENAI_BASE    = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

# ---- อื่น ๆ --------------------------------------------------------------
STORE_FILE = "rag_store.pkl"    # เก็บคลังความรู้ไว้ ปิดโปรแกรมแล้วเปิดใหม่ยังอยู่
MAX_MB     = 50
HOST, PORT = "127.0.0.1", 5001

SYSTEM_PROMPT = """คุณคือผู้ช่วยตอบคำถามจากเอกสารที่ผู้ใช้อัปโหลด

กฎการตอบ:
1. ตอบจากข้อมูลใน CONTEXT เท่านั้น ห้ามเดาหรือเติมความรู้ภายนอก
2. ถ้าข้อมูลใน CONTEXT ไม่พอ ให้บอกตรง ๆ ว่าไม่พบในเอกสาร แล้วแนะนำว่าควรถามใหม่ว่าอย่างไร
3. อ้างอิงหน้าเอกสารท้ายประโยคที่ใช้ข้อมูลนั้น เช่น (หน้า 42)
4. ถ้าเป็นขั้นตอน ให้ตอบเป็นข้อ ๆ สั้น กระชับ
5. ตอบเป็นภาษาไทย เว้นแต่ผู้ใช้ถามเป็นภาษาอื่น"""


# ============================================================================
#  2) อ่าน PDF  (สไลด์ 5-6)
# ============================================================================

def read_pdf(data: bytes):
    """คืนค่า [(เลขหน้า, ข้อความ), ...]"""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        if txt:
            pages.append((i, txt))
    return pages


# ============================================================================
#  3) ตัด Chunk  (สไลด์ 7-9)
# ============================================================================

def chunk_pages(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    รวมข้อความทุกหน้าแล้วตัดเป็นชิ้นยาว ~size ตัวอักษร ให้เหลื่อมกัน overlap
    พร้อมจำไว้ว่าชิ้นนั้นมาจากหน้าไหนถึงหน้าไหน
    """
    # ทำเป็นสายอักขระเดียว + ตารางบอกว่าตำแหน่งไหนอยู่หน้าอะไร
    buf, marks = [], []
    pos = 0
    for pno, txt in pages:
        buf.append(txt)
        marks.append((pos, pos + len(txt), pno))
        pos += len(txt) + 2
    full = "\n\n".join(buf)

    def page_of(idx):
        for s, e, p in marks:
            if s <= idx < e:
                return p
        return marks[-1][2] if marks else 1

    chunks, start = [], 0
    n = len(full)
    while start < n:
        end = min(start + size, n)
        if end < n:  # พยายามตัดที่ย่อหน้า/ประโยค ไม่ตัดกลางคำ
            window = full[start:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "), window.rfind("। "))
            if cut > size * 0.5:
                end = start + cut + 1
        text = full[start:end].strip()
        if len(text) > 30:
            chunks.append({
                "text": text,
                "page_from": page_of(start),
                "page_to": page_of(max(start, end - 1)),
            })
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# ============================================================================
#  4) Embedding  —  แปลงข้อความเป็น "พิกัดความหมาย"  (สไลด์ 10-12)
# ============================================================================

_st_model = None
_st_lock = threading.Lock()
SIMPLE_DIM = 2048


def _simple_embed(texts):
    """
    ตัวสำรอง: ไม่ต้องลงอะไรเพิ่ม ใช้ character 3-gram + hashing
    ใช้ได้กับภาษาไทยโดยไม่ต้องตัดคำ แม่นน้อยกว่าโมเดลจริงแต่ใช้เรียนรู้ flow ได้
    """
    out = np.zeros((len(texts), SIMPLE_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        s = re.sub(r"\s+", " ", t.lower())
        feats = re.findall(r"[a-z0-9]+", s)
        feats += [s[j:j + 3] for j in range(max(0, len(s) - 2))]
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode("utf-8")).digest()[:4], "little") % SIMPLE_DIM
            out[i, h] += 1.0
        np.log1p(out[i], out=out[i])
        nrm = np.linalg.norm(out[i])
        if nrm:
            out[i] /= nrm
    return out


def _st_embed(texts, is_query=False):
    global _st_model
    with _st_lock:
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer(ST_MODEL)
    # โมเดลตระกูล e5 ต้องใส่ prefix ให้ถูกฝั่ง
    if "e5" in ST_MODEL.lower():
        pre = "query: " if is_query else "passage: "
        texts = [pre + t for t in texts]
    v = _st_model.encode(texts, normalize_embeddings=True, batch_size=16)
    return np.asarray(v, dtype=np.float32)


def _no_model_error():
    return RuntimeError(
        f"Ollama ยังไม่มีโมเดล '{OLLAMA_EMBED}' — เปิด terminal แล้วสั่ง:  ollama pull {OLLAMA_EMBED}\n"
        f"(หรือจะข้าม Ollama ไปใช้  pip install sentence-transformers  แทนก็ได้)")


def _ollama_embed_old(texts):
    """Ollama รุ่นเก่า: /api/embeddings รับได้ทีละข้อความ"""
    out = []
    for t in texts:
        r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                          json={"model": OLLAMA_EMBED, "prompt": t}, timeout=300)
        if r.status_code == 404:
            raise _no_model_error()
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out


def _ollama_embed(texts):
    # Ollama รุ่นใหม่: /api/embed ส่งได้ทีเดียวหลายข้อความ (เร็วกว่ามาก)
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": OLLAMA_EMBED, "input": texts}, timeout=300)
    if r.status_code == 404:
        try:
            err = r.json().get("error", "")            # ตอบเป็น JSON = รู้จัก endpoint แต่ไม่มีโมเดล
        except Exception:
            err = ""
        if err:
            raise _no_model_error()
        vecs = _ollama_embed_old(texts)                # ตอบเป็น text = Ollama รุ่นเก่า ไม่มี endpoint นี้
    else:
        r.raise_for_status()
        vecs = r.json()["embeddings"]
    v = np.asarray(vecs, dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v


def embed(texts, is_query=False):
    if EMBED_ACTIVE == "st":
        return _st_embed(texts, is_query)
    if EMBED_ACTIVE == "ollama":
        return _ollama_embed(texts)
    return _simple_embed(texts)


# ============================================================================
#  5) Vector DB  —  เก็บพิกัด + ค้นเพื่อนบ้านที่ใกล้ที่สุด  (สไลด์ 13-14)
# ============================================================================

class VectorStore:
    def __init__(self):
        self.chunks = []                                   # [{text, page_from, page_to, doc}]
        self.vectors = np.zeros((0, 1), dtype=np.float32)
        self.docs = {}                                     # ชื่อไฟล์ -> สถิติ
        self.lock = threading.Lock()

    def add(self, doc_name, chunks, vectors):
        with self.lock:
            if self.vectors.shape[0] == 0:
                self.vectors = vectors
            else:
                if self.vectors.shape[1] != vectors.shape[1]:
                    self.reset()          # เปลี่ยนโมเดล embedding แล้ว มิติไม่ตรงกัน
                    self.vectors = vectors
                else:
                    self.vectors = np.vstack([self.vectors, vectors])
            for c in chunks:
                c["doc"] = doc_name
            self.chunks.extend(chunks)
            self.docs[doc_name] = {
                "chunks": len(chunks),
                "pages": max((c["page_to"] for c in chunks), default=0),
                "added": time.time(),
            }
        self.save()

    def search(self, qvec, k=TOP_K):
        with self.lock:
            if not self.chunks:
                return []
            scores = self.vectors @ qvec.ravel()
            idx = np.argsort(-scores)[:k]
            return [{**self.chunks[i], "score": float(scores[i]), "id": int(i)} for i in idx]

    def remove(self, doc_name):
        with self.lock:
            keep = [i for i, c in enumerate(self.chunks) if c.get("doc") != doc_name]
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 1), dtype=np.float32)
            self.docs.pop(doc_name, None)
        self.save()

    def reset(self):
        self.chunks, self.docs = [], {}
        self.vectors = np.zeros((0, 1), dtype=np.float32)

    def stats(self):
        return {
            "docs": [{"name": k, **v} for k, v in sorted(self.docs.items(), key=lambda x: -x[1]["added"])],
            "chunks": len(self.chunks),
            "dim": int(self.vectors.shape[1]) if self.vectors.shape[0] else 0,
        }

    def save(self):
        try:
            with open(STORE_FILE, "wb") as f:
                pickle.dump({"chunks": self.chunks, "vectors": self.vectors,
                             "docs": self.docs, "embed": EMBED_ACTIVE}, f)
        except Exception:
            pass

    def load(self):
        if not os.path.exists(STORE_FILE):
            return
        try:
            with open(STORE_FILE, "rb") as f:
                d = pickle.load(f)
            if d.get("embed") != EMBED_ACTIVE:   # คนละโมเดล = พิกัดคนละแผนที่ ใช้ร่วมกันไม่ได้
                return
            self.chunks, self.vectors, self.docs = d["chunks"], d["vectors"], d["docs"]
        except Exception:
            pass


STORE = VectorStore()


# ============================================================================
#  6) LLM  —  อ่าน Context แล้วเรียบเรียงคำตอบ  (สไลด์ 18-20)
# ============================================================================

def build_prompt(context_chunks, question):
    parts = []
    for c in context_chunks:
        pg = f"หน้า {c['page_from']}" if c["page_from"] == c["page_to"] else f"หน้า {c['page_from']}-{c['page_to']}"
        parts.append(f"[{c['doc']} · {pg}]\n{c['text']}")
    return f"CONTEXT:\n\n" + "\n\n---\n\n".join(parts) + f"\n\n\nQUESTION:\n{question}"


def stream_answer(prompt):
    """yield ข้อความทีละชิ้นจาก LLM"""
    if LLM_ACTIVE == "none":
        yield ("ยังไม่ได้ต่อ LLM จึงแสดงเฉพาะเนื้อหาที่ค้นเจอด้านล่างนี้แทน\n\n"
               "วิธีต่อ: ติดตั้ง Ollama แล้วสั่ง `ollama pull qwen2.5:7b` "
               "หรือใส่ ANTHROPIC_API_KEY / OPENAI_API_KEY เป็น environment variable แล้วรันใหม่")
        return

    if LLM_ACTIVE == "ollama":
        r = requests.post(f"{OLLAMA_URL}/api/chat", stream=True, timeout=600, json={
            "model": OLLAMA_LLM,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
            "stream": True, "options": {"temperature": 0.2},
        })
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            if d.get("message", {}).get("content"):
                yield d["message"]["content"]
            if d.get("done"):
                break

    elif LLM_ACTIVE == "openai":
        r = requests.post(f"{OPENAI_BASE}/chat/completions", stream=True, timeout=600,
                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                          json={"model": OPENAI_MODEL, "stream": True, "temperature": 0.2,
                                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                             {"role": "user", "content": prompt}]})
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            body = line[6:]
            if body == b"[DONE]":
                break
            try:
                delta = json.loads(body)["choices"][0]["delta"].get("content")
            except Exception:
                delta = None
            if delta:
                yield delta

    elif LLM_ACTIVE == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", stream=True, timeout=600,
                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                   "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": ANTHROPIC_MODEL, "max_tokens": 2000, "stream": True,
                                "system": SYSTEM_PROMPT,
                                "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            if d.get("type") == "content_block_delta" and d["delta"].get("text"):
                yield d["delta"]["text"]


# ============================================================================
#  7) เลือก provider อัตโนมัติตอนเริ่มโปรแกรม
# ============================================================================

WARNINGS = []


def _ollama_models():
    """คืนรายชื่อโมเดลที่ติดตั้งใน Ollama — ถ้าต่อไม่ได้คืน None"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def _has(names, want):
    base = want.split(":")[0]
    return any(n == want or n.split(":")[0] == base for n in names)


def pick_providers():
    global EMBED_ACTIVE, LLM_ACTIVE, OLLAMA_LLM

    models = _ollama_models()          # None = ไม่ได้เปิด Ollama ไว้
    alive = models is not None

    # -- Embedding --
    e = EMBED_PROVIDER
    if e == "auto":
        try:
            import sentence_transformers  # noqa: F401
            e = "st"
        except ImportError:
            e = "ollama" if alive else "simple"

    # ถ้าจะใช้ Ollama ต้องเช็คก่อนว่ามีโมเดล embedding จริงไหม ไม่งั้นจะเจอ 404 ตอนอัปโหลด
    if e == "ollama":
        if not alive:
            WARNINGS.append("ต่อ Ollama ไม่ได้ — สลับไปใช้ตัวค้นหาแบบพื้นฐานแทน")
            e = "simple"
        elif not _has(models, OLLAMA_EMBED):
            WARNINGS.append(
                f"Ollama ยังไม่มีโมเดล embedding '{OLLAMA_EMBED}' — สลับไปใช้ตัวค้นหาแบบพื้นฐานแทน\n"
                f"           อยากได้ semantic search ให้สั่ง:  ollama pull {OLLAMA_EMBED}\n"
                f"           หรือ:  pip install sentence-transformers")
            e = "simple"
    EMBED_ACTIVE = e

    # -- LLM --
    l = LLM_PROVIDER
    if l == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            l = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            l = "openai"
        elif alive:
            l = "ollama"
        else:
            l = "none"
    LLM_ACTIVE = l

    # ถ้าใช้ Ollama แต่ไม่มีรุ่นที่ตั้งไว้ ให้หยิบรุ่นที่มีอยู่แทน
    if LLM_ACTIVE == "ollama":
        names = models or []
        if not _has(names, OLLAMA_LLM):
            chat = [n for n in names if "embed" not in n and "bge" not in n]
            if chat:
                WARNINGS.append(f"ไม่พบโมเดล '{OLLAMA_LLM}' ใน Ollama — ใช้ '{chat[0]}' ที่มีอยู่แทน")
                OLLAMA_LLM = chat[0]
            else:
                WARNINGS.append(f"Ollama ยังไม่มีโมเดลสำหรับตอบคำถาม — สั่ง:  ollama pull {OLLAMA_LLM}")
                LLM_ACTIVE = "none"


EMBED_ACTIVE = "simple"
LLM_ACTIVE = "none"

EMBED_LABEL = {
    "st": ("Semantic", f"sentence-transformers · {ST_MODEL.split('/')[-1]}"),
    "ollama": ("Semantic", f"Ollama · {OLLAMA_EMBED}"),
    "simple": ("พื้นฐาน", "built-in char n-gram (ลง sentence-transformers เพื่อความแม่นยำ)"),
}


def llm_label():
    return {"anthropic": ANTHROPIC_MODEL, "openai": OPENAI_MODEL,
            "ollama": OLLAMA_LLM, "none": "ยังไม่ได้ต่อ"}[LLM_ACTIVE]


# ============================================================================
#  8) Flask
# ============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def sse(event, **data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def index():
    return render_template_string(PAGE, cfg={
        "embed_mode": EMBED_LABEL[EMBED_ACTIVE][0],
        "embed_detail": EMBED_LABEL[EMBED_ACTIVE][1],
        "llm": llm_label(),
        "llm_ready": LLM_ACTIVE != "none",
        "top_k": TOP_K,
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
    })


@app.get("/api/status")
def api_status():
    return jsonify(STORE.stats())


@app.post("/api/upload")
def api_upload():
    """อัปโหลด PDF แล้วส่งความคืบหน้ากลับเป็น stream: อ่าน → ตัด → embed → เก็บ"""
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "รับเฉพาะไฟล์ .pdf"}), 400
    name, data = os.path.basename(f.filename), f.read()
    size = int(request.form.get("chunk_size", CHUNK_SIZE))
    lap = int(request.form.get("overlap", CHUNK_OVERLAP))

    def gen():
        try:
            yield sse("step", step="read", msg="กำลังอ่านไฟล์ PDF")
            pages = read_pdf(data)
            if not pages:
                yield sse("error", msg="อ่านข้อความจาก PDF นี้ไม่ได้ — น่าจะเป็นไฟล์สแกนเป็นรูป ต้องทำ OCR ก่อน")
                return
            yield sse("step", step="read", msg=f"อ่านได้ {len(pages)} หน้า", done=True)

            yield sse("step", step="chunk", msg="กำลังตัดเป็น Chunk")
            chunks = chunk_pages(pages, size, lap)
            yield sse("step", step="chunk", msg=f"ได้ {len(chunks)} chunk", done=True)

            yield sse("step", step="embed", msg="กำลังสร้าง Embedding (ครั้งแรกอาจนานหน่อย)")
            vecs, B = [], 32
            for i in range(0, len(chunks), B):
                vecs.append(embed([c["text"] for c in chunks[i:i + B]]))
                yield sse("step", step="embed",
                          msg=f"สร้าง Embedding {min(i + B, len(chunks))}/{len(chunks)}")
            vectors = np.vstack(vecs)
            yield sse("step", step="embed", msg=f"สร้างครบ {len(chunks)} เวกเตอร์", done=True)

            yield sse("step", step="store", msg="กำลังเก็บลง Vector DB")
            STORE.add(name, chunks, vectors)
            yield sse("step", step="store", msg="เก็บเรียบร้อย", done=True)
            yield sse("done", name=name, pages=len(pages), chunks=len(chunks), stats=STORE.stats())
        except Exception as e:
            traceback.print_exc()
            yield sse("error", msg=f"{type(e).__name__}: {e}")

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.post("/api/ask")
def api_ask():
    """ถาม → ค้น (ส่งผลค้นกลับก่อน) → ให้ LLM อ่านแล้วตอบทีละคำ"""
    body = request.get_json(force=True)
    q = (body.get("question") or "").strip()
    k = int(body.get("top_k") or TOP_K)
    if not q:
        return jsonify({"error": "กรุณาพิมพ์คำถาม"}), 400

    def gen():
        try:
            if not STORE.chunks:
                yield sse("error", msg="ยังไม่มีเอกสารในคลัง — อัปโหลด PDF ก่อนนะครับ")
                return

            t0 = time.time()
            qvec = embed([q], is_query=True)
            hits = STORE.search(qvec, k)
            yield sse("retrieval", hits=[{
                "doc": h["doc"], "score": round(h["score"], 3),
                "page_from": h["page_from"], "page_to": h["page_to"], "text": h["text"],
            } for h in hits], ms=int((time.time() - t0) * 1000))

            prompt = build_prompt(hits, q)
            yield sse("generating")
            for piece in stream_answer(prompt):
                yield sse("token", t=piece)
            yield sse("end")
        except Exception as e:
            traceback.print_exc()
            yield sse("error", msg=f"{type(e).__name__}: {e}")

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.errorhandler(413)
def too_big(_):
    return jsonify({"error": f"ไฟล์ใหญ่เกิน {MAX_MB} MB"}), 413


@app.post("/api/remove")
def api_remove():
    STORE.remove(request.get_json(force=True).get("name", ""))
    return jsonify(STORE.stats())


@app.post("/api/reset")
def api_reset():
    STORE.reset()
    STORE.save()
    return jsonify(STORE.stats())


# ============================================================================
#  9) หน้าเว็บ  (HTML + CSS + JS อยู่ในไฟล์เดียวกัน)
# ============================================================================

PAGE = r"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>RAG Chat · ถามเอกสารของคุณ</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#10141c; --panel:#171c27; --raise:#1e2432; --line:#2a3243;
  --text:#e7eaf1; --dim:#98a2b8; --faint:#69738a;
  --brass:#e0a13c; --brass-soft:#3a2f1a;
  --ok:#5ec2a4; --warn:#e0725c;
  --r:14px;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0; background:var(--ink); color:var(--text);
  font-family:'Anuphan',-apple-system,'Segoe UI',sans-serif;
  font-size:16px; line-height:1.65; -webkit-font-smoothing:antialiased;
}
button,input,textarea{font-family:inherit;font-size:inherit;color:inherit}
button{cursor:pointer;border:none;background:none}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:#2c3446;border-radius:20px;border:3px solid var(--ink)}
:focus-visible{outline:2px solid var(--brass);outline-offset:2px;border-radius:6px}

/* ---------- โครงหน้า ---------- */
.app{display:grid;grid-template-columns:326px 1fr;height:100dvh}
aside{background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;overflow:hidden}
main{display:flex;flex-direction:column;min-width:0}

/* ---------- Sidebar ---------- */
.brand{padding:22px 22px 16px;border-bottom:1px solid var(--line)}
.brand h1{margin:0;font-size:20px;font-weight:600;letter-spacing:-.01em}
.brand p{margin:2px 0 0;font-size:13.5px;color:var(--faint)}
.side-scroll{overflow-y:auto;padding:18px 22px 22px;flex:1}
.sec{margin-bottom:26px}
.sec h2{margin:0 0 10px;font-size:13.5px;font-weight:600;color:var(--dim)}

#drop{
  border:1.5px dashed #35405a;border-radius:var(--r);padding:26px 16px;text-align:center;
  background:#141924;transition:border-color .15s,background .15s;cursor:pointer;
}
#drop:hover,#drop.over{border-color:var(--brass);background:#1a1f2b}
#drop .big{font-size:15px;font-weight:500}
#drop .sm{font-size:13px;color:var(--faint);margin-top:3px}

.pipe{margin-top:14px;display:none}
.pipe.on{display:block}
.pstep{display:flex;gap:10px;align-items:flex-start;padding:5px 0;font-size:13.5px;color:var(--faint)}
.pstep .dot{width:9px;height:9px;border-radius:50%;background:#333c52;margin-top:8px;flex:none;transition:.2s}
.pstep.run{color:var(--text)} .pstep.run .dot{background:var(--brass);box-shadow:0 0 0 4px var(--brass-soft)}
.pstep.fin{color:var(--dim)} .pstep.fin .dot{background:var(--ok)}

.doc{display:flex;gap:10px;align-items:center;padding:10px 12px;background:var(--raise);
     border:1px solid var(--line);border-radius:11px;margin-bottom:8px}
.doc .nm{flex:1;min-width:0;font-size:14px}
.doc .nm b{display:block;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.doc .nm span{font-size:12.5px;color:var(--faint)}
.doc button{color:var(--faint);font-size:19px;line-height:1;padding:0 3px}
.doc button:hover{color:var(--warn)}
.empty{font-size:13.5px;color:var(--faint);padding:12px 0}

.field{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:7px 0;font-size:14px}
.field input[type=number]{width:76px;background:var(--raise);border:1px solid var(--line);
     border-radius:8px;padding:5px 9px;text-align:right}
.field .val{color:var(--brass);font-variant-numeric:tabular-nums;min-width:20px;text-align:right}
input[type=range]{-webkit-appearance:none;flex:1;height:3px;background:#333c52;border-radius:3px}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;border-radius:50%;background:var(--brass);cursor:pointer}
input[type=range]::-moz-range-thumb{width:15px;height:15px;border:none;border-radius:50%;background:var(--brass);cursor:pointer}

.engine{font-size:13px;color:var(--faint);line-height:1.55}
.engine div{padding:4px 0;border-bottom:1px solid #212838}
.engine div:last-child{border:none}
.engine b{color:var(--dim);font-weight:500}
.link{color:var(--faint);font-size:13px;text-decoration:underline;text-underline-offset:3px}
.link:hover{color:var(--warn)}

/* ---------- แชท ---------- */
.stream{flex:1;overflow-y:auto;padding:34px 0 20px}
.wrap{max-width:760px;margin:0 auto;padding:0 26px}

.hero{max-width:760px;margin:auto;padding:6vh 26px}
.hero h2{font-size:30px;font-weight:600;margin:0 0 10px;letter-spacing:-.02em;line-height:1.35}
.hero p{color:var(--dim);margin:0 0 26px;max-width:52ch}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:30px}
.flow i{font-style:normal;padding:6px 13px;border:1px solid var(--line);border-radius:999px;
        font-size:13.5px;color:var(--dim);background:var(--panel)}
.flow u{text-decoration:none;color:#3f4a63}
.hero .tips{font-size:14px;color:var(--faint)}
.hero .tips b{display:block;color:var(--dim);font-weight:500;margin-bottom:6px}

.msg{margin-bottom:26px}
.me{display:flex;justify-content:flex-end}
.me p{margin:0;background:var(--raise);border:1px solid var(--line);
      padding:11px 16px;border-radius:16px 16px 4px 16px;max-width:80%;white-space:pre-wrap}
.bot .who{font-size:13px;color:var(--faint);margin-bottom:7px}
.bot .body{white-space:pre-wrap}
.bot .body b{color:#fff}
.cursor{display:inline-block;width:8px;height:17px;background:var(--brass);
        vertical-align:-3px;animation:bl .9s steps(2) infinite}
@keyframes bl{50%{opacity:0}}

.sources{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}
.sources>summary{cursor:pointer;font-size:13.5px;color:var(--dim);list-style:none;display:flex;gap:8px;align-items:center}
.sources>summary::-webkit-details-marker{display:none}
.sources>summary:before{content:'▸';color:var(--faint);transition:.15s}
.sources[open]>summary:before{transform:rotate(90deg)}
.src{margin-top:10px;background:var(--panel);border:1px solid var(--line);border-radius:11px;overflow:hidden}
.src .h{display:flex;gap:10px;align-items:center;padding:9px 13px;font-size:13px;color:var(--dim);background:#141924}
.src .h .pg{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.src .h .sc{font-variant-numeric:tabular-nums;color:var(--brass)}
.bar{height:3px;background:#232b3b}
.bar i{display:block;height:100%;background:var(--brass)}
.src .t{padding:11px 14px;font-size:14px;color:#cfd6e6;background:#12161f;
        max-height:112px;overflow:hidden;position:relative;line-height:1.6}
.src.open .t{max-height:none}
.src .t:after{content:'';position:absolute;left:0;right:0;bottom:0;height:34px;
        background:linear-gradient(transparent,#12161f)}
.src.open .t:after{display:none}

.err{background:#2a1c1a;border:1px solid #5a3229;color:#f0b3a5;
     padding:11px 15px;border-radius:11px;font-size:14.5px}

/* ---------- กล่องพิมพ์ ---------- */
.composer{border-top:1px solid var(--line);background:var(--ink);padding:16px 0 20px}
.box{max-width:760px;margin:0 auto;padding:0 26px;display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:var(--r);
   padding:11px 16px;resize:none;min-height:50px;max-height:180px;line-height:1.6}
textarea:focus{border-color:#3c4761;outline:none}
textarea::placeholder{color:var(--faint)}
#send{background:var(--brass);color:#171205;font-weight:600;border-radius:12px;padding:0 22px;height:50px;flex:none}
#send:hover:not(:disabled){background:#eeae47}
#send:disabled{background:#2a3243;color:var(--faint);cursor:not-allowed}
.hint{max-width:760px;margin:9px auto 0;padding:0 26px;font-size:12.5px;color:#4e586e}

@media(max-width:900px){
  .app{grid-template-columns:1fr;grid-template-rows:auto 1fr}
  aside{border-right:none;border-bottom:1px solid var(--line);max-height:44dvh}
  .brand{padding:14px 20px 12px} .brand h1{font-size:18px}
  .side-scroll{padding:14px 20px 20px}
  .wrap,.box,.hint{padding-left:18px;padding-right:18px}
  .hero{padding-top:24px}.hero h2{font-size:24px}
}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="app">

<aside>
  <div class="brand">
    <h1>ถามเอกสารของคุณ</h1>
    <p>RAG · ค้นให้เจอ แล้วค่อยให้ AI ตอบ</p>
  </div>

  <div class="side-scroll">
    <div class="sec">
      <h2>คลังความรู้</h2>
      <div id="drop" tabindex="0" role="button">
        <div class="big">ลากไฟล์ PDF มาวาง</div>
        <div class="sm">หรือคลิกเพื่อเลือกไฟล์</div>
      </div>
      <input type="file" id="file" accept=".pdf" hidden>
      <div class="pipe" id="pipe"></div>
      <div id="docs" style="margin-top:14px"></div>
    </div>

    <div class="sec">
      <h2>การค้นหา</h2>
      <div class="field">
        <label for="k">หยิบ Chunk มาตอบ</label>
        <input type="range" id="k" min="1" max="8" value="{{ cfg.top_k }}">
        <span class="val" id="kv">{{ cfg.top_k }}</span>
      </div>
      <div class="field">
        <label for="cs">ขนาด Chunk</label>
        <input type="number" id="cs" value="{{ cfg.chunk_size }}" min="200" max="3000" step="50">
      </div>
      <div class="field">
        <label for="ov">Overlap</label>
        <input type="number" id="ov" value="{{ cfg.overlap }}" min="0" max="600" step="25">
      </div>
    </div>

    <div class="sec">
      <h2>เครื่องยนต์</h2>
      <div class="engine">
        <div><b>ค้นหา</b> · {{ cfg.embed_mode }}<br>{{ cfg.embed_detail }}</div>
        <div><b>ผู้ตอบ</b> · {{ cfg.llm }}</div>
      </div>
      <div style="margin-top:12px"><button class="link" id="reset">ล้างคลังความรู้ทั้งหมด</button></div>
    </div>
  </div>
</aside>

<main>
  <div class="stream" id="stream">
    <div class="hero" id="hero">
      <h2>อัปโหลดคู่มือ แล้วถามได้เลย</h2>
      <p>ระบบจะตัดเอกสารเป็นชิ้นเล็ก ๆ แปลงเป็นพิกัดความหมาย แล้วหยิบเฉพาะส่วนที่ตรงคำถามส่งให้ AI อ่านก่อนตอบ — ทุกคำตอบเปิดดูได้ว่าอ้างจากหน้าไหน</p>
      <div class="flow">
        <i>PDF</i><u>→</u><i>Chunk</i><u>→</u><i>Embedding</i><u>→</u><i>Vector DB</i>
        <u>→</u><i>ค้นหา</i><u>→</u><i>LLM ตอบ</i>
      </div>
      <div class="tips">
        <b>ลองถามแบบนี้</b>
        ถามเป็นประโยคเต็ม เช่น “Nozzle อุดตันต้องทำอย่างไร” จะได้ผลดีกว่าคำเดี่ยว ๆ<br>
        ถ้าคำตอบไม่ตรง ลองเพิ่มจำนวน Chunk ที่หยิบมาตอบทางซ้าย
        {% if cfg.embed_mode != 'Semantic' %}
        <div style="margin-top:14px;padding:12px 15px;border:1px solid var(--line);border-radius:11px;background:var(--panel);color:var(--dim)">
          ตอนนี้ใช้ตัวค้นหาแบบพื้นฐาน ซึ่งจับได้แค่คำที่เขียนคล้ายกัน
          ติดตั้ง <b style="color:var(--text);font-weight:500">pip install sentence-transformers</b>
          แล้วรันใหม่ เพื่อให้ค้นตามความหมายได้จริง เช่น ถาม “หัวพิมพ์ตัน” แล้วเจอ “Nozzle อุดตัน”
        </div>
        {% endif %}
      </div>
    </div>
  </div>

  <div class="composer">
    <div class="box">
      <textarea id="q" rows="1" placeholder="พิมพ์คำถามเกี่ยวกับเอกสาร…"></textarea>
      <button id="send">ถาม</button>
    </div>
    <div class="hint">Enter เพื่อส่ง · Shift+Enter ขึ้นบรรทัดใหม่ · AI ตอบจากเอกสารที่อัปโหลดเท่านั้น</div>
  </div>
</main>
</div>

<script>
const CFG = {{ cfg|tojson }};
const $ = s => document.querySelector(s);
const stream = $('#stream'), pipe = $('#pipe');
let busy = false;

const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const md  = s => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
const btm = () => stream.scrollTop = stream.scrollHeight;

/* ---- ตัวอ่าน stream แบบ SSE ---- */
async function sse(url, opts, on){
  const res = await fetch(url, opts);
  if(!res.ok){
    let m = 'เกิดข้อผิดพลาด (' + res.status + ')';
    if(res.status === 413) m = 'ไฟล์ใหญ่เกินไป';
    try{ const j = await res.json(); m = j.error || m; }catch(_){}
    on('error', {msg:m}); return;
  }
  const rd = res.body.getReader(), dec = new TextDecoder();
  let buf = '';
  for(;;){
    const {done, value} = await rd.read();
    if(done) break;
    buf += dec.decode(value, {stream:true});
    const parts = buf.split('\n\n'); buf = parts.pop();
    for(const p of parts){
      const ev = /event: (.+)/.exec(p), dt = /data: ([\s\S]+)/.exec(p);
      if(ev && dt) on(ev[1], JSON.parse(dt[1]));
    }
  }
}

/* ================= อัปโหลด ================= */
const STEPS = [['read','อ่าน PDF'],['chunk','ตัดเป็น Chunk'],['embed','สร้าง Embedding'],['store','เก็บลง Vector DB']];

function drawPipe(state){
  pipe.className = 'pipe on';
  pipe.innerHTML = STEPS.map(([k,label]) => {
    const s = state[k] || {};
    return `<div class="pstep ${s.cls||''}"><span class="dot"></span><span>${s.msg||label}</span></div>`;
  }).join('');
}

async function upload(file){
  if(busy) return; busy = true; $('#send').disabled = true;
  const fd = new FormData();
  fd.append('file', file);
  fd.append('chunk_size', $('#cs').value);
  fd.append('overlap', $('#ov').value);
  const state = {};
  drawPipe(state);
  await sse('/api/upload', {method:'POST', body:fd}, (ev,d) => {
    if(ev === 'step'){
      state[d.step] = {msg:d.msg, cls: d.done ? 'fin' : 'run'};
      drawPipe(state);
    } else if(ev === 'done'){
      renderDocs(d.stats);
      setTimeout(()=>{pipe.className='pipe'}, 2200);
      $('#hero')?.remove();
      addNote(`เพิ่ม <b>${esc(d.name)}</b> แล้ว — ${d.pages} หน้า แบ่งเป็น ${d.chunks} chunk ถามได้เลยครับ`);
    } else if(ev === 'error'){
      pipe.className = 'pipe';
      addErr(d.msg || d.error);
    }
  }).catch(e => addErr(String(e)));
  busy = false; $('#send').disabled = false;
}

/* ================= ถาม ================= */
async function ask(){
  const q = $('#q').value.trim();
  if(!q || busy) return;
  busy = true; $('#send').disabled = true;
  $('#q').value = ''; $('#q').style.height = 'auto';
  $('#hero')?.remove();

  const me = document.createElement('div');
  me.className = 'msg wrap me';
  me.innerHTML = `<p>${esc(q)}</p>`;
  stream.append(me);

  const bot = document.createElement('div');
  bot.className = 'msg wrap bot';
  bot.innerHTML = `<div class="who">กำลังค้นหาในเอกสาร…</div><div class="body"></div>`;
  stream.append(bot); btm();

  const body = bot.querySelector('.body'), who = bot.querySelector('.who');
  let text = '', hits = null;

  await sse('/api/ask', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({question:q, top_k:+$('#k').value})
  }, (ev,d) => {
    if(ev === 'retrieval'){
      hits = d.hits;
      who.textContent = `พบ ${d.hits.length} ส่วนที่เกี่ยวข้อง (${d.ms} ms) · กำลังให้ AI อ่าน`;
    } else if(ev === 'generating'){
      body.innerHTML = '<span class="cursor"></span>'; btm();
    } else if(ev === 'token'){
      text += d.t;
      body.innerHTML = md(text) + '<span class="cursor"></span>'; btm();
    } else if(ev === 'end'){
      who.textContent = CFG.llm_ready ? `ตอบโดย ${CFG.llm}` : 'ผลการค้นหา';
      body.innerHTML = md(text);
      if(hits) body.after(sourcesEl(hits));
      btm();
    } else if(ev === 'error'){
      who.remove(); body.outerHTML = `<div class="err">${esc(d.msg || d.error)}</div>`;
      btm();
    }
  }).catch(e => addErr(String(e)));

  busy = false; $('#send').disabled = false; $('#q').focus();
}

function sourcesEl(hits){
  const d = document.createElement('details');
  d.className = 'sources';
  d.innerHTML = `<summary>ข้อมูลที่ใช้ตอบ · ${hits.length} ส่วน</summary>` + hits.map(h => {
    const pg = h.page_from === h.page_to ? `หน้า ${h.page_from}` : `หน้า ${h.page_from}–${h.page_to}`;
    const pct = Math.max(3, Math.round(Math.max(0, h.score) * 100));
    return `<div class="src">
      <div class="h"><span class="pg">${esc(h.doc)} · ${pg}</span><span class="sc">${h.score.toFixed(3)}</span></div>
      <div class="bar"><i style="width:${pct}%"></i></div>
      <div class="t">${esc(h.text)}</div>
    </div>`;
  }).join('');
  d.querySelectorAll('.src').forEach(s => s.onclick = () => s.classList.toggle('open'));
  return d;
}

function addNote(html){
  const el = document.createElement('div');
  el.className = 'msg wrap bot';
  el.innerHTML = `<div class="body" style="color:var(--dim);font-size:14.5px">${html}</div>`;
  stream.append(el); btm();
}
function addErr(msg){
  const el = document.createElement('div');
  el.className = 'msg wrap';
  el.innerHTML = `<div class="err">${esc(msg)}</div>`;
  stream.append(el); btm();
}

/* ================= รายการเอกสาร ================= */
function renderDocs(st){
  const box = $('#docs');
  if(!st.docs.length){
    box.innerHTML = '<div class="empty">ยังไม่มีเอกสารในคลัง</div>'; return;
  }
  box.innerHTML = st.docs.map(d => `<div class="doc">
      <div class="nm"><b>${esc(d.name)}</b><span>${d.pages} หน้า · ${d.chunks} chunk</span></div>
      <button title="เอาออก" data-n="${esc(d.name)}">×</button>
    </div>`).join('');
  box.querySelectorAll('button').forEach(b => b.onclick = async () => {
    const r = await fetch('/api/remove', {method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({name:b.dataset.n})});
    renderDocs(await r.json());
  });
}

/* ================= เชื่อม event ================= */
$('#drop').onclick = () => $('#file').click();
$('#drop').onkeydown = e => { if(e.key === 'Enter' || e.key === ' ') $('#file').click(); };
$('#file').onchange = e => { if(e.target.files[0]) upload(e.target.files[0]); e.target.value = ''; };
['dragenter','dragover'].forEach(t => $('#drop').addEventListener(t, e => {
  e.preventDefault(); $('#drop').classList.add('over');
}));
['dragleave','drop'].forEach(t => $('#drop').addEventListener(t, e => {
  e.preventDefault(); $('#drop').classList.remove('over');
}));
$('#drop').addEventListener('drop', e => { if(e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => e.preventDefault());

$('#send').onclick = ask;
$('#q').addEventListener('keydown', e => {
  if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); ask(); }
});
$('#q').addEventListener('input', e => {
  e.target.style.height = 'auto';
  e.target.style.height = Math.min(e.target.scrollHeight, 180) + 'px';
});
$('#k').oninput = e => $('#kv').textContent = e.target.value;
$('#reset').onclick = async () => {
  if(!confirm('ลบเอกสารทั้งหมดออกจากคลัง?')) return;
  renderDocs(await (await fetch('/api/reset', {method:'POST'})).json());
};

fetch('/api/status').then(r => r.json()).then(renderDocs);
$('#q').focus();
</script>
</body>
</html>"""


# ============================================================================
#  10) เริ่มทำงาน
# ============================================================================

if __name__ == "__main__":
    pick_providers()
    STORE.load()
    mode, detail = EMBED_LABEL[EMBED_ACTIVE]
    print("\n" + "=" * 62)
    print("  RAG Chat  ·  ถาม-ตอบจากไฟล์ PDF")
    print("=" * 62)
    print(f"  ค้นหา  : {mode} — {detail}")
    print(f"  ผู้ตอบ : {llm_label()}")
    if LLM_ACTIVE == "none":
        print("           (ยังไม่ได้ต่อ LLM — ระบบจะแสดงเฉพาะเนื้อหาที่ค้นเจอ)")
    if STORE.chunks:
        print(f"  คลัง   : {len(STORE.docs)} เอกสาร · {len(STORE.chunks)} chunk")
    for w in WARNINGS:
        print(f"\n  [!]    {w}")
    print(f"\n  เปิด   : http://{HOST}:{PORT}")
    print("=" * 62 + "\n")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
