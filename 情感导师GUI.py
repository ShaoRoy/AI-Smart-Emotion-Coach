# -*- coding: utf-8 -*-
# ============================================
# 情感助手 — 本地 RAG 智能问答系统
# 作者: RoyShao
# 项目地址: https://github.com/ShaoRoy/AI-Smart-Emotion-Coach
# 联系邮箱: yingshao113113@163.com
# 许可: CC BY-NC-ND 4.0（仅供个人学习使用，禁止商用）
# ============================================
__author__ = "RoyShao"
__copyright__ = "Copyright (c) 2025 RoyShao. All rights reserved."
__license__ = "CC BY-NC-ND 4.0"
__project__ = "AI-Smart-Emotion-Coach"
__github__ = "https://github.com/ShaoRoy/AI-Smart-Emotion-Coach"
__email__ = "yingshao113113@163.com"
__signature__ = "RoyShao\u200b2025\u200bAI-Smart-Emotion-Coach"
# ============================================================
# 启动保护：单实例锁 + 早期闪屏（必须在 import 大库之前）
# ============================================================
import sys as _sys
import os as _os

# ① 单实例锁（Windows）：已经有实例在跑 → 直接退出
_MUTEX_HANDLE = None
if _os.name == "nt":
    try:
        import ctypes as _ctypes
        _MUTEX_NAME = "Global\\AI_Smart_Emotion_Coach_SingleInstance_v1"
        _MUTEX_HANDLE = _ctypes.windll.kernel32.CreateMutexW(None, False, _MUTEX_NAME)
        if _ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            _sys.exit(0)
    except Exception:
        pass


# ============================================================
# 后续 import 大库
# ============================================================
import os
import re
import sys
import json
import random
import shutil
import pickle
import logging
import threading
import subprocess
import queue
import time
import tempfile
import tkinter as tk
import tkinter.messagebox
from pathlib import Path

import numpy as np
import faiss
import requests
from cryptography.fernet import Fernet
from difflib import SequenceMatcher

from license_core import decrypt_index, decrypt_bm25, encrypt_bm25

# 重排序：直接用 transformers（不依赖 FlagEmbedding）
_RERANKER_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _RERANKER_AVAILABLE = True
    print("transformers + torch loaded, reranker available")
except Exception as _e:
    print("transformers/torch load failed:", _e)
    _RERANKER_AVAILABLE = False

try:
    import customtkinter as ctk
except Exception:
    ctk = None

try:
    import jieba
    from rank_bm25 import BM25Okapi
except Exception:
    jieba = None
    BM25Okapi = None

# ============================================================
# 一、配置区
# ============================================================

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
    RESOURCE_DIR = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else BASE_DIR
else:
    BASE_DIR = Path(__file__).resolve().parent
    RESOURCE_DIR = BASE_DIR

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = None

ENC_INDEX_FILE = BASE_DIR / "index_data" / "integrated_index.faiss.enc"
ENC_CHUNKS_FILE = BASE_DIR / "index_data" / "integrated_chunks.pkl.enc"

STAGE_DATA_FILE = RESOURCE_DIR / "stage_data.json"
FEEDBACK_FILE = BASE_DIR / "feedback.json"
PHRASES_FILE = BASE_DIR / "quick_phrases.json"

RETRIEVE_K = 16
DISTANCE_THRESHOLD = 1.0
CTX_COUNT = 5
MAX_CTX_LEN = 1500
TEMPERATURE = 0.6
NUM_PREDICT = 500
HISTORY_ROUNDS = 5

OLLAMA_START_TIMEOUT = 45

R_MAIN = 14
R_CARD = 8
R_BTN  = 8
PAD_MAIN = 10

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("jieba").setLevel(logging.ERROR)
logger = logging.getLogger("qingan_daoshi")

# ============================================================
# 二、问答引擎（SmartQA）
# ============================================================

class SmartQA:
    """
    情感助手 - RAG 问答引擎
    作者: RoyShao
    项目: AI-Smart-Emotion-Coach
    License: CC BY-NC-ND 4.0
    """
    def __init__(self, progress_callback=None):
        self.index = None
        self.chunks = []
        self.history = []
        self.embed_model = None
        self.chat_model = CHAT_MODEL
        self.ollama_proc = None
        self.reranker = None
        self._rerank_tokenizer = None
        self._rerank_model = None
        self.question_to_stage = {}
        self.stage_questions = {}
        self.temperature = TEMPERATURE
        self.num_predict = NUM_PREDICT
        self.ctx_count = CTX_COUNT
        self.bm25 = None
        self._bm25_map = []
        self._progress = progress_callback
        self._step_idx = 0

        self._begin_step("正在连接 Ollama 服务...")
        self._ensure_ollama()
        self._finish_step("Ollama 服务已在运行")
        self._check_models()
        self._begin_step("正在解密索引文件（约 1.7GB）...")
        self._load_index()
        self._finish_step(f"已加载 {self.index.ntotal} 个向量，{len(self.chunks)} 个数据块")
        self._begin_step("正在构建关键词索引(BM25)...")
        self._build_bm25()
        self._begin_step("正在加载重排序模型...")
        self._load_reranker_smart()
        self._begin_step("正在加载阶段数据...")
        self._load_stage_data()
        self._finish_step(f"阶段数据已加载: {len(self.stage_questions)} 个阶段，"
                          f"{len(self.question_to_stage)} 个问题映射")
        self._report("")

    def set_progress_callback(self, fn):
        self._progress = fn

    def _report(self, text):
        if self._progress:
            try:
                self._progress("work", text)
            except Exception:
                pass

    def _begin_step(self, detail):
        if self._progress:
            try:
                self._progress("work", detail)
            except Exception:
                pass

    def _finish_step(self, detail):
        if self._progress:
            try:
                self._progress("done", detail)
            except Exception:
                pass
        self._step_idx += 1

    @staticmethod
    def _ping():
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _find_ollama_exe():
        portable = BASE_DIR / "runtime" / "ollama" / "ollama.exe"
        if portable.exists():
            return portable
        which = shutil.which("ollama")
        if which:
            return Path(which)
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if local.exists():
            return local
        return None

    def _ensure_ollama(self):
        if self._ping():
            logger.info("✅ Ollama 服务已在运行")
            return

        logger.info("🔍 未检测到 Ollama 服务，尝试自动启动...")
        exe = self._find_ollama_exe()
        if exe is None:
            raise RuntimeError("未检测到 Ollama 服务，且未找到 ollama.exe。请先安装 Ollama。")

        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.ollama_proc = subprocess.Popen([str(exe), "serve"],
                                            creationflags=flags,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)

        deadline = time.time() + OLLAMA_START_TIMEOUT
        while time.time() < deadline:
            if self._ping():
                logger.info("✅ Ollama 服务已就绪")
                return
            if self.ollama_proc.poll() is not None:
                raise RuntimeError("Ollama 进程启动后立即退出，请检查安装是否正常。")
            time.sleep(1)

        raise RuntimeError(f"Ollama 服务启动超时（{OLLAMA_START_TIMEOUT}s），请手动启动后重试。")

    def _check_models(self):
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            models = [m['name'] for m in resp.json()['models']]
        except Exception as e:
            raise RuntimeError(f"连接 Ollama 失败：{e}")

        self._begin_step("正在校验聊天模型...")
        for m in models:
            if CHAT_MODEL in m:
                self.chat_model = m
                break
        self._finish_step(f"聊天模型: {self.chat_model}")

        self._begin_step("正在校验嵌入模型...")
        for m in models:
            if 'bge' in m.lower() or 'embed' in m.lower():
                self.embed_model = m
                break
        if not self.embed_model:
            raise RuntimeError("❌ 未找到专用嵌入模型（如 bge-m3），请先执行：ollama pull bge-m3")
        self._finish_step(f"嵌入模型: {self.embed_model}")
        logger.info(f"✅ 聊天模型: {self.chat_model}")
        logger.info(f"✅ 嵌入模型: {self.embed_model}")

    def _load_index(self):
        if not ENC_INDEX_FILE.exists() or not ENC_CHUNKS_FILE.exists():
            raise RuntimeError(
                f"未找到加密索引文件：\n{ENC_INDEX_FILE}\n{ENC_CHUNKS_FILE}\n"
                "请确认 index_data 目录与本程序在同一目录下。")

        logger.info("🔓 开始解密索引文件（约 1.7GB，请耐心等待）...")

        index_bytes = decrypt_index(ENC_INDEX_FILE.read_bytes())
        chunk_bytes = decrypt_index(ENC_CHUNKS_FILE.read_bytes())
        logger.info("🔓 解密完成，正在加载到内存...")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".faiss")
        try:
            os.close(tmp_fd)
            with open(tmp_path, "wb") as f:
                f.write(index_bytes)
            self.index = faiss.read_index(tmp_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
        del index_bytes

        self.chunks = pickle.loads(chunk_bytes)
        del chunk_bytes

        logger.info(f"📚 已加载 {self.index.ntotal} 个向量，{len(self.chunks)} 个数据块。")

    def _build_bm25(self):
        self._bm25_map = []
        if jieba is None or BM25Okapi is None:
            logger.warning("⚠️ 未安装 jieba/rank_bm25，关键词检索不可用，使用纯向量检索")
            self.bm25 = None
            self._finish_step("BM25 关键词索引不可用（未安装 jieba/rank_bm25）")
            return

        cache_path = BASE_DIR / "index_data" / "bm25_cache.pkl.enc"
        chunks_len = len(self.chunks)

        try:
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    cache = pickle.loads(decrypt_bm25(f.read()))
                if cache.get("chunks_len") == chunks_len:
                    self.bm25 = cache["bm25"]
                    self._bm25_map = cache["bm25_map"]
                    logger.info(f"🔑 BM25 关键词索引已从缓存加载（{len(self._bm25_map)} 篇）")
                    self._finish_step(f"BM25 关键词索引已加载（缓存，{len(self._bm25_map)} 篇）")
                    return
        except Exception as e:
            logger.warning(f"BM25 缓存加载失败，重新构建: {e}")

        try:
            jieba.setLogLevel(60)
            docs = []
            for i, c in enumerate(self.chunks):
                if c.get('type') == 'qa':
                    q = c.get('question', '')
                    if q:
                        docs.append(" ".join(jieba.cut(str(q))))
                        self._bm25_map.append(i)
                if i % 50000 == 0 and i > 0:
                    self._report(f"正在构建关键词索引... {i}/{chunks_len}")

            self.bm25 = BM25Okapi([d.split() for d in docs])
            try:
                with open(cache_path, "wb") as f:
                    f.write(encrypt_bm25(pickle.dumps({
                        "bm25": self.bm25,
                        "bm25_map": self._bm25_map,
                        "chunks_len": chunks_len
                    })))
                logger.info(f"💾 BM25 索引已缓存: {cache_path.name}")
            except Exception as e:
                logger.warning(f"BM25 缓存保存失败（下次会重建）: {e}")
            logger.info(f"🔑 BM25 关键词索引已构建（{len(self._bm25_map)} 篇文档）")
            self._finish_step(f"BM25 关键词索引已构建（{len(self._bm25_map)} 篇）")
        except Exception as e:
            logger.warning(f"BM25 构建失败，降级为纯向量检索: {e}")
            self.bm25 = None
            self._bm25_map = []
            self._finish_step("BM25 关键词索引构建失败（降级为纯向量检索）")

    def _bm25_search(self, query, top_k=20):
        if self.bm25 is None or not self._bm25_map:
            return []
        try:
            tokens = list(jieba.cut(str(query)))
            if not tokens:
                return []
            scores = self.bm25.get_scores(tokens)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            out = []
            for pos in ranked:
                if scores[pos] <= 0:
                    break
                out.append((self._bm25_map[pos], float(scores[pos])))
                if len(out) >= top_k:
                    break
            return out
        except Exception:
            return []

    @staticmethod
    def _rrf_fuse(vector_list, bm25_list, k=60):
        scores = {}
        for rank, (idx, _) in enumerate(vector_list):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(bm25_list):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _load_reranker_smart(self):
        if not _RERANKER_AVAILABLE:
            logger.warning("⚠️ 未安装 transformers/torch，重排序不可用，将使用距离排序")
            self._finish_step("重排序不可用（未安装 transformers/torch，使用距离排序）")
            return
        standard = BASE_DIR / "models" / "BAAI_bge-reranker-base"
        if self._try_load_from_path(standard):
            self._finish_step(f"重排序模型已启用: {standard.name}")
            return
        models_root = BASE_DIR / "models"
        if models_root.exists():
            for folder in models_root.iterdir():
                if folder.is_dir() and "bge-reranker" in folder.name.lower():
                    if self._try_load_from_path(folder):
                        self._finish_step(f"重排序模型已启用: {folder.name}")
                        return
        logger.warning("⚠️ 未找到 bge-reranker 模型，重排序降级为距离排序")
        self._finish_step("重排序未启用（未找到模型，使用距离排序）")

    def _try_load_from_path(self, path):
        try:
            if not (path / "config.json").exists():
                return False
            if not _RERANKER_AVAILABLE:
                return False

            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._rerank_tokenizer = AutoTokenizer.from_pretrained(str(path))
            self._rerank_model = AutoModelForSequenceClassification.from_pretrained(str(path))
            self._rerank_model.eval()

            if torch.cuda.is_available():
                try:
                    self._rerank_model = self._rerank_model.cuda()
                    logger.info("重排序使用 GPU 加速")
                except Exception:
                    logger.info("重排序使用 CPU")
            else:
                logger.info("重排序使用 CPU")

            self.reranker = True
            logger.info(f"✅ 重排序模型已启用: {path.name}")
            return True
        except Exception as e:
            logger.warning(f"加载重排序模型失败: {e}")
            self.reranker = None
            return False

    def _load_stage_data(self):
        if not STAGE_DATA_FILE.exists():
            logger.warning("⚠️ 未找到 stage_data.json，阶段推荐功能不可用")
            return
        try:
            with open(STAGE_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"读取 stage_data.json 失败: {e}")
            return

        temp_stage_q = {}
        for item in data:
            q = item.get("question", "")
            stage = item.get("stage", "未知")
            if q:
                self.question_to_stage[q] = stage
                temp_stage_q.setdefault(stage, []).append(q)
        self.stage_questions = {s: list(set(lst)) for s, lst in temp_stage_q.items()}
        logger.info(f"📂 阶段数据已加载: {len(self.stage_questions)} 个阶段，"
                    f"{len(self.question_to_stage)} 个问题映射")

    def get_stage_questions(self, stage, limit=8):
        qs = self.stage_questions.get(stage, [])
        if not qs:
            return []
        return random.sample(qs, min(limit, len(qs)))

    def get_stage_list(self):
        order = ["初识", "吸引", "熟悉", "舒适", "暧昧", "确认关系"]
        return [s for s in order if s in self.stage_questions]

    def _get_embedding(self, text):
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/embed",
                                 json={"model": self.embed_model, "input": [text[:1000]]},
                                 timeout=30)
            if resp.status_code == 200:
                return np.array(resp.json()['embeddings'][0], dtype='float32')
        except Exception as e:
            logger.warning(f"获取嵌入失败: {e}")
        return None

    @staticmethod
    def _clean_answer(text):
        if not text:
            return ""
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        lines = text.split('\n')
        cleaned_lines = []
        term_count = 0
        for line in lines:
            if re.match(r'\d+\.\s+[A-Za-z一-鿿]+[：:]', line):
                term_count += 1
            else:
                term_count = 0
            if term_count >= 3:
                continue
            cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(?<=[。！？；])[ \t]*', '\n', text)

        tail_patterns = [
            r'\n\s*句号[。\.]?\s*$',
            r'\n\s*以上[。\.]?\s*$',
            r'\n\s*完毕[。\.]?\s*$',
            r'\n\s*[（\(]完[）\)]\s*$',
            r'\n\s*回答完毕[。\.]?\s*$',
        ]
        for pat in tail_patterns:
            text = re.sub(pat, '', text)

        text = re.sub(r'^句号[。\.]?\s*', '', text)

        lines = [ln for ln in text.split('\n') if ln.strip()]
        while lines and lines[-1].strip() in ["句号。", "句号", "以上。", "以上", "完毕。", "完毕", "（完）", "(完)"]:
            lines.pop()
        text = '\n'.join(lines)

        return text.strip()

    def _generate_answer(self, query, context, history_text=""):
        prompt = f"""你是一个专业情感导师。请根据下面的参考资料，回答用户的问题。
要求：
- 使用自然口语，像朋友聊天一样。
- **重要**：如果用户的问题包含多个部分（比如同时问了"哪里"和"怎么"），你必须分别回答每一个部分，并清楚地分开表述（例如使用"首先……其次……"或分段说明）。不要遗漏任何一个问题。
- 如果资料中有明确定义或方法，请优先采用，不要自己编造。
- 只回答用户直接问到的问题，不要过度延伸，不要罗列无关术语。
- 不要使用任何 Markdown 符号（如 **、#、* 等），保持纯文本。
- 每说完一层意思就换行，不要把多句话堆成一大段。

{history_text}
参考资料：
{context}

用户问题：{query}
助手回答："""
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate",
                                 json={
                                     "model": self.chat_model,
                                     "prompt": prompt,
                                     "stream": False,
                                     "options": {"temperature": self.temperature,
                                                 "num_predict": self.num_predict}
                                 },
                                 timeout=120)
            if resp.status_code == 200:
                return self._clean_answer(resp.json().get('response', ''))
        except Exception as e:
            logger.warning(f"生成回答失败: {e}")
        return None

    def _search(self, query):
        vec = self._get_embedding(query)
        if vec is None:
            return "抱歉，无法处理您的问题。", []

        vec = vec.reshape(1, -1)
        k = min(RETRIEVE_K, len(self.chunks))
        distances, indices = self.index.search(vec, k)

        vector_cands = []
        for d, idx in zip(distances[0], indices[0]):
            if d > DISTANCE_THRESHOLD:
                continue
            if idx < 0 or idx >= len(self.chunks):
                continue
            vector_cands.append((idx, float(d)))

        bm25_cands = self._bm25_search(query, top_k=20)

        if bm25_cands and vector_cands:
            fused = self._rrf_fuse(vector_cands, bm25_cands)
        else:
            fused = [(idx, 1.0 / (1 + rank)) for rank, (idx, _) in enumerate(vector_cands)]

        qa_pairs = []
        suggestions = []
        seen_questions = set()

        for idx, _score in fused:
            chunk = self.chunks[idx]
            if chunk.get('type') == 'qa':
                q = chunk.get('question', '')
                a = chunk.get('answer', '')
                if q and a:
                    qa_pairs.append((0.0, q, a))
                    if q != query and q not in seen_questions:
                        suggestions.append((0.0, q))
                        seen_questions.add(q)
            else:
                content = chunk.get('content', '')
                if content:
                    qa_pairs.append((0.0, '', content))

        if not qa_pairs:
            return "抱歉，没有找到相关资料。", []

        cleaned_pairs = []
        for dist, q, a in qa_pairs:
            if q:
                cleaned_pairs.append((dist, q, a))
            else:
                clean_c = re.sub(r'\[[^\]]*\]\s*', '', a)
                clean_c = re.sub(r'\n{2,}', '\n', clean_c).strip()
                cleaned_pairs.append((dist, q, clean_c))
        qa_pairs = cleaned_pairs

        pairs = [[query, f"{q} {a}" if q else a] for _, q, a in qa_pairs]
        try:
            if self.reranker and _RERANKER_AVAILABLE and self._rerank_model is not None:
                import torch

                texts_a = [p[0] for p in pairs]
                texts_b = [p[1] for p in pairs]

                with torch.no_grad():
                    inputs = self._rerank_tokenizer(
                        texts_a, texts_b,
                        padding=True, truncation=True,
                        return_tensors="pt", max_length=512
                    )
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    logits = self._rerank_model(**inputs).logits.view(-1).float()
                    scores = torch.sigmoid(logits).cpu().tolist()
            else:
                scores = [1.0 / (1.0 + dist) for dist, _, _ in qa_pairs]
        except Exception as e:
            logger.warning(f"重排序计算失败，降级为距离排序: {e}")
            scores = [1.0 / (1.0 + dist) for dist, _, _ in qa_pairs]

        scored = list(zip(scores, qa_pairs))
        scored.sort(key=lambda x: x[0], reverse=True)

        final_items = []
        if scored:
            top_score = scored[0][0]
            for s, item in scored:
                if s < 0.3:
                    break
                if s < top_score * 0.6:
                    break
                _, q, a = item
                text = a[:200].replace(' ', '') if a else ''
                dup = False
                for fs, (_, fq, fa) in final_items:
                    ftext = fa[:200].replace(' ', '') if fa else ''
                    if text and ftext and SequenceMatcher(None, text, ftext).ratio() > 0.65:
                        dup = True
                        break
                if not dup:
                    final_items.append((s, item))
        qa_pairs_reranked = [x[1] for x in final_items[:self.ctx_count]]

        context_parts = []
        total_len = 0
        for d, q, a in qa_pairs_reranked:
            part = f"问：{q}\n答：{a}" if q else a.strip()
            if total_len >= MAX_CTX_LEN:
                break
            if total_len + len(part) > MAX_CTX_LEN:
                continue
            context_parts.append(part)
            total_len += len(part)
        context_text = "\n\n".join(context_parts)

        history_text = ""
        if self.history:
            turns = []
            for q_, a_ in self.history[-HISTORY_ROUNDS:]:
                turns.append(f"用户：{q_}\n助手：{a_}")
            history_text = "对话历史：\n" + "\n\n".join(turns) + "\n"

        answer = self._generate_answer(query, context_text, history_text)
        if not answer:
            answer = context_parts[0][:200] + "..."

        final_suggestions = []
        seen = set()

        current_stage = self.question_to_stage.get(query, None)
        if not current_stage and qa_pairs_reranked:
            first_q = qa_pairs_reranked[0][1]
            if first_q:
                current_stage = self.question_to_stage.get(first_q, None)
        if current_stage and current_stage in self.stage_questions:
            stage_qs = [q for q in self.stage_questions[current_stage] if q != query]
            random.shuffle(stage_qs)
            for q in stage_qs[:4]:
                if q not in seen:
                    final_suggestions.append(q)
                    seen.add(q)

        if len(final_suggestions) < 4:
            for _, q, _ in qa_pairs_reranked:
                if q and q != query and q not in seen:
                    final_suggestions.append(q)
                    seen.add(q)
        if len(final_suggestions) < 4:
            for _, q in suggestions:
                if q and q != query and q not in seen:
                    final_suggestions.append(q)
                    seen.add(q)
        return answer, final_suggestions[:4]

    def set_params(self, temperature=None, num_predict=None, ctx_count=None):
        if temperature is not None:
            self.temperature = temperature
        if num_predict is not None:
            self.num_predict = num_predict
        if ctx_count is not None:
            self.ctx_count = ctx_count

    def answer(self, query):
        answer, suggestions = self._search(query)
        self.history.append((query, answer))
        if len(self.history) > HISTORY_ROUNDS:
            self.history = self.history[-HISTORY_ROUNDS:]
        return answer, suggestions

    def clear_history(self):
        self.history = []

    def stop(self):
        if self.ollama_proc and self.ollama_proc.poll() is None:
            try:
                self.ollama_proc.terminate()
            except Exception:
                pass
# ============================================================
# 三、图形界面
# ============================================================

DARK_COLORS = {
    "bg":        "#1E1E2E",
    "sidebar":   "#181825",
    "panel":     "#23233a",
    "hover":     "#313244",
    "border":    "#45475A",
    "divider":   "#313244",
    "user_bub":  "#FAB387",
    "ai_bub":    "#313244",
    "accent":    "#FAB387",
    "accent_h":  "#F9A46F",
    "accent_d":  "#E8844C",
    "on_accent": "#1E1E2E",
    "text":      "#CDD6F4",
    "text_dim":  "#A6ADC8",
    "text_faint":"#7F849C",
    "danger":    "#F38BA8",
    "ok":        "#A6E3A1",
    "info":      "#89B4FA",
}

LIGHT_COLORS = {
    "bg":        "#F2F3F7",
    "sidebar":   "#E8EAEF",
    "panel":     "#FFFFFF",
    "hover":     "#DDE0E6",
    "border":    "#C3C7D0",
    "divider":   "#D5D8DF",
    "user_bub":  "#E8833A",
    "ai_bub":    "#FFFFFF",
    "accent":    "#D9762A",
    "accent_h":  "#E8833A",
    "accent_d":  "#B45F1E",
    "on_accent": "#FFFFFF",
    "text":      "#1C1D24",
    "text_dim":  "#4E505C",
    "text_faint":"#82848F",
    "danger":    "#D03030",
    "ok":        "#3E8E4E",
    "info":      "#2D6FB8",
}

COLORS = DARK_COLORS


class ChatApp:
    """
    情感助手 - 图形界面
    作者: RoyShao
    项目: AI-Smart-Emotion-Coach
    License: CC BY-NC-ND 4.0
    """
    def __init__(self, root):
        self.root = root
        self.engine = None
        self._busy = False
        self._closing = False
        self._msg_queue = queue.Queue()

        self.font_size = 13
        self.typewriter = True
        self.temperature = TEMPERATURE
        self.num_predict = NUM_PREDICT
        self.ctx_count = CTX_COUNT
        self.text_color = COLORS["text"]
        self.bubble_color = COLORS["ai_bub"]
        self.theme = "dark"
        self._rendered = []

        self._thinking_frame = None
        self._thinking_label = None
        self._thinking_dots = 0

        self._stage_panel = None
        self._stage_panel_move_id = None

        self._log = []
        self._log_dir = BASE_DIR / "聊天记录"
        self._log_file = None
        try:
            self._log_dir.mkdir(exist_ok=True)
        except Exception:
            logger.warning("无法创建聊天记录目录，记录功能将只在窗口内生效")

        self._load_phrases()

        try:
            self.root.configure(fg_color=COLORS["bg"])
        except Exception:
            pass

        self._build_ui()
        self._set_status("正在初始化...")

        self._show_loading()

        threading.Thread(target=self._init_engine, daemon=True).start()
        self.root.after(50, self._poll_queue)

    def _load_phrases(self):
        self.phrases = [
            "帮我分析一下我的情况",
            "给我一些具体建议",
            "这种情况我该怎么做",
            "帮我判断一下她的想法",
        ]
        try:
            if PHRASES_FILE.exists():
                data = json.loads(PHRASES_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    self.phrases = data
        except Exception:
            pass

    def _save_phrases(self):
        try:
            PHRASES_FILE.write_text(json.dumps(self.phrases, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        except Exception:
            pass

    def _apply_ctk_theme(self):
        theme_file = ("catppuccin-mocha-peach.json" if self.theme == "dark"
                      else "catppuccin-latte-peach.json")
        try:
            theme_path = str(RESOURCE_DIR / "themes" / theme_file)
            ctk.set_default_color_theme(theme_path)
        except Exception:
            ctk.set_default_color_theme("dark-blue" if self.theme == "dark" else "blue")

    def _center_win(self, win, w, h):
        win.update_idletasks()
        try:
            rx = self.root.winfo_x()
            ry = self.root.winfo_y()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            x = rx + (rw - w) // 2
            y = ry + (rh - h) // 2
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            win.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            win.geometry(f"{w}x{h}")
        try:
            win.lift()
            win.attributes("-topmost", True)

            def _safe_unpin():
                try:
                    if win.winfo_exists():
                        win.attributes("-topmost", False)
                except Exception:
                    pass

            win.after(300, _safe_unpin)

            try:
                if win.winfo_exists():
                    win.focus_force()
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _unpin_topmost(win):
        try:
            win.attributes("-topmost", False)
        except Exception:
            pass

    def _build_ui(self):
        self._stage_panel = None
        self._stage_panel_move_id = None

        try:
            self.root.configure(fg_color=COLORS["bg"])
        except Exception:
            pass

        self.root.title("情感助手")

        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            w = min(1200, sw - 100)
            h = min(840, sh - 100)
        except Exception:
            w, h = 1200, 840
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(1000, 720)

        ctk.set_appearance_mode("dark" if self.theme == "dark" else "light")
        self._apply_ctk_theme()

        self._f_lg = ("Microsoft YaHei UI", 14)
        self._f_md = ("Microsoft YaHei UI", 13)
        self._f_sm = ("Microsoft YaHei UI", 12)
        self._f_xs = ("Microsoft YaHei UI", 11)
        self._ui_font = self._f_md
        P8, P12, P16, R8 = 8, 12, 16, R_BTN

        # ===== 顶部标题栏 =====
        top = ctk.CTkFrame(self.root, fg_color=COLORS["panel"], corner_radius=R_MAIN, height=40)
        top.pack(side="top", fill="x", padx=PAD_MAIN, pady=(PAD_MAIN, 6))
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="情感助手", text_color=COLORS["text"],
                     font=("Microsoft YaHei UI", 13, "bold")).pack(side="left", padx=P12)

        self.status_var = ctk.StringVar(value="初始化中")
        ctk.CTkLabel(top, textvariable=self.status_var, text_color=COLORS["text_dim"],
                     font=self._f_xs).pack(side="right", padx=P8)
        self.theme_btn = ctk.CTkButton(top, text="◐ 浅色", width=64, height=26,
                                       corner_radius=R_BTN,
                                       fg_color=COLORS["bg"], hover_color=COLORS["hover"],
                                       text_color=COLORS["text"], font=self._f_xs,
                                       command=self._toggle_theme)
        self.theme_btn.pack(side="right", padx=(P8, P8))
        self.mode_var = ctk.StringVar(value="问答")
        self.mode_sel = ctk.CTkSegmentedButton(top, values=["问答", "接入App智能聊天"],
                                               variable=self.mode_var, height=26,
                                               corner_radius=R_BTN,
                                               width=230,
                                               font=self._f_xs,
                                               text_color="#000000",
                                               selected_color=COLORS["accent"],
                                               selected_hover_color=COLORS["accent_h"],
                                               unselected_color="#E5E5E5",
                                               unselected_hover_color="#D5D5D5",
                                               command=self._on_mode_change)
        self.mode_sel.pack(side="right", padx=(P8, 0), pady=7)

        # ===== 主体 =====
        body = ctk.CTkFrame(self.root, fg_color=COLORS["bg"], corner_radius=0)
        body.pack(side="top", fill="both", expand=True, padx=PAD_MAIN, pady=6)
        self._body = body

        # 侧边栏
        sidebar = ctk.CTkFrame(body, fg_color=COLORS["sidebar"], width=260,
                               corner_radius=R_MAIN)
        sidebar.pack(side="left", fill="y", padx=(0, PAD_MAIN))
        sidebar.pack_propagate(False)

        btn_new = ctk.CTkButton(sidebar, text="＋ 新会话", command=self._new_session,
                                fg_color=COLORS["accent"], hover_color=COLORS["accent_h"],
                                text_color=COLORS["on_accent"], height=36,
                                corner_radius=R_BTN,
                                font=("Microsoft YaHei UI", 13, "bold"))
        btn_new.pack(fill="x", padx=P12, pady=(P12, P8))

        ctk.CTkLabel(sidebar, text="历史会话", text_color=COLORS["text_faint"],
                     font=self._f_xs).pack(padx=P12, pady=(0, 6), anchor="w")
        self._history_box = ctk.CTkScrollableFrame(sidebar, fg_color="transparent",
                                                   corner_radius=0, height=80)
        self._history_box.pack(fill="x", padx=P8, pady=2)
        self._refresh_history_list()
        ctk.CTkButton(sidebar, text="查看全部记录", height=28, corner_radius=R_BTN,
                      fg_color=COLORS["panel"], hover_color=COLORS["hover"],
                      text_color=COLORS["text"], font=self._f_xs,
                      command=self._view_history).pack(fill="x", padx=P12, pady=(2, 8))

        stage_label_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        stage_label_frame.pack(fill="x", padx=P12, pady=(0, 6), anchor="w")

        ctk.CTkLabel(stage_label_frame, text="阶段库",
                     text_color=COLORS["text"],
                     font=("Microsoft YaHei UI", 11, "bold")).pack(side="left")

        ctk.CTkLabel(stage_label_frame, text="（每次打开都会刷新热点问题）",
                     text_color=COLORS["info"],
                     font=("Microsoft YaHei UI", 10)).pack(side="left", padx=(4, 0))

        stage_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        stage_frame.pack(fill="x", padx=P12, pady=(0, 8))
        for i, stage in enumerate(["初识", "吸引", "熟悉", "舒适", "暧昧", "确认关系"]):
            b = ctk.CTkButton(stage_frame, text=stage, height=28, corner_radius=R_BTN,
                              fg_color=COLORS["panel"], hover_color=COLORS["hover"],
                              text_color=COLORS["text"], font=self._f_xs,
                              command=lambda s=stage: self._view_stage(s))
            b.grid(row=i // 3, column=i % 3, padx=4, pady=4, sticky="ew")
            stage_frame.grid_columnconfigure(i % 3, weight=1)

        donate_card = ctk.CTkFrame(sidebar, fg_color=COLORS["panel"],
                                    corner_radius=R_CARD,
                                    border_width=1, border_color=COLORS["border"])
        donate_card.pack(side="bottom", fill="x", padx=P12, pady=(8, P12))

        ctk.CTkLabel(donate_card, text="作者：RoyShao",
                     text_color=COLORS["text"],
                     font=("Microsoft YaHei UI", 11, "bold")).pack(pady=(8, 1))

        ctk.CTkLabel(donate_card, text="邮箱：yingshao113113@163.com",
                     text_color=COLORS["text_dim"],
                     font=("Microsoft YaHei UI", 9)).pack(pady=1)

        ctk.CTkLabel(donate_card, text="github.com/ShaoRoy",
                     text_color=COLORS["info"],
                     font=("Microsoft YaHei UI", 9)).pack(pady=(1, 6))

        ctk.CTkButton(donate_card, text="☕ 支持作者（打赏）",
                      command=self._open_donate,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_h"],
                      text_color=COLORS["on_accent"], height=30, corner_radius=R_BTN,
                      font=("Microsoft YaHei UI", 11, "bold")).pack(fill="x", padx=12, pady=(0, 6))

        btn_row = ctk.CTkFrame(donate_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkButton(btn_row, text="设置",
                      command=self._open_settings,
                      fg_color=COLORS["bg"], hover_color=COLORS["hover"],
                      text_color=COLORS["text_dim"], height=26, corner_radius=R_BTN,
                      font=("Microsoft YaHei UI", 10)).pack(side="left", fill="x", expand=True, padx=(0, 3))

        ctk.CTkButton(btn_row, text="关于",
                      command=self._open_about,
                      fg_color=COLORS["bg"], hover_color=COLORS["hover"],
                      text_color=COLORS["text_dim"], height=26, corner_radius=R_BTN,
                      font=("Microsoft YaHei UI", 10)).pack(side="right", fill="x", expand=True, padx=(3, 0))

        # 聊天区
        chat_wrap = ctk.CTkFrame(body, fg_color=COLORS["panel"],
                                  corner_radius=R_MAIN,
                                  border_width=1, border_color=COLORS["border"])
        chat_wrap.pack(side="left", fill="both", expand=True)

        guide = ctk.CTkFrame(chat_wrap, fg_color=COLORS["bg"], corner_radius=R_CARD, height=36)
        guide.pack(fill="x", padx=P12, pady=(P12, 6))
        guide.pack_propagate(False)
        ctk.CTkLabel(guide, text="初识  →  吸引  →  熟悉  →  舒适  →  暧昧  →  确认关系",
                     text_color=COLORS["text_dim"], font=self._f_xs).pack(pady=9)

        self.chat_area = ctk.CTkScrollableFrame(chat_wrap, fg_color="transparent",
                                                corner_radius=0)
        self.chat_area.pack(fill="both", expand=True, padx=P12, pady=(0, P8))

        # 底部输入区
        bottom = ctk.CTkFrame(self.root, fg_color=COLORS["panel"],
                               corner_radius=R_MAIN, height=76)
        bottom.pack(side="bottom", fill="x", padx=PAD_MAIN, pady=(6, PAD_MAIN))
        bottom.pack_propagate(False)

        self._phrase_menu = ctk.CTkOptionMenu(bottom, values=self.phrases,
                                              width=112, height=36, corner_radius=R_BTN,
                                              fg_color=COLORS["bg"],
                                              button_color=COLORS["bg"],
                                              button_hover_color=COLORS["hover"],
                                              text_color=COLORS["text"],
                                              dropdown_fg_color=COLORS["panel"],
                                              dropdown_hover_color=COLORS["hover"],
                                              dropdown_text_color=COLORS["text"],
                                              font=self._f_xs,
                                              command=self._use_phrase)
        self._phrase_menu.pack(side="left", padx=(P12, P8), pady=20)
        self._phrase_menu.set("快捷短语")

        self.entry = ctk.CTkEntry(bottom, placeholder_text="输入你的问题，回车发送…",
                                  height=36, corner_radius=R_BTN,
                                  fg_color=COLORS["bg"], border_color=COLORS["border"],
                                  font=self._f_md)
        self.entry.pack(side="left", fill="x", expand=True, padx=P8, pady=20)
        self.entry.bind("<Return>", lambda e: self._send())

        self.send_btn = ctk.CTkButton(bottom, text="发送", width=80, height=36,
                                      corner_radius=R_BTN,
                                      fg_color=COLORS["accent"], hover_color=COLORS["accent_h"],
                                      text_color="#000000", state="disabled",
                                      font=("Microsoft YaHei UI", 13, "bold"),
                                      command=self._send)
        self.send_btn.pack(side="left", padx=(P8, P12), pady=20)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- 阶段面板 ----------
    def _hide_stage_panel(self):
        move_id = getattr(self, "_stage_panel_move_id", None)
        if move_id:
            try:
                self.root.unbind("<Configure>", move_id)
            except Exception:
                pass
        if getattr(self, "_stage_panel", None) is not None:
            try:
                self._stage_panel.destroy()
            except Exception:
                pass
        self._stage_panel = None
        self._stage_panel_move_id = None

    def _view_stage(self, stage):
        if self.engine is None:
            self._append_error("引擎尚未就绪，请稍候再试。")
            return

        self._hide_stage_panel()

        questions = self.engine.get_stage_questions(stage, limit=30)

        PANEL_W = 360

        TRANSPARENT_KEY = "#ff00fe"
        panel = ctk.CTkToplevel(self.root)
        panel.overrideredirect(True)
        transparent_ok = False
        try:
            panel.configure(fg_color=TRANSPARENT_KEY)
        except Exception:
            try:
                panel.configure(bg=TRANSPARENT_KEY)
            except Exception:
                pass
        try:
            panel.wm_attributes("-transparentcolor", TRANSPARENT_KEY)
            transparent_ok = True
        except Exception:
            pass

        def _calc_geom():
            self.root.update_idletasks()
            rx = self.root.winfo_x()
            ry = self.root.winfo_y()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            px = rx - PANEL_W - 6
            if px < 0:
                px = rx + rw + 6
            return PANEL_W, rh, px, ry

        w, h, x, y = _calc_geom()
        panel.geometry(f"{w}x{h}+{x}+{y}")
        panel.lift()

        if transparent_ok:
            inner = ctk.CTkFrame(panel, fg_color=COLORS["panel"],
                                 corner_radius=R_MAIN,
                                 border_width=1, border_color=COLORS["border"])
            inner.pack(fill="both", expand=True, padx=0, pady=0)
        else:
            panel.configure(fg_color=COLORS["bg"])
            inner = ctk.CTkFrame(panel, fg_color=COLORS["panel"],
                                 corner_radius=R_MAIN,
                                 border_width=1, border_color=COLORS["border"])
            inner.pack(fill="both", expand=True, padx=6, pady=6)

        header = ctk.CTkFrame(inner, fg_color=COLORS["sidebar"],
                              corner_radius=R_MAIN, height=48)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(header, text=f"📍 {stage} 阶段",
                     text_color=COLORS["accent"],
                     font=("Microsoft YaHei UI", 14, "bold")).pack(side="left", padx=14)

        def _close_panel():
            self._hide_stage_panel()

        ctk.CTkButton(header, text="✕", width=32, height=28, corner_radius=R_BTN,
                      fg_color=COLORS["bg"], hover_color=COLORS["danger"],
                      text_color=COLORS["text"],
                      font=("Microsoft YaHei UI", 12, "bold"),
                      command=_close_panel).pack(side="right", padx=10)

        ctk.CTkLabel(inner, text="点击问题直接发送",
                     text_color=COLORS["text_faint"],
                     font=("Microsoft YaHei UI", 10)).pack(anchor="w", padx=14, pady=(10, 6))

        if not questions:
            ctk.CTkLabel(inner, text="（该阶段暂无问题）",
                         text_color=COLORS["text_dim"]).pack(pady=20)
        else:
            qs_frame = ctk.CTkScrollableFrame(inner, fg_color="transparent",
                                              corner_radius=0)
            qs_frame.pack(fill="both", expand=True, padx=6, pady=(0, 8))
            for i, q in enumerate(questions, 1):
                ctk.CTkButton(qs_frame, text=f"{i}. {q}", height=40,
                              corner_radius=R_BTN,
                              fg_color=COLORS["bg"], hover_color=COLORS["accent"],
                              text_color=COLORS["text"],
                              font=("Microsoft YaHei UI", 12),
                              anchor="w",
                              command=lambda x=q: self._send_from_stage(x)).pack(
                                  fill="x", pady=3, padx=4)

        def _on_root_move(event=None):
            try:
                w2, h2, x2, y2 = _calc_geom()
                panel.geometry(f"{w2}x{h2}+{x2}+{y2}")
            except Exception:
                pass

        move_id = self.root.bind("<Configure>", _on_root_move, add="+")

        self._stage_panel = panel
        self._stage_panel_move_id = move_id

    def _send_from_stage(self, q):
        self._hide_stage_panel()
        self.entry.delete(0, "end")
        self.entry.insert(0, q)
        self._send()

    # ---------- 弹窗 ----------
    def _open_donate(self):
        win = ctk.CTkToplevel(self.root)
        win.title("支持作者")
        win.resizable(False, False)
        self._center_win(win, 540, 600)

        ctk.CTkLabel(win, text="☕ 支持作者",
                     font=("Microsoft YaHei UI", 22, "bold"),
                     text_color=COLORS["accent"]).pack(pady=(26, 4))

        ctk.CTkLabel(win, text="如果这个工具帮到了你，欢迎请我喝杯奶茶",
                     font=("Microsoft YaHei UI", 12),
                     text_color=COLORS["text_dim"]).pack(pady=(0, 20))

        qr_frame = ctk.CTkFrame(win, fg_color="transparent")
        qr_frame.pack(pady=(0, 16))

        QR_BIG_H = 240
        self._donate_qr_images = []

        for qr_name in ["wechat.png", "alipay.png"]:
            qr_path = RESOURCE_DIR / qr_name
            if not qr_path.exists():
                continue
            try:
                from PIL import Image
                img = Image.open(qr_path).convert("RGB")
                w, h = img.size
                target_w = int(w * QR_BIG_H / h)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                        size=(target_w, QR_BIG_H))
                self._donate_qr_images.append(ctk_img)
                ctk.CTkLabel(qr_frame, image=ctk_img, text="").pack(side="left", padx=12)
            except Exception:
                pass

        ctk.CTkFrame(win, fg_color=COLORS["divider"], height=1).pack(fill="x", padx=40, pady=12)

        ctk.CTkLabel(win, text="微信 / 支付宝 · 扫一扫",
                     font=("Microsoft YaHei UI", 12),
                     text_color=COLORS["text_faint"]).pack(pady=(0, 6))

        ctk.CTkLabel(win, text="感谢支持，你的支持是我研发的动力。",
                     font=("Microsoft YaHei UI", 12),
                     text_color=COLORS["text_dim"]).pack(pady=(0, 10))

    def _copy_with_feedback(self, text, btn):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            original_text = btn.cget("text")
            original_color = btn.cget("text_color")
            btn.configure(text="✓ 已复制", text_color=COLORS["ok"])
            def restore():
                try:
                    btn.configure(text=original_text, text_color=original_color)
                except Exception:
                    pass
            self.root.after(1500, restore)
            self._set_status(f"已复制：{text[:40]}")
        except Exception:
            self._set_status("复制失败")

    def _open_about(self):
        win = ctk.CTkToplevel(self.root)
        win.title("关于")
        win.resizable(False, False)
        self._center_win(win, 500, 520)

        ctk.CTkLabel(win, text="情感助手",
                     font=("Microsoft YaHei UI", 22, "bold"),
                     text_color=COLORS["accent"]).pack(pady=(26, 4))

        ctk.CTkLabel(win, text="本地 RAG 智能问答系统",
                     font=("Microsoft YaHei UI", 12),
                     text_color=COLORS["text_dim"]).pack(pady=(0, 2))

        ctk.CTkLabel(win, text="版本 v1.0.0",
                     font=("Microsoft YaHei UI", 11),
                     text_color=COLORS["text_faint"]).pack(pady=(0, 18))

        info_frame = ctk.CTkFrame(win, fg_color=COLORS["panel"],
                                   corner_radius=R_CARD)
        info_frame.pack(fill="x", padx=24, pady=6)

        ctk.CTkLabel(info_frame, text="作者：RoyShao",
                     font=("Microsoft YaHei UI", 13, "bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=16, pady=(14, 4))

        ctk.CTkLabel(info_frame, text="意见、合作请联系邮箱：",
                     font=("Microsoft YaHei UI", 11),
                     text_color=COLORS["text"]).pack(anchor="w", padx=16, pady=(0, 2))

        email_row = ctk.CTkFrame(info_frame, fg_color="transparent")
        email_row.pack(fill="x", padx=16, pady=3)

        ctk.CTkLabel(email_row, text="yingshao113113@163.com",
                     font=("Microsoft YaHei UI", 11),
                     text_color=COLORS["text_dim"]).pack(side="left")

        email_btn = ctk.CTkButton(email_row, text="复制", width=48, height=24,
                                   corner_radius=R_BTN,
                                   fg_color=COLORS["bg"],
                                   hover_color=COLORS["accent"],
                                   text_color=COLORS["text_dim"],
                                   font=("Microsoft YaHei UI", 10))
        email_btn.configure(command=lambda: self._copy_with_feedback(
            "yingshao113113@163.com", email_btn))
        email_btn.pack(side="right")

        proj_row = ctk.CTkFrame(info_frame, fg_color="transparent")
        proj_row.pack(fill="x", padx=16, pady=(3, 14))

        ctk.CTkLabel(proj_row, text="github.com/ShaoRoy/AI-Smart-Emotion-Coach",
                     font=("Microsoft YaHei UI", 10),
                     text_color=COLORS["info"]).pack(side="left")

        proj_btn = ctk.CTkButton(proj_row, text="复制", width=48, height=24,
                                  corner_radius=R_BTN,
                                  fg_color=COLORS["bg"],
                                  hover_color=COLORS["accent"],
                                  text_color=COLORS["text_dim"],
                                  font=("Microsoft YaHei UI", 10))
        proj_btn.configure(command=lambda: self._copy_with_feedback(
            "https://github.com/ShaoRoy/AI-Smart-Emotion-Coach", proj_btn))
        proj_btn.pack(side="right")

        ctk.CTkFrame(win, fg_color=COLORS["divider"], height=1).pack(fill="x", padx=40, pady=14)

        ctk.CTkLabel(win, text="License: CC BY-NC-ND 4.0",
                     font=("Microsoft YaHei UI", 11),
                     text_color=COLORS["text_faint"]).pack(pady=(0, 4))

        ctk.CTkLabel(win, text="仅供个人学习使用，禁止商用",
                     font=("Microsoft YaHei UI", 10),
                     text_color=COLORS["text_faint"]).pack(pady=(0, 10))

        ctk.CTkButton(win, text="☕ 支持作者（打赏）", command=self._open_donate,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_h"],
                      text_color=COLORS["on_accent"], height=34, corner_radius=R_BTN,
                      font=("Microsoft YaHei UI", 12, "bold")).pack(pady=(0, 16))

    def _refresh_history_list(self):
        for w in self._history_box.winfo_children():
            w.destroy()

        try:
            all_paths = sorted(self._log_dir.glob("会话_*.txt"), reverse=True) if self._log_dir.exists() else []
            paths = [p for p in all_paths if p.stat().st_size > 0]
        except Exception:
            paths = []

        if not paths:
            ctk.CTkLabel(self._history_box, text="（暂无历史会话）",
                         text_color=COLORS["text_dim"],
                         font=("Microsoft YaHei UI", 11)).pack(pady=10)
            return

        for path in paths:
            card = ctk.CTkFrame(self._history_box, fg_color=COLORS["panel"],
                                corner_radius=R_CARD,
                                border_width=1, border_color=COLORS["border"])
            card.pack(fill="x", padx=2, pady=3)

            body = ctk.CTkButton(card, text="", height=44, corner_radius=R_CARD,
                                 fg_color=COLORS["panel"], hover_color=COLORS["hover"],
                                 text_color=COLORS["text"],
                                 font=("Microsoft YaHei UI", 11),
                                 anchor="w",
                                 command=lambda p=path: self._view_history())
            body.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=2)
            preview = self._session_preview(path)
            body.configure(text=f"{path.stem}\n{preview}")

            ctk.CTkButton(card, text="删除", width=42, height=28, corner_radius=R_BTN,
                          fg_color=COLORS["bg"], hover_color=COLORS["danger"],
                          text_color=COLORS["text_faint"],
                          font=("Microsoft YaHei UI", 11),
                          command=lambda p=path: self._delete_history_item(p)).pack(
                              side="right", padx=(2, 6), pady=2)

    def _delete_history_item(self, path):
        if not tk.messagebox.askyesno("确认删除",
                                      f"确定删除会话「{path.stem}」吗？\n删除后不可恢复。"):
            return
        try:
            path.unlink(missing_ok=True)
            self._refresh_history_list()
            self._set_status("会话已删除")
        except Exception as e:
            self._set_status(f"删除失败：{e}")

    def _open_history_session(self):
        self._view_history()

    def _delete_selected_history(self):
        self._set_status("请在列表中选择会话后，点右侧 🗑 按钮删除")

    def _on_mode_change(self, value):
        if value == "问答":
            self._set_status("问答模式")
        else:
            self.mode_var.set("问答")
            self._set_status("聊天模式开发中，敬请期待") 
    def _toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"

        global COLORS
        COLORS = LIGHT_COLORS if self.theme == "light" else DARK_COLORS

        ctk.set_appearance_mode("light" if self.theme == "light" else "dark")
        self._apply_ctk_theme()
        self.text_color = COLORS["text"]
        self.bubble_color = COLORS["ai_bub"]

        try:
            self.root.configure(fg_color=COLORS["bg"])
        except Exception:
            pass

        self._hide_stage_panel()
        saved = list(self._rendered)
        self._rendered = []
        for w in self.root.winfo_children():
            w.destroy()
        self._build_ui()

        self._append_banner()

        for role, content, sug in saved:
            self._append_msg(role, content, sug, record=False)

        self._rendered = list(saved)

        self.theme_btn.configure(text="◐ 深色" if self.theme == "light" else "◐ 浅色")
        self._set_status("已切换为浅色主题" if self.theme == "light" else "已切换为深色主题")

    def _show_loading(self):
        self._loading_dots = 0
        self._loading_done = 0
        self._loading_total = 7

        self._loading_frame = ctk.CTkFrame(self.chat_area, fg_color="transparent",
                                            corner_radius=0)
        self._loading_frame.pack(fill="x", pady=30, padx=60)

        self._loading_lbl = ctk.CTkLabel(
            self._loading_frame, text="⏳ 正在加载，请稍候...",
            text_color=COLORS["accent"],
            font=("Microsoft YaHei UI", 18, "bold"))
        self._loading_lbl.pack(pady=(0, 14))

        self._loading_rows = []
        for _ in range(self._loading_total):
            row = ctk.CTkFrame(self._loading_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            mark = ctk.CTkLabel(row, text="·", width=24, text_color=COLORS["text_faint"],
                                font=("Microsoft YaHei UI", 13, "bold"))
            mark.pack(side="left")
            txt = ctk.CTkLabel(row, text="", text_color=COLORS["text_dim"],
                               font=("Microsoft YaHei UI", 12))
            txt.pack(side="left")
            self._loading_rows.append((mark, txt))

        self._loading_bar = ctk.CTkProgressBar(self._loading_frame, height=10,
                                               corner_radius=R_BTN,
                                               fg_color=COLORS["bg"],
                                               progress_color=COLORS["accent"])
        self._loading_bar.pack(fill="x", pady=(16, 4))
        self._loading_bar.set(0.0)
        self._loading_pct = ctk.CTkLabel(self._loading_frame, text="0%",
                                         text_color=COLORS["text_faint"],
                                         font=("Microsoft YaHei UI", 11))
        self._loading_pct.pack()

        self._loading_cur = 0
        self._animate_loading()

    def _update_loading_stage(self, status, detail):
        if getattr(self, "_loading_rows", None) is None:
            return
        if status == "done":
            if self._loading_cur < len(self._loading_rows):
                mark, txt = self._loading_rows[self._loading_cur]
                mark.configure(text="✅", text_color=COLORS["ok"])
                txt.configure(text=detail, text_color=COLORS["text"])
                self._loading_cur += 1
            self._loading_done = self._loading_cur
            self._loading_bar.set(self._loading_done / self._loading_total)
            self._loading_pct.configure(text=f"{int(self._loading_done / self._loading_total * 100)}%")
        else:
            if self._loading_cur < len(self._loading_rows):
                mark, txt = self._loading_rows[self._loading_cur]
                txt.configure(text=detail, text_color=COLORS["text_dim"])

    def _animate_loading(self):
        if self._closing:
            return
        if getattr(self, "_loading_lbl", None) is None:
            return
        self._loading_dots = (self._loading_dots + 1) % 4
        dots = "." * self._loading_dots
        self._loading_lbl.configure(text=f"⏳ 正在加载，请稍候{dots}")
        self.root.after(400, self._animate_loading)

    def _hide_loading(self):
        if getattr(self, "_loading_frame", None) is not None:
            try:
                self._loading_frame.destroy()
            except Exception:
                pass
        self._loading_frame = None
        self._loading_lbl = None
        self._loading_rows = None
        self._loading_bar = None
        self._loading_pct = None
        self._append_banner()

    def _append_banner(self):
        self._append_msg("system", "—— 欢迎使用 情感助手 ——\n输入你的问题，我会基于资料库给你建议；也可以点击回答下方的相关推荐快速追问。")

    def _append_msg(self, role, content, suggestions=None, record=True):
        if record and role != "system":
            self._rendered.append((role, content, suggestions))

        if role == "system":
            wrap = ctk.CTkFrame(self.chat_area, fg_color="transparent", corner_radius=0)
            wrap.pack(fill="x", pady=12, anchor="center")
            ctk.CTkLabel(wrap, text=content, justify="center", wraplength=560,
                         text_color=COLORS["text_faint"],
                         font=("Microsoft YaHei UI", self.font_size - 1)).pack(pady=4)
            self._scroll_bottom()
            return

        align = "e" if role == "user" else "w"
        padx_side = 0.4 if role == "user" else 0.0
        wrap = ctk.CTkFrame(self.chat_area, fg_color="transparent", corner_radius=0)
        wrap.pack(fill="x", padx=0, pady=5, anchor=align)

        if role == "error":
            bub_fg = COLORS["danger"]
            bub_border = COLORS["danger"]
            bub_text = "#ffffff"
        elif role == "user":
            bub_fg = COLORS["user_bub"]
            bub_border = COLORS["user_bub"]
            bub_text = COLORS["on_accent"] if COLORS["user_bub"] == COLORS["accent"] else COLORS["text"]
        else:
            bub_fg = self.bubble_color
            bub_border = COLORS["border"]
            bub_text = self.text_color

        bub = ctk.CTkFrame(wrap, fg_color=bub_fg, corner_radius=R_CARD,
                           border_width=1, border_color=bub_border)
        bub.pack(fill="x", anchor=align, padx=(0, padx_side) if role == "user" else (padx_side, 0))

        label = ctk.CTkLabel(bub, text=content,
                             wraplength=500, justify="left",
                             text_color=bub_text,
                             font=("Microsoft YaHei UI", self.font_size))
        label.pack(fill="x", padx=12, pady=8)

        if role == "ai" and self.typewriter and record:
            label.configure(text="")
            self._start_typewriter(label, "", content)

        ts = ctk.CTkLabel(wrap, text=time.strftime("%H:%M"),
                          text_color=COLORS["text_faint"],
                          font=("Microsoft YaHei UI", 11))
        ts.pack(anchor=align, padx=6, pady=(1, 0))

        if role == "ai":
            act = ctk.CTkFrame(wrap, fg_color="transparent", corner_radius=0)
            act.pack(fill="x", padx=4, pady=(2, 0))
            ctk.CTkButton(act, text="有用", width=52, height=24, corner_radius=R_BTN,
                          fg_color=COLORS["bg"], hover_color=COLORS["hover"],
                          text_color=COLORS["text_faint"],
                          font=("Microsoft YaHei UI", 11),
                          command=lambda: self._save_feedback(content, 1)).pack(side="left", padx=(2, 2))
            ctk.CTkButton(act, text="没用", width=52, height=24, corner_radius=R_BTN,
                          fg_color=COLORS["bg"], hover_color=COLORS["danger"],
                          text_color=COLORS["text_faint"],
                          font=("Microsoft YaHei UI", 11),
                          command=lambda: self._save_feedback(content, 0)).pack(side="left", padx=2)
            ctk.CTkButton(act, text="复制", width=52, height=24, corner_radius=R_BTN,
                          fg_color=COLORS["bg"], hover_color=COLORS["hover"],
                          text_color=COLORS["text_faint"],
                          font=("Microsoft YaHei UI", 11),
                          command=lambda: self._copy_text(content)).pack(side="right", padx=(2, 2))

        if suggestions:
            rec = ctk.CTkFrame(wrap, fg_color="transparent", corner_radius=0)
            rec.pack(fill="x", padx=4, pady=(6, 2))
            ctk.CTkLabel(rec, text="相关推荐 · 点击发送", text_color=COLORS["text_faint"],
                         font=("Microsoft YaHei UI", 11)).pack(anchor="w", padx=2, pady=(0, 4))
            for i, s in enumerate(suggestions[:4]):
                ctk.CTkButton(rec, text=f"{i+1}.  {s}", height=32, corner_radius=R_BTN,
                              fg_color=COLORS["panel"], hover_color=COLORS["hover"],
                              border_width=1, border_color=COLORS["border"],
                              text_color=COLORS["text"],
                              font=("Microsoft YaHei UI", 11),
                              command=lambda q=s: self._send_recommend(q)).pack(
                                  fill="x", pady=3)

        self._scroll_bottom()

    def _append_error(self, text):
        self._append_msg("error", text)

    def _show_thinking(self):
        if getattr(self, "_thinking_frame", None) is not None:
            return
        self._thinking_frame = ctk.CTkFrame(self.chat_area, fg_color="transparent",
                                             corner_radius=0)
        self._thinking_frame.pack(fill="x", pady=8)

        box = ctk.CTkFrame(self._thinking_frame, fg_color=COLORS["panel"],
                           corner_radius=R_CARD, border_width=2, border_color=COLORS["accent"])
        box.pack(anchor="w")

        self._thinking_label = ctk.CTkLabel(
            box, text="⏳ 正在思考中...",
            text_color=COLORS["accent"],
            font=("Microsoft YaHei UI", 13, "bold"))
        self._thinking_label.pack(padx=20, pady=12)

        self._thinking_dots = 0
        self._animate_thinking()
        self._scroll_bottom()

    def _animate_thinking(self):
        if self._closing:
            return
        if getattr(self, "_thinking_label", None) is None:
            return
        self._thinking_dots = (self._thinking_dots + 1) % 4
        dots = "·" * self._thinking_dots
        try:
            self._thinking_label.configure(text=f"⏳ 正在思考中{dots}")
            self._scroll_bottom()
            self.root.after(400, self._animate_thinking)
        except Exception:
            pass

    def _hide_thinking(self):
        if getattr(self, "_thinking_frame", None) is not None:
            try:
                self._thinking_frame.destroy()
            except Exception:
                pass
        self._thinking_frame = None
        self._thinking_label = None

    def _scroll_bottom(self):
        def _do():
            try:
                self.chat_area._parent_canvas.update_idletasks()
                self.chat_area._parent_canvas.yview_moveto(1.0)
            except Exception:
                try:
                    self.chat_area._parent_canvas.yview_moveto(1.0)
                except Exception:
                    pass
        try:
            self.root.after(20, _do)
        except Exception:
            _do()

    def _start_typewriter(self, label, prefix, full_text, interval_ms=12, chunk=2):
        if self._closing:
            label.configure(text=prefix + full_text)
            return
        state = {"i": 0}
        label.configure(text=prefix)

        def _tick():
            if self._closing:
                return
            i = state["i"]
            if i >= len(full_text):
                label.configure(text=prefix + full_text)
                self._scroll_bottom()
                return
            i = min(i + chunk, len(full_text))
            state["i"] = i
            label.configure(text=prefix + full_text[:i])
            self._scroll_bottom()
            try:
                self.root.after(interval_ms, _tick)
            except Exception:
                label.configure(text=prefix + full_text)

        _tick()

    def _save_feedback(self, answer, good):
        try:
            data = []
            if FEEDBACK_FILE.exists():
                try:
                    data = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
                except Exception:
                    data = []
            data.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "good": bool(good),
                "answer": answer[:2000],
                "_author": "RoyShao",
                "_project": "AI-Smart-Emotion-Coach",
            })
            FEEDBACK_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
            self._set_status("已记录反馈 ✓" if good else "已记录反馈")
        except Exception:
            pass

    def _copy_text(self, text):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._set_status("已复制")
        except Exception:
            pass

    def _send_recommend(self, q):
        self.entry.delete(0, "end")
        self.entry.insert(0, q)
        self._send()

    def _rebuild_phrase_menu(self):
        try:
            self._phrase_menu.configure(values=self.phrases)
            if self.phrases:
                self._phrase_menu.set(self.phrases[0])
        except Exception:
            pass

    def _use_phrase(self, p):
        if p == "快捷短语 ▾":
            return
        self.entry.delete(0, "end")
        self.entry.insert(0, p)
        self.entry.focus_set()
        self._phrase_menu.set("快捷短语 ▾")

    def _edit_phrases(self):
        win = ctk.CTkToplevel(self.root)
        win.title("编辑快捷短语")
        win.resizable(False, False)
        self._center_win(win, 460, 420)

        ctk.CTkLabel(win, text="每行一个短语：", text_color=COLORS["text_dim"]).pack(anchor="w", padx=16, pady=(14, 4))
        txt = ctk.CTkTextbox(win, height=260, corner_radius=R_CARD,
                             fg_color=COLORS["panel"], font=("Microsoft YaHei UI", 12))
        txt.pack(fill="both", expand=True, padx=16, pady=4)
        txt.insert("1.0", "\n".join(self.phrases))

        def save():
            self.phrases = [l.strip() for l in txt.get("1.0", "end").splitlines() if l.strip()]
            if not self.phrases:
                self.phrases = ["帮我分析一下我的情况"]
            self._save_phrases()
            self._rebuild_phrase_menu()
            win.destroy()

        ctk.CTkButton(win, text="保存", command=save, height=34, corner_radius=R_BTN,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_h"],
                      text_color=COLORS["on_accent"]).pack(padx=16, pady=12, fill="x")

    def _init_engine(self):
        try:
            eng = SmartQA(progress_callback=lambda s, d: self._msg_queue.put(("loading", s, d)))
            self._msg_queue.put(("ready", eng))
        except Exception as e:
            logger.exception("初始化失败")
            self._msg_queue.put(("init_failed", str(e)))

    def _send(self):
        if self._busy or self.engine is None:
            return
        q = self.entry.get().strip()
        if not q:
            return
        self.entry.delete(0, "end")
        self._append_msg("user", q)
        self._log_entry("用户", q)

        self._busy = True
        self.send_btn.configure(state="disabled")
        self._set_status("思考中...")
        self._show_thinking()
        threading.Thread(target=self._worker, args=(q,), daemon=True).start()

    def _worker(self, q):
        try:
            answer, suggestions = self.engine.answer(q)
            self._msg_queue.put(("answer", answer, suggestions))
        except Exception as e:
            logger.exception("问答失败")
            self._msg_queue.put(("error", str(e)))

    def _poll_queue(self):
        if self._closing:
            return
        while True:
            try:
                item = self._msg_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_item(item)
        try:
            self.root.after(50, self._poll_queue)
        except tk.TclError:
            pass

    def _handle_item(self, item):
        kind = item[0]
        if kind == "loading":
            status = item[1] if len(item) > 1 else "work"
            detail = item[2] if len(item) > 2 else ""
            self._update_loading_stage(status, detail)
        elif kind == "ready":
            self.engine = item[1]
            self._hide_loading()
            self._set_status("就绪")
            self.send_btn.configure(state="normal")
        elif kind == "init_failed":
            self._hide_loading()
            self._set_status("初始化失败")
            self._append_error(item[1])
        elif kind == "answer":
            self._hide_thinking()
            _, answer, suggestions = item
            self._append_msg("ai", answer, suggestions)
            self._log_entry("助手", answer)
            self._set_status("就绪")
            self.send_btn.configure(state="normal")
            self._busy = False
            self._scroll_bottom()
        elif kind == "error":
            self._hide_thinking()
            self._append_error(item[1])
            self._set_status("就绪")
            self.send_btn.configure(state="normal")
            self._busy = False

    def _new_session(self):
        if self._busy:
            return
        if self.engine:
            self.engine.clear_history()
        self._hide_stage_panel()
        self._log = []
        self._rendered = []
        self._log_file = None
        for w in self.chat_area.winfo_children():
            w.destroy()
        self._append_banner()
        self._refresh_history_list()
        self._set_status("新会话")

    def _clear_chat(self):
        self._new_session()

    def _log_entry(self, role, content):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log.append((ts, role, content))
        if self._log_file is None:
            try:
                self._log_file = self._log_dir / f"会话_{time.strftime('%Y%m%d_%H%M%S')}.txt"
                self._log_file.write_text("", encoding="utf-8")
            except Exception:
                self._log_file = None
        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] {role}：{content}\n")
            except Exception:
                pass

    def _set_status(self, text):
        self.status_var.set(text)

    @staticmethod
    def _parse_log_file(path):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return []
        pattern = re.compile(r'(?m)^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (用户|助手)：')
        matches = list(pattern.finditer(text))
        entries = []
        for i, m in enumerate(matches):
            ts, role = m.group(1), m.group(2)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if content:
                entries.append((ts, role, content))
        return entries

    def _view_history(self):
        win = ctk.CTkToplevel(self.root)
        win.title("聊天记录")
        win.minsize(520, 400)
        self._center_win(win, 760, 580)

        bar = ctk.CTkFrame(win, fg_color="transparent", corner_radius=0)
        bar.pack(fill="x", padx=12, pady=(12, 6))

        def get_sessions():
            try:
                all_p = sorted(self._log_dir.glob("会话_*.txt")) if self._log_dir.exists() else []
                return [p for p in all_p if p.stat().st_size > 0]
            except Exception:
                return []

        ctk.CTkLabel(bar, text="选择会话：", text_color=COLORS["text"],
                     font=("Microsoft YaHei UI", 12)).pack(side="left")
        listbox = tk.Listbox(bar, width=34, height=6,
                             bg=COLORS["panel"], fg=COLORS["text"],
                             selectbackground=COLORS["accent"], selectforeground=COLORS["on_accent"],
                             highlightthickness=0, borderwidth=0,
                             font=("Microsoft YaHei UI", 11), activestyle="none")
        listbox.pack(side="left", padx=8)

        detail = ctk.CTkTextbox(win, wrap="word", corner_radius=R_CARD,
                                fg_color=COLORS["panel"], font=("Microsoft YaHei UI", 12))
        detail.pack(fill="both", expand=True, padx=12, pady=6)

        def load_detail(path):
            detail.delete("1.0", "end")
            if not path:
                detail.insert("end", "（请选择左侧会话查看内容）")
                return
            entries = self._parse_log_file(path)
            if not entries:
                detail.insert("end", "（该会话无内容）")
                return
            detail.insert("end", f"━━━ {path.stem} ━━━\n\n")
            for ts, role, content in entries:
                detail.insert("end", f"[{ts}] {role}：\n{content}\n\n")

        def refresh():
            listbox.delete(0, "end")
            for path in get_sessions():
                preview = self._session_preview(path)
                listbox.insert("end", f"{path.stem}  {preview}")
            if listbox.size() > 0:
                listbox.selection_set(0)
                listbox.event_generate("<<ListboxSelect>>")

        def on_select(_evt=None):
            sel = listbox.curselection()
            if sel:
                paths = get_sessions()
                if 0 <= sel[0] < len(paths):
                    load_detail(paths[sel[0]])

        listbox.bind("<<ListboxSelect>>", on_select)
        listbox.bind("<Double-Button-1>", lambda e: load_detail(
            get_sessions()[listbox.curselection()[0]] if listbox.curselection() else None))

        btn_frame = ctk.CTkFrame(bar, fg_color="transparent", corner_radius=0)
        btn_frame.pack(side="right")

        def rename_session():
            sel = listbox.curselection()
            if not sel:
                self._set_status("请先选择要重命名的会话")
                return
            path = get_sessions()[sel[0]]
            dlg = ctk.CTkToplevel(self.root)
            dlg.title("重命名会话")
            dlg.resizable(False, False)
            self._center_win(dlg, 360, 150)
            ctk.CTkLabel(dlg, text="新名称（无需扩展名）：", text_color=COLORS["text"]).pack(anchor="w", padx=16, pady=(14, 4))
            ent = ctk.CTkEntry(dlg, height=34, corner_radius=R_BTN, fg_color=COLORS["panel"])
            ent.pack(fill="x", padx=16, pady=4)
            ent.insert(0, path.stem)

            def do_rename():
                new_name = ent.get().strip()
                if not new_name:
                    return
                new_path = self._log_dir / f"{new_name}.txt"
                try:
                    path.rename(new_path)
                    dlg.destroy()
                    refresh()
                    self._refresh_history_list()
                    self._set_status("已重命名")
                except Exception as e:
                    self._set_status(f"重命名失败：{e}")
            ent.bind("<Return>", lambda e: do_rename())
            ctk.CTkButton(dlg, text="确定", command=do_rename, height=32, corner_radius=R_BTN,
                          fg_color=COLORS["accent"], hover_color=COLORS["accent_h"],
                          text_color=COLORS["on_accent"]).pack(padx=16, pady=10, fill="x")

        def delete_session():
            sel = listbox.curselection()
            if not sel:
                self._set_status("请先选择要删除的会话")
                return
            path = get_sessions()[sel[0]]
            if not tk.messagebox.askyesno("确认删除",
                                          f"确定删除会话「{path.stem}」吗？\n删除后不可恢复。"):
                return
            try:
                path.unlink(missing_ok=True)
                refresh()
                self._refresh_history_list()
                self._set_status("会话已删除")
            except Exception as e:
                self._set_status(f"删除失败：{e}")

        ctk.CTkButton(btn_frame, text="重命名", width=70, height=32, corner_radius=R_BTN,
                      fg_color=COLORS["panel"], hover_color=COLORS["accent"],
                      text_color=COLORS["text"], command=rename_session).pack(side="left", padx=3)
        ctk.CTkButton(btn_frame, text="删除", width=70, height=32, corner_radius=R_BTN,
                      fg_color=COLORS["panel"], hover_color=COLORS["danger"],
                      text_color=COLORS["danger"], command=delete_session).pack(side="left", padx=3)

        ctk.CTkLabel(win, text=f"存档目录：{self._log_dir}", text_color=COLORS["text_dim"],
                     font=("Microsoft YaHei UI", 10)).pack(side="bottom", fill="x", padx=12, pady=6)

        refresh()

    def _session_preview(self, path):
        try:
            entries = self._parse_log_file(path)
            for ts, role, content in entries:
                if role == "用户":
                    return content[:18] + ("…" if len(content) > 18 else "")
        except Exception:
            pass
        return ""

    @staticmethod
    def _color_name(color, presets):
        for name, val in presets.items():
            if val == color:
                return name
        return list(presets.keys())[0]

    def _open_settings(self):
        win = ctk.CTkToplevel(self.root)
        win.title("设置")
        win.resizable(False, False)
        self._center_win(win, 520, 620)

        ctk.CTkLabel(win, text="⚙ 设置", font=("Microsoft YaHei UI", 18, "bold"),
                     text_color=COLORS["accent"]).pack(anchor="w", padx=20, pady=(16, 8))

        info = ctk.CTkFrame(win, fg_color=COLORS["panel"], corner_radius=R_CARD)
        info.pack(fill="x", padx=20, pady=6)
        chat_m = self.engine.chat_model if self.engine else CHAT_MODEL
        embed_m = self.engine.embed_model if self.engine else "检测中"
        rerank_s = ("已启用" if (self.engine and self.engine.reranker)
                    else "未启用（距离排序）")
        ctk.CTkLabel(info, text=f"聊天模型：{chat_m}\n嵌入模型：{embed_m}\n重排序：{rerank_s}",
                     justify="left", text_color=COLORS["text_dim"],
                     font=("Microsoft YaHei UI", 12)).pack(anchor="w", padx=14, pady=10)

        param = ctk.CTkFrame(win, fg_color=COLORS["panel"], corner_radius=R_CARD)
        param.pack(fill="x", padx=20, pady=6)

        PARAM_HELP = {
            "温度": "控制回答的随机性/创造性。\n\n• 越低(如 0.2)：回答更稳定、严谨、保守\n• 越高(如 1.2)：回答更灵活、有创意\n• 推荐区间：0.4~0.8\n• 项目默认：0.6",
            "最大字数": "单次回答允许生成的最大字数上限。\n\n• 太小(如 200)：回答可能不完整\n• 太大(如 800)：回答冗长、等待更久\n• 项目默认：500",
            "上下文条数": "每次回答参考的知识库条目数量。\n\n• 越多：参考信息更全，但可能引入噪音\n• 越少：回答更聚焦，但可能漏信息\n• 项目默认：5",
        }

        def _slider_row(parent, label, low, high, var, fmt, help_text, steps):
            row = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0)
            row.pack(fill="x", padx=12, pady=4)
            ctk.CTkLabel(row, text=label, text_color=COLORS["text"],
                         font=("Microsoft YaHei UI", 12)).pack(side="left")
            ctk.CTkButton(row, text="?", width=22, height=22, corner_radius=11,
                          fg_color=COLORS["bg"], hover_color=COLORS["accent"],
                          text_color=COLORS["text_dim"],
                          font=("Microsoft YaHei UI", 11, "bold"),
                          command=lambda h=help_text: tk.messagebox.showinfo(
                              "参数说明", h, parent=win)).pack(side="left", padx=4)
            val = ctk.CTkLabel(row, text=str(var.get()), width=40,
                               text_color=COLORS["accent"],
                               font=("Microsoft YaHei UI", 12, "bold"))
            val.pack(side="right")
            slider = ctk.CTkSlider(row, from_=low, to=high, number_of_steps=steps,
                                   variable=var, width=180,
                                   command=lambda v, l=val, f=fmt: l.configure(text=f(v)))
            slider.pack(side="right", padx=8)

        self._tmp_temp = ctk.DoubleVar(value=self.temperature)
        self._tmp_pred = ctk.IntVar(value=self.num_predict)
        self._tmp_ctx = ctk.IntVar(value=self.ctx_count)
        _slider_row(param, "温度 (0~1.5)", 0, 1.5, self._tmp_temp,
                    lambda v: f"{float(v):.2f}", PARAM_HELP["温度"], steps=30)
        _slider_row(param, "最大字数", 200, 1000, self._tmp_pred,
                    lambda v: str(int(v)), PARAM_HELP["最大字数"], steps=80)
        _slider_row(param, "上下文条数", 3, 8, self._tmp_ctx,
                    lambda v: str(int(v)), PARAM_HELP["上下文条数"], steps=5)

        def reset_params():
            self._tmp_temp.set(TEMPERATURE)
            self._tmp_pred.set(NUM_PREDICT)
            self._tmp_ctx.set(CTX_COUNT)

        ctk.CTkButton(param, text="↺ 恢复初始状态（温度 0.6 / 字数 500 / 条目 5）",
                      command=reset_params, height=30, corner_radius=R_BTN,
                      fg_color=COLORS["bg"], hover_color=COLORS["accent"],
                      text_color=COLORS["accent"], font=("Microsoft YaHei UI", 11)
                      ).pack(fill="x", padx=12, pady=(2, 10))

        appear = ctk.CTkFrame(win, fg_color=COLORS["panel"], corner_radius=R_CARD)
        appear.pack(fill="x", padx=20, pady=6)
        ctk.CTkLabel(appear, text="字体大小", text_color=COLORS["text"],
                     font=("Microsoft YaHei UI", 12)).pack(side="left", padx=12, pady=8)
        self._tmp_font = ctk.IntVar(value=self.font_size)
        ctk.CTkOptionMenu(appear, values=["11", "12", "13", "14", "15", "16"],
                          variable=ctk.StringVar(value=str(self.font_size)),
                          width=80, corner_radius=R_BTN,
                          command=lambda v: self._tmp_font.set(int(v))).pack(side="right", padx=12)
        self._tmp_typewriter = ctk.BooleanVar(value=self.typewriter)
        ctk.CTkSwitch(appear, text="打字机效果", variable=self._tmp_typewriter,
                      text_color=COLORS["text"],
                      font=("Microsoft YaHei UI", 12)).pack(side="left", padx=20, pady=8)

        color_panel = ctk.CTkFrame(win, fg_color=COLORS["panel"], corner_radius=R_CARD)
        color_panel.pack(fill="x", padx=20, pady=6)

        color_presets = {
            "mocha 白": "#CDD6F4",
            "纯白": "#ffffff",
            "淡蓝": "#89B4FA",
            "薄荷绿": "#A6E3A1",
            "樱花粉": "#F5C2E7",
        }
        bubble_presets = {
            "深灰紫": "#313244",
            "暖棕": "#2e2620",
            "深蓝": "#24344d",
            "深紫": "#35294d",
            "深绿": "#24332b",
            "深红": "#42272a",
        }

        row1 = ctk.CTkFrame(color_panel, fg_color="transparent", corner_radius=0)
        row1.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(row1, text="字体颜色", text_color=COLORS["text"],
                     font=("Microsoft YaHei UI", 12)).pack(side="left")
        self._tmp_text_color = ctk.StringVar(value=self._color_name(self.text_color, color_presets))
        self._tmp_text_color_val = self.text_color
        ctk.CTkOptionMenu(row1, values=list(color_presets.keys()),
                          variable=self._tmp_text_color, width=100, corner_radius=R_BTN,
                          command=lambda v: setattr(self, "_tmp_text_color_val",
                                                    color_presets[v])).pack(side="right", padx=12)

        row2 = ctk.CTkFrame(color_panel, fg_color="transparent", corner_radius=0)
        row2.pack(fill="x", padx=12, pady=(2, 8))
        ctk.CTkLabel(row2, text="AI气泡颜色", text_color=COLORS["text"],
                     font=("Microsoft YaHei UI", 12)).pack(side="left")
        self._tmp_bubble_color = ctk.StringVar(value=self._color_name(self.bubble_color, bubble_presets))
        self._tmp_bubble_color_val = self.bubble_color
        ctk.CTkOptionMenu(row2, values=list(bubble_presets.keys()),
                          variable=self._tmp_bubble_color, width=100, corner_radius=R_BTN,
                          command=lambda v: setattr(self, "_tmp_bubble_color_val",
                                                    bubble_presets[v])).pack(side="right", padx=12)

        def apply_settings():
            self.temperature = round(float(self._tmp_temp.get()), 2)
            self.num_predict = int(self._tmp_pred.get())
            self.ctx_count = int(self._tmp_ctx.get())
            self.font_size = int(self._tmp_font.get())
            self.typewriter = bool(self._tmp_typewriter.get())
            self._ui_font = ("Microsoft YaHei UI", self.font_size)
            if self.engine:
                try:
                    self.engine.set_params(temperature=self.temperature,
                                           num_predict=self.num_predict,
                                           ctx_count=self.ctx_count)
                except Exception:
                    pass
            tc = getattr(self, "_tmp_text_color_val", None)
            bc = getattr(self, "_tmp_bubble_color_val", None)
            if tc:
                self.text_color = tc
            if bc:
                self.bubble_color = bc
            win.destroy()
            self._set_status("设置已保存")

        ctk.CTkButton(win, text="保存设置", command=apply_settings, height=40,
                      corner_radius=R_BTN,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_h"],
                      text_color=COLORS["on_accent"], font=("Microsoft YaHei UI", 13, "bold")).pack(
                          fill="x", padx=20, pady=14)

        ctk.CTkButton(win, text="清空全部聊天记录", command=self._clear_all_logs,
                      fg_color=COLORS["panel"], hover_color=COLORS["danger"],
                      text_color=COLORS["danger"], height=32, corner_radius=R_BTN,
                      font=("Microsoft YaHei UI", 12)).pack(fill="x", padx=20, pady=(0, 18))

    def _clear_all_logs(self):
        if not tk.messagebox.askyesno("确认清空",
                                      "确定清空全部聊天记录吗？\n所有会话文件将被永久删除，不可恢复！"):
            return
        try:
            count = 0
            for path in self._log_dir.glob("会话_*.txt"):
                path.unlink(missing_ok=True)
                count += 1
            self._refresh_history_list()
            self._set_status(f"已清空 {count} 个会话记录")
        except Exception:
            self._set_status("清空失败")

    def _on_close(self):
        self._closing = True
        self._hide_stage_panel()
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
        self.root.destroy()


# ============================================================
# 四、程序入口
# ============================================================
def main():
    if ctk is None:
        logger.error("customtkinter 未安装，请执行: pip install customtkinter")
        return

    root = ctk.CTk()

    # 关闭 PyInstaller 原生闪屏（如果存在）
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass

    def _safe_callback_exc(exc, val, tb):
        if isinstance(val, tk.TclError):
            msg = str(val)
            if ("bad window path" in msg
                    or "invalid command name" in msg
                    or "application has been destroyed" in msg):
                return
        import traceback
        traceback.print_exception(exc, val, tb)

    root.report_callback_exception = _safe_callback_exc

    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()