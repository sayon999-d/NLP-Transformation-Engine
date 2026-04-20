from __future__ import annotations

import logging
import os
import re
import textwrap
import time
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _load_streamlit_secrets():
    try:
        for key in ("GROQ_API_KEY", "LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("nlp_engine")


class NLPEngine:

    TONE_RANGES = [
        (0, 20, 0.15, "Minimal Rewrite"),
        (21, 50, 0.4, "Light Naturalization"),
        (51, 80, 0.65, "Moderate Humanization"),
        (81, 100, 0.9, "Strong Rewriting"),
    ]

    COMPRESSION = {
        "short": (0.20, 0.30),
        "medium": (0.40, 0.60),
        "long": (0.70, 0.80),
    }

    LONG_TEXT_THRESHOLD = 1500

    def __init__(self, llm=None):
        self._llm = llm

    def set_llm(self, llm) -> None:
        self._llm = llm

    def transform(
        self,
        text: str,
        mode: str = "humanize",
        style: str = "professional",
        summary_length: str = "medium",
        humanization_level: float = 0.5,
        auto_clipboard: bool = False,
        clipboard_watcher: bool = False,
        realtime_mode: bool = False,
        tone_slider: int | None = None,
    ) -> dict[str, Any]:
        if not text or not text.strip():
            return {"error": "Empty input text", "output": ""}

        analysis = self.analyze_text(text)

        if tone_slider is not None:
            humanization_level = self.map_tone_slider(tone_slider)
            analysis["tone_slider_applied"] = tone_slider
            analysis["humanization_level"] = humanization_level

        is_long = len(text) > self.LONG_TEXT_THRESHOLD
        pipeline_order = self._decide_pipeline_order(mode, is_long)
        analysis["pipeline_order"] = pipeline_order

        processed = text
        stages_applied = []

        for stage in pipeline_order:
            if stage == "summarize":
                processed = self._run_summarization(
                    processed, summary_length, style,
                    realtime_mode=realtime_mode,
                )
                stages_applied.append("summarize")
            elif stage == "humanize":
                processed = self._run_humanization(
                    processed, humanization_level, style,
                    realtime_mode=realtime_mode,
                    clipboard_watcher=clipboard_watcher,
                )
                stages_applied.append("humanize")

        processed = self._post_process(processed)

        if auto_clipboard:
            processed = processed.rstrip() + "\n\n[CLIPBOARD_COPY_READY]"

        readability = self.compute_readability(processed)

        return {
            "output": processed,
            "analysis": analysis,
            "readability": readability,
            "stages_applied": stages_applied,
            "input_length": len(text),
            "output_length": len(processed),
            "compression_ratio": round(len(processed) / max(len(text), 1), 2),
            "mode": mode,
            "style": style,
            "humanization_level": humanization_level,
        }

    def analyze_text(self, text: str) -> dict[str, Any]:
        words = text.split()
        sentences = self._split_sentences(text)
        word_count = len(words)
        sentence_count = max(len(sentences), 1)

        if len(words) > 1:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            unique_bigrams = set(bigrams)
            repetition_ratio = 1 - (len(unique_bigrams) / max(len(bigrams), 1))
        else:
            repetition_ratio = 0

        avg_sentence_len = word_count / sentence_count
        verbosity = (
            "high" if avg_sentence_len > 25
            else "medium" if avg_sentence_len > 15
            else "low"
        )

        ai_patterns = [
            r'\bfurthermore\b', r'\bmoreover\b', r'\bin conclusion\b',
            r'\bit is worth noting\b', r'\bit\'s important to note\b',
            r'\bdelve\b', r'\btap into\b', r'\bunlock\b',
            r'\beverchanging\b', r'\bever-changing\b',
            r'\blandscape\b', r'\brealm\b', r'\bseamless(?:ly)?\b',
            r'\bleverag\w+\b', r'\bpivotal\b', r'\brobust\b',
            r'\bcutting[\s-]edge\b', r'\bparadigm\b',
        ]
        ai_pattern_count = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in ai_patterns
        )
        ai_likelihood = min(ai_pattern_count / max(sentence_count, 1) * 100, 100)

        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_sentence_length": round(avg_sentence_len, 1),
            "verbosity": verbosity,
            "repetition_ratio": round(repetition_ratio, 3),
            "ai_pattern_count": ai_pattern_count,
            "ai_likelihood_pct": round(ai_likelihood, 1),
            "text_is_long": len(text) > self.LONG_TEXT_THRESHOLD,
        }

    def compute_readability(self, text: str) -> dict[str, Any]:
        if not text or not text.strip():
            return {}

        sentences = self._split_sentences(text)
        words = text.split()
        syllable_count = sum(self._count_syllables(w) for w in words)

        wc = len(words)
        sc = max(len(sentences), 1)
        awps = wc / sc
        asyl = syllable_count / max(wc, 1)

        fk_grade = 0.39 * awps + 11.8 * asyl - 15.59
        fre = 206.835 - 1.015 * awps - 84.6 * asyl
        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
        fog = 0.4 * (awps + 100 * complex_words / max(wc, 1))

        return {
            "word_count": wc,
            "sentence_count": sc,
            "avg_words_per_sentence": round(awps, 1),
            "flesch_kincaid_grade": round(max(fk_grade, 0), 1),
            "flesch_reading_ease": round(min(max(fre, 0), 100), 1),
            "gunning_fog_index": round(max(fog, 0), 1),
            "complexity_rating": self._complexity_label(fk_grade),
        }

    @staticmethod
    def map_tone_slider(slider: int) -> float:
        slider = max(0, min(100, slider))
        for low, high, level, _ in NLPEngine.TONE_RANGES:
            if low <= slider <= high:
                return level
        return 0.5

    @staticmethod
    def tone_label(slider: int) -> str:
        slider = max(0, min(100, slider))
        for low, high, _, label in NLPEngine.TONE_RANGES:
            if low <= slider <= high:
                return label
        return "Moderate Humanization"

    def _decide_pipeline_order(self, mode: str, is_long: bool) -> list[str]:
        if mode == "summarize":
            return ["summarize"]
        elif mode == "humanize":
            return ["humanize"]
        elif mode == "both":
            return ["summarize", "humanize"] if is_long else ["humanize", "summarize"]
        return ["humanize"]

    def _run_summarization(self, text, summary_length, style, realtime_mode=False):
        if self._llm is None:
            raise RuntimeError("No LLM configured. Set GROQ_API_KEY in .env")

        compression = self.COMPRESSION.get(summary_length, (0.40, 0.60))
        target_pct = f"{int(compression[0]*100)}-{int(compression[1]*100)}%"

        style_guide = {
            "casual": "Use a relaxed, conversational tone. Keep it easy to read.",
            "professional": "Use a structured, neutral, and clear tone.",
            "academic": "Use formal, precise, and technical language.",
            "concise": "Use dense, minimal wording. Every word must earn its place.",
        }

        prompt = textwrap.dedent(f"""\
            You are a precise text summarizer. Compress the following text to approximately {target_pct} of its original length.

            Style: {style_guide.get(style, style_guide['professional'])}

            Rules:
            - Preserve all key semantic meaning and logical structure
            - Do NOT introduce new facts or opinions
            - Do NOT distort the original meaning
            - Maintain coherence and flow
            {"- Prioritize speed over perfection. Keep original structure." if realtime_mode else ""}

            TEXT TO SUMMARIZE:
            {text}

            Return ONLY the summarized text. No explanations, no metadata.
        """)

        return self._call_llm(prompt)

    def _run_humanization(self, text, level, style, realtime_mode=False, clipboard_watcher=False):
        if self._llm is None:
            raise RuntimeError("No LLM configured. Set GROQ_API_KEY in .env")

        intensity = (
            "Make only minimal edits - fix awkward phrasing but preserve most of the original."
            if level < 0.3 else
            "Apply light naturalization - smooth transitions, reduce robotic phrasing."
            if level < 0.5 else
            "Apply moderate humanization - rewrite for natural flow, varied sentence structure."
            if level < 0.8 else
            "Apply strong human-like rewriting - make it sound like a skilled human writer."
        )

        style_guide = {
            "casual": "Conversational, relaxed, and highly readable. Grade 6-8 level.",
            "professional": "Structured, neutral, and clear. Grade 9-12 level.",
            "academic": "Formal, precise, and technical. Higher education level.",
            "concise": "Dense and minimal. Every word counts.",
        }

        prompt = textwrap.dedent(f"""\
            You are an expert text humanizer. Transform the following text to sound naturally human-written.

            Intensity: {intensity}
            Style: {style_guide.get(style, style_guide['professional'])}
            Humanization Level: {level:.1f}/1.0

            AI-Detection Resistance Rules:
            - Vary sentence openings - never start consecutive sentences the same way
            - Avoid repetitive syntax patterns
            - Introduce natural rhythm and phrasing variation
            - Avoid overused AI phrases: "furthermore", "moreover", "it's worth noting", "delve",
              "leverage", "robust", "cutting-edge", "paradigm", "landscape", "realm", "seamlessly"
            - Use contractions where natural
            - Mix short and long sentences

            Post-Processing Rules:
            - Ensure grammatical correctness
            - Maintain coherence and original meaning
            - Do NOT introduce new facts
            - Do NOT distort meaning
            - If the text is already high-quality, avoid over-editing
            - If highly technical, preserve terminology
            - If very short, avoid unnecessary expansion
            {"- Perform minimal-latency transformation. Prioritize partial refinement over full rewriting." if realtime_mode else ""}
            {"- Output must be clean and ready for clipboard. No commentary." if clipboard_watcher else ""}

            TEXT TO HUMANIZE:
            {text}

            Return ONLY the humanized text. No explanations, no metadata.
        """)

        return self._call_llm(prompt)

    def _post_process(self, processed: str) -> str:
        processed = processed.strip()
        meta_patterns = [
            r'^(?:Here(?:\'s| is) (?:the|your) (?:humanized|summarized|processed|rewritten) (?:text|version|output):?\s*\n?)',
            r'^(?:Sure[,!]?\s*(?:here(?:\'s| is))?.*?:\s*\n?)',
            r'^(?:Certainly[,!]?\s*(?:here(?:\'s| is))?.*?:\s*\n?)',
        ]
        for pattern in meta_patterns:
            processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)
        return processed.strip()

    def _call_llm(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=(
                "You are a precise text transformation engine. "
                "Follow instructions exactly. Return ONLY the processed text. "
                "No explanations, no labels, no metadata."
            )),
            HumanMessage(content=prompt),
        ]

        try:
            response = self._llm.invoke(messages)
            return response.content.strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise RuntimeError(f"LLM call failed: {exc}") from exc

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower().strip(".,!?;:'\"()-")
        if not word:
            return 0
        if len(word) <= 3:
            return 1
        if word.endswith('e'):
            word = word[:-1]
        count = len(re.findall(r'[aeiouy]+', word))
        return max(count, 1)

    @staticmethod
    def _complexity_label(grade: float) -> str:
        if grade <= 6:
            return "Very Easy"
        elif grade <= 8:
            return "Easy"
        elif grade <= 10:
            return "Moderate"
        elif grade <= 12:
            return "Difficult"
        elif grade <= 14:
            return "Very Difficult"
        return "Expert"


def create_llm():
    _load_streamlit_secrets()
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or ""
    base_url = os.getenv("LLM_BASE_URL", "")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    if not api_key and provider != "ollama":
        raise RuntimeError(f"No API key found. Set GROQ_API_KEY in Streamlit secrets or .env")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider in ("openai", "openai_compatible"):
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": model or "gpt-4o",
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    elif provider in ("anthropic", "claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-pro",
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    elif provider == "together":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "meta-llama/Llama-3-70b-chat-hf",
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=model or "mistral-large-latest",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or "llama3.2:3b",
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
            num_predict=max_tokens,
        )
    else:
        logger.warning("Unknown provider '%s', falling back to Groq", provider)
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def get_engine() -> NLPEngine:
    if "nlp_engine" in st.session_state and st.session_state.nlp_engine._llm is not None:
        return st.session_state.nlp_engine
    try:
        llm = create_llm()
        engine = NLPEngine(llm=llm)
        logger.info("NLP Engine initialized (provider=%s, model=%s)",
                     os.getenv("LLM_PROVIDER", "groq"),
                     os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
        st.session_state.nlp_engine = engine
        return engine
    except Exception as exc:
        logger.error("LLM init failed: %s", exc)
        return NLPEngine()


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .hero-header {
        background: #000000;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
        position: relative;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
        position: relative;
    }

    .stage-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .stage-humanize {
        background: rgba(255, 255, 255, 0.08);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .stage-summarize {
        background: rgba(255, 255, 255, 0.08);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .stage-arrow {
        color: rgba(255, 255, 255, 0.3);
        font-size: 1.1rem;
    }

    .metric-card {
        background: #000000;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: rgba(255, 255, 255, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
        font-weight: 600;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-connected {
        background: rgba(255, 255, 255, 0.08);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .status-disconnected {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ffffff;
        animation: pulse-dot 2s ease-in-out infinite;
    }

    .pulse-dot-red {
        background: #888888;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.3); }
        50% { opacity: 0.6; box-shadow: 0 0 0 8px rgba(255, 255, 255, 0); }
    }

    .output-box {
        background: #000000;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        font-size: 0.95rem;
        line-height: 1.75;
        color: #e0e0e0;
        white-space: pre-wrap;
        word-wrap: break-word;
        min-height: 200px;
        font-family: 'Inter', sans-serif;
    }

    .gradient-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        border: none;
        margin: 1.5rem 0;
    }

    .tone-label-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.08);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stTextArea textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        line-height: 1.7 !important;
        border-radius: 12px !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
"""


def main():
    st.set_page_config(
        page_title="NLP Transformation Engine",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    engine = get_engine()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "output_text" not in st.session_state:
        st.session_state.output_text = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    llm_ready = engine._llm is not None
    provider_name = os.getenv("LLM_PROVIDER", "groq")
    model_name = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")

    status_html = (
        f'<span class="status-pill status-connected"><span class="pulse-dot"></span> {provider_name} / {model_name}</span>'
        if llm_ready else
        '<span class="status-pill status-disconnected"><span class="pulse-dot pulse-dot-red"></span> LLM Offline</span>'
    )

    st.markdown(f"""
        <div class="hero-header">
            <p class="hero-title">NLP Transformation Engine</p>
            <p class="hero-subtitle">Humanization / Summarization / Readability Optimization / Clipboard Automation</p>
            <div style="margin-top: 1rem;">{status_html}</div>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Pipeline Configuration")
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        mode = st.radio(
            "**Processing Mode**",
            options=["humanize", "summarize", "both"],
            format_func=lambda x: {"humanize": "Humanize", "summarize": "Summarize", "both": "Both"}[x],
            help="Select the transformation pipeline to apply.",
        )

        st.markdown("")

        style = st.radio(
            "**Writing Style**",
            options=["casual", "professional", "academic", "concise"],
            index=1,
            format_func=lambda x: {
                "casual": "Casual",
                "professional": "Professional",
                "academic": "Academic",
                "concise": "Concise",
            }[x],
        )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown("**Transformation Intensity**")
        tone_slider = st.slider(
            "Tone",
            min_value=0, max_value=100, value=50,
            label_visibility="collapsed",
        )
        tone_lbl = NLPEngine.tone_label(tone_slider)
        st.markdown(f'<span class="tone-label-badge">{tone_slider} - {tone_lbl}</span>', unsafe_allow_html=True)

        st.markdown("")

        if mode in ("summarize", "both"):
            summary_length = st.select_slider(
                "**Summary Length**",
                options=["short", "medium", "long"],
                value="medium",
                format_func=lambda x: {"short": "Short (20-30%)", "medium": "Medium (40-60%)", "long": "Long (70-80%)"}[x],
            )
        else:
            summary_length = "medium"

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown("**Options**")
        auto_clipboard = st.checkbox("Auto-copy to clipboard", value=False)
        realtime_mode = st.checkbox("Real-time mode (faster, less thorough)", value=False)
        clipboard_watcher = st.checkbox("Clipboard watcher mode", value=False)

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        if st.session_state.history:
            with st.expander(f"History ({len(st.session_state.history)})"):
                for i, h in enumerate(reversed(st.session_state.history[-10:])):
                    ts = h.get("timestamp", "")
                    m = h.get("mode", "?")
                    wc = h.get("input_words", 0)
                    st.caption(f"#{len(st.session_state.history) - i} - {m} / {wc} words / {ts}")

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### Input Text")
        input_text = st.text_area(
            "Paste or type your text",
            height=320,
            placeholder="Paste AI-generated text here...\n\nThe engine will analyze it for AI patterns, verbosity, and readability, then transform it based on your pipeline configuration.",
            label_visibility="collapsed",
            key="input_text_area",
        )

        if input_text and input_text.strip():
            analysis = engine.analyze_text(input_text)
            readability = engine.compute_readability(input_text)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Words", f"{analysis['word_count']:,}")
            c2.metric("Sentences", analysis["sentence_count"])
            c3.metric("AI Likelihood", f"{analysis['ai_likelihood_pct']:.0f}%")
            c4.metric("Verbosity", analysis["verbosity"].title())

            with st.expander("Detailed Analysis"):
                ac1, ac2 = st.columns(2)
                with ac1:
                    st.markdown("**Input Analysis**")
                    st.json({
                        "paragraphs": analysis["paragraph_count"],
                        "avg_sentence_length": analysis["avg_sentence_length"],
                        "repetition_ratio": analysis["repetition_ratio"],
                        "ai_patterns_found": analysis["ai_pattern_count"],
                        "is_long_text": analysis["text_is_long"],
                    })
                with ac2:
                    st.markdown("**Readability**")
                    st.json({
                        "flesch_kincaid_grade": readability.get("flesch_kincaid_grade", "-"),
                        "flesch_reading_ease": readability.get("flesch_reading_ease", "-"),
                        "gunning_fog_index": readability.get("gunning_fog_index", "-"),
                        "complexity": readability.get("complexity_rating", "-"),
                    })

        st.markdown("")
        btn_disabled = not (input_text and input_text.strip()) or not llm_ready
        btn_label = "Transform Text" if llm_ready else "LLM Offline - Configure GROQ_API_KEY in .env"

        if st.button(btn_label, type="primary", use_container_width=True, disabled=btn_disabled):
            with st.spinner("Running NLP pipeline..."):
                start = time.time()
                try:
                    result = engine.transform(
                        text=input_text,
                        mode=mode,
                        style=style,
                        summary_length=summary_length,
                        tone_slider=tone_slider,
                        auto_clipboard=auto_clipboard,
                        clipboard_watcher=clipboard_watcher,
                        realtime_mode=realtime_mode,
                    )
                    elapsed = time.time() - start

                    if "error" in result and not result.get("output"):
                        st.error(result["error"])
                    else:
                        st.session_state.output_text = result["output"]
                        st.session_state.last_result = result
                        st.session_state.last_elapsed = elapsed

                        st.session_state.history.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "mode": mode,
                            "style": style,
                            "input_words": result.get("analysis", {}).get("word_count", 0),
                            "output_length": result.get("output_length", 0),
                            "compression": result.get("compression_ratio", 0),
                        })

                        st.rerun()

                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                    logger.exception("Transform failed")

    with col_output:
        st.markdown("### Output")

        result = st.session_state.last_result

        if result and result.get("output"):
            output = result["output"]

            stages = result.get("stages_applied", [])
            if stages:
                stage_html = ""
                for i, s in enumerate(stages):
                    cls = "stage-humanize" if s == "humanize" else "stage-summarize"
                    stage_html += f'<span class="stage-badge {cls}">{s.title()}</span>'
                    if i < len(stages) - 1:
                        stage_html += ' <span class="stage-arrow"> > </span> '
                st.markdown(stage_html, unsafe_allow_html=True)
                st.markdown("")

            st.markdown(f'<div class="output-box">{output}</div>', unsafe_allow_html=True)

            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.download_button(
                    "Download .txt",
                    data=output,
                    file_name="transformed_text.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with bc2:
                if st.button("Copy to Clipboard", use_container_width=True):
                    try:
                        import pyperclip
                        clean_output = output.replace("[CLIPBOARD_COPY_READY]", "").strip()
                        pyperclip.copy(clean_output)
                        st.success("Copied!")
                    except ImportError:
                        st.code(output, language=None)
                        st.caption("Select and copy manually (install pyperclip for auto-copy)")
            with bc3:
                if st.button("Clear Output", use_container_width=True):
                    st.session_state.output_text = ""
                    st.session_state.last_result = None
                    st.rerun()

            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Transformation Metrics")

            elapsed = getattr(st.session_state, "last_elapsed", 0)

            m1, m2, m3, m4, m5, m6 = st.columns(6)

            compression = result.get("compression_ratio", 0)
            comp_pct = f"{compression * 100:.0f}%"

            m1.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{comp_pct}</div>
                    <div class="metric-label">Compression</div>
                </div>
            """, unsafe_allow_html=True)

            in_words = result.get("analysis", {}).get("word_count", 0)
            m2.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{in_words:,}</div>
                    <div class="metric-label">Input Words</div>
                </div>
            """, unsafe_allow_html=True)

            out_words = result.get("readability", {}).get("word_count", 0)
            m3.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{out_words:,}</div>
                    <div class="metric-label">Output Words</div>
                </div>
            """, unsafe_allow_html=True)

            ai_pct = result.get("analysis", {}).get("ai_likelihood_pct", 0)
            m4.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{ai_pct:.0f}%</div>
                    <div class="metric-label">AI Likelihood</div>
                </div>
            """, unsafe_allow_html=True)

            fre = result.get("readability", {}).get("flesch_reading_ease", 0)
            m5.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{fre:.0f}</div>
                    <div class="metric-label">Reading Ease</div>
                </div>
            """, unsafe_allow_html=True)

            fkg = result.get("readability", {}).get("flesch_kincaid_grade", 0)
            m6.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{fkg:.1f}</div>
                    <div class="metric-label">FK Grade</div>
                </div>
            """, unsafe_allow_html=True)

            if elapsed:
                st.caption(f"Processed in {elapsed:.1f}s / "
                          f"Complexity: {result.get('readability', {}).get('complexity_rating', '-')} / "
                          f"Level: {result.get('humanization_level', 0):.1f}")

        else:
            st.markdown("""
                <div style="text-align: center; padding: 4rem 2rem; opacity: 0.4;">
                    <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">Ready to Transform</div>
                    <div style="font-size: 0.9rem;">Enter text and configure the pipeline to get started.</div>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
