import streamlit as st
import requests
import json
import uuid

# ─── Config ──────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="QanoonAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f1117; }
[data-testid="stSidebar"] {
    background-color: #1a1d26;
    border-right: 1px solid #2e3140;
}
.user-msg {
    background: #1e3a5f;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px auto;
    color: #e8f4f8;
    max-width: 75%;
    text-align: right;
}
.ai-msg {
    background: #1a1d26;
    border: 1px solid #2e3140;
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px;
    margin: 8px auto 8px 0;
    color: #e8f4f8;
    max-width: 75%;
}
.logo-text { font-size: 24px; font-weight: 800; color: #4a9eff; }
.tagline   { font-size: 12px; color: #6b7280; margin-top: -6px; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────
def new_thread_id() -> str:
    return str(uuid.uuid4())


def fetch_threads() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/threads", timeout=5)
        return r.json() if r.status_code == 200 else []
    except:
        return []


def fetch_history(thread_id: str) -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/threads/{thread_id}/messages", timeout=5)
        if r.status_code == 200:
            return r.json().get("messages", [])
        return []
    except:
        return []


def get_preview(thread_id: str) -> str:
    """First user message as preview, fallback to short uuid."""
    msgs = st.session_state.chat_history.get(thread_id, [])
    for m in msgs:
        if m.get("role") == "user":
            txt = m["content"]
            return txt[:40] + "..." if len(txt) > 40 else txt
    return f"New Chat • {thread_id[:6]}"


def stream_response(query: str, thread_id: str) -> str:
    url = f"{API_URL}/chat/stream"
    full_text = ""
    placeholder = st.empty()

    try:
        with requests.post(url, json={"query": query, "thread_id": thread_id},
                           stream=True, timeout=180) as r:
            for line in r.iter_lines():
                if line:
                    raw = line.decode("utf-8")
                    if raw.startswith("data: "):
                        data = json.loads(raw[6:])
                        if "token" in data:
                            full_text += data["token"]
                            placeholder.markdown(
                                f'<div class="ai-msg">⚖️ {full_text}▌</div>',
                                unsafe_allow_html=True)
                        elif data.get("done"):
                            placeholder.markdown(
                                f'<div class="ai-msg">⚖️ {full_text}</div>',
                                unsafe_allow_html=True)
                        elif "error" in data:
                            placeholder.error(data["error"])
    except Exception as e:
        placeholder.error(f"Connection error: {e}")

    return full_text


# ─── Session State ────────────────────────────────────────────
# thread_id  : currently active thread
# thread_list: ordered list of thread ids (newest first)
# chat_history: {thread_id: [messages]}

if "thread_id" not in st.session_state:
    # create first thread on app load
    tid = new_thread_id()
    st.session_state.thread_id   = tid
    st.session_state.thread_list = [tid]
    st.session_state.chat_history = {tid: []}

# pull any threads saved in DB that aren't in local list yet
try:
    db_threads = fetch_threads()
    for t in db_threads:
        tid = t["thread_id"]
        if tid not in st.session_state.thread_list:
            st.session_state.thread_list.append(tid)
        if tid not in st.session_state.chat_history:
            st.session_state.chat_history[tid] = fetch_history(tid)
except:
    pass


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">⚖️ QanoonAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Pakistan Legal Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    # New Chat — generate fresh UUID, switch to it
    if st.button("➕  New Chat", use_container_width=True, type="primary"):
        tid = new_thread_id()
        st.session_state.thread_id = tid
        st.session_state.thread_list.insert(0, tid)   # newest first
        st.session_state.chat_history[tid] = []
        st.rerun()

    st.markdown("### 💬 Chats")

    for tid in st.session_state.thread_list:
        preview  = get_preview(tid)
        is_active = tid == st.session_state.thread_id
        label    = f"▶  {preview}" if is_active else f"💬  {preview}"

        if st.button(label, key=f"btn_{tid}", use_container_width=True):
            # switch to this thread and load its history
            st.session_state.thread_id = tid
            if not st.session_state.chat_history.get(tid):
                st.session_state.chat_history[tid] = fetch_history(tid)
            st.rerun()

    st.markdown("---")
    st.caption(f"ID: `{st.session_state.thread_id[:8]}...`")


# ─── Main Chat ────────────────────────────────────────────────
active_tid  = st.session_state.thread_id
active_msgs = st.session_state.chat_history.get(active_tid, [])

st.markdown("### ⚖️ QanoonAI")
st.caption("Ask me anything about Pakistani law — Urdu or English!")
st.markdown("---")

# render history
for msg in active_msgs:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-msg">⚖️ {msg["content"]}</div>',
                    unsafe_allow_html=True)

# input box
user_input = st.chat_input("Type your legal question here... (Urdu or English)")

if user_input:
    # show user bubble immediately
    st.markdown(f'<div class="user-msg">🧑 {user_input}</div>',
                unsafe_allow_html=True)

    # append to local history
    active_msgs.append({"role": "user", "content": user_input})
    st.session_state.chat_history[active_tid] = active_msgs

    # stream answer
    ai_response = stream_response(user_input, active_tid)

    if ai_response:
        active_msgs.append({"role": "assistant", "content": ai_response})
        st.session_state.chat_history[active_tid] = active_msgs

        # move this thread to top of sidebar list
        if active_tid in st.session_state.thread_list:
            st.session_state.thread_list.remove(active_tid)
        st.session_state.thread_list.insert(0, active_tid)

    st.rerun()