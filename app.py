import os
from datetime import timedelta
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from engine import Store, SERVICES

st.set_page_config(page_title="GlowBot – Salon Appointment Assistant", page_icon="💇", layout="wide")
st.markdown("""<style>
@keyframes grad{0%{background-position:0 50%}50%{background-position:100% 50%}100%{background-position:0 50%}}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@keyframes pop{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.hero{background:linear-gradient(120deg,#a8325e,#e0719b,#7a4fd1,#a8325e);background-size:300% 300%;animation:grad 12s ease infinite;
 color:#fff;padding:18px 22px;border-radius:16px;display:flex;gap:14px;align-items:center;margin-bottom:10px}
.hero .av{font-size:38px;animation:bob 2.4s ease-in-out infinite}.hero h1{margin:0;font-size:26px;color:#fff}.hero p{margin:0;opacity:.9}
[data-testid="stChatMessage"]{animation:pop .3s ease-out}
.out{border-left:3px solid #a8325e;padding:6px 10px;margin-bottom:6px;border-radius:6px;background:rgba(168,50,94,.08);font-size:13px}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
</style>
<div class="hero"><div class="av">💇</div><div><h1>GlowBot</h1><p>AI appointment assistant · book · reschedule · cancel · reminders · no-show handling</p></div></div>""",
            unsafe_allow_html=True)

CSV = os.path.join(os.path.dirname(__file__), "appointments_anonymized.csv")
if "store" not in st.session_state:
    st.session_state.store = Store(CSV)
    st.session_state.msgs = [("assistant", "Hi! I'm GlowBot ✨ I can check availability, book, reschedule or cancel "
                              "appointments, send reminders, handle no-shows — and answer other questions too. "
                              "Try: *“Book a balayage on Friday at 2pm for Client_064”*.")]
S = st.session_state.store

# ---------- sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = None
    key = key or os.getenv("GEMINI_API_KEY") or st.text_input("Gemini API key", type="password", help="Free key: aistudio.google.com/apikey")
    model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"])
    d = st.date_input("Demo date (data covers 2025)", S.now.date())
    S.now = S.now.replace(year=d.year, month=d.month, day=d.day)
    st.header("🧑‍💼 Staff panel")
    nxt = sorted([a for a in S.appts if a["start"] > S.now and a["status"] == "booked"], key=lambda a: a["start"])[:4]
    for a in nxt:
        c1, c2 = st.columns([3, 2])
        c1.caption(f"{a['start']:%d %b %H:%M} · {a['client']}\n\n{a['service']}")
        if c2.button("No-show", key="ns" + a["ref"]):
            r = S.mark_no_show(a["ref"])
            st.session_state.msgs.append(("assistant", f"🚫 {r['client']} marked no-show. {r['policy_applied']}."))
            st.rerun()
    st.header("📨 Outbox (simulated SMS)")
    for o in S.outbox[:8]:
        st.markdown(f"<div class='out'><b>{o['kind']}</b> → {o['to']} · <i>{o['when']}</i><br>{o['text']}</div>", unsafe_allow_html=True)
    with st.expander("📊 Dataset insights"):
        df = pd.read_csv(CSV)
        st.write(f"{len(df)} appointments · {df.client_pid.nunique()} clients")
        st.bar_chart(df.status.value_counts())
        st.bar_chart(pd.to_datetime(df.appt_datetime_utc).dt.hour.value_counts().sort_index())

# ---------- Gemini agent
def system_prompt():
    return f"""You are GlowBot, the friendly AI front-desk assistant of a hair salon with one stylist chair.
Current date/time (demo clock): {S.now:%A %Y-%m-%d %H:%M}. Salon is open daily 09:00-18:00, slots every 30 minutes.
Services (minutes): {', '.join(f'{k} {v}' for k, v in SERVICES.items())}.
Policies: free cancel/reschedule with 24h+ notice; under 24h = late cancellation (fee may apply); a reminder is sent 24h before;
no-shows: 1st warning, 2nd deposit required, 3rd manual approval.
Rules: ALWAYS use the tools for availability, booking, cancelling, rescheduling and listing appointments - never guess.
Convert relative dates ('Friday', 'tomorrow') to YYYY-MM-DD using the current date. Before booking, cancelling or rescheduling,
make sure you have the client ID (like Client_064), service and date/time, and confirm the details with the customer.
After a late cancellation, mention the policy. Use mark_no_show only if the user says they are staff.
For other questions (policies, hair-care, general knowledge, small talk) answer helpfully in under 100 words, then offer to help book.
Never invent prices, addresses or phone numbers - say you don't have them. Reply in the customer's language."""

def check_availability(date: str, service: str = "Women's Haircut") -> dict:
    """List free start times on a date (YYYY-MM-DD) for a service name."""
    from engine import match_service
    return S.free_slots(date, SERVICES.get(match_service(service) or "", 60))

def book_appointment(client_id: str, service: str, date: str, time: str) -> dict:
    """Book an appointment. client_id like Client_064, date YYYY-MM-DD, time HH:MM (24h)."""
    return S.book(client_id, service, date, time)

def list_my_appointments(client_id: str) -> list:
    """List a client's upcoming appointments with their reference codes."""
    return S.my_appointments(client_id)

def cancel_appointment(ref: str) -> dict:
    """Cancel an appointment by its reference code."""
    return S.cancel(ref)

def reschedule_appointment(ref: str, new_date: str, new_time: str) -> dict:
    """Move an appointment to new_date (YYYY-MM-DD) and new_time (HH:MM)."""
    return S.reschedule(ref, new_date, new_time)

def mark_no_show(ref: str) -> dict:
    """STAFF ONLY: mark an appointment as a no-show and apply the no-show policy."""
    return S.mark_no_show(ref)

TOOLS = [check_availability, book_appointment, list_my_appointments, cancel_appointment, reschedule_appointment, mark_no_show]

def ask(prompt):
    client = genai.Client(api_key=key)
    hist = [types.Content(role="user" if r == "user" else "model", parts=[types.Part(text=t)])
            for r, t in st.session_state.msgs[-13:-1] if not t.startswith(("⚠️", "🚫"))]
    while hist and hist[0].role == "model":  # Gemini requires the conversation to start with a user turn
        hist.pop(0)
    hist.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    cfg = types.GenerateContentConfig(system_instruction=system_prompt(), tools=TOOLS, temperature=0.4)
    return client.models.generate_content(model=model, contents=hist, config=cfg).text

# ---------- chat UI
for role, text in st.session_state.msgs:
    with st.chat_message(role, avatar="💇" if role == "assistant" else None):
        st.markdown(text)

chips = ["Check availability tomorrow", "What services do you offer?", "How does your no-show policy work?", "Tell me a hair-care tip"]
st.session_state.setdefault("pk", 0)
picked = st.pills("Quick questions", chips, key=f"pill{st.session_state.pk}", label_visibility="collapsed") if hasattr(st, "pills") else None
prompt = st.chat_input("Ask anything or book an appointment…") or picked
if prompt:
    if not key:
        st.warning("Add your free Gemini API key in the sidebar (or in Streamlit secrets) to chat.")
        st.stop()
    st.session_state.pk += 1  # resets the pills so they do not re-send
    st.session_state.msgs.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="💇"):
        with st.spinner("GlowBot is typing…"):
            try:
                reply = ask(prompt)
            except Exception as e:
                reply = f"⚠️ AI error — {type(e).__name__}: {str(e)[:350]}"
        st.markdown(reply)
    st.session_state.msgs.append(("assistant", reply))
    if S.events and S.events[-1] == "booked":
        st.balloons()
    S.events.clear()
    st.rerun()
