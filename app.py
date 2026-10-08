import os
import time
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from engine import Store, SERVICES, DEMO_PRICES, ICONS, quote

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

@st.cache_data
def load_df():
    return pd.read_csv(CSV)

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
    MODELS = ["gemini-3.5-flash-lite", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]
    model = st.selectbox("Model", MODELS)
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
        df = load_df()
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
            for r, t in st.session_state.msgs[-9:-1] if not t.startswith(("⚠️", "🚫"))]
    while hist and hist[0].role == "model":  # Gemini requires the conversation to start with a user turn
        hist.pop(0)
    hist.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    base = dict(system_instruction=system_prompt(), tools=TOOLS, temperature=0.4, max_output_tokens=600)
    cfgs = [types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low"), **base),
            types.GenerateContentConfig(**base)]  # 2nd = plain config if a model rejects the thinking setting
    last = None
    for m in [model] + [x for x in MODELS if x != model]:  # fall back to other models if one is unavailable
        for cfg in cfgs:
            try:
                return client.models.generate_content(model=m, contents=hist, config=cfg).text
            except Exception as e:
                last = e
                msg = str(e)
                if any(k in msg for k in ("404", "NOT_FOUND", "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")):
                    time.sleep(0.5)
                    break  # model missing / overloaded / rate-limited -> try the next model
                if "thinking" in msg.lower() or "INVALID_ARGUMENT" in msg:
                    continue  # retry same model without the thinking setting
                raise
    raise last

def chat_ui():
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
                    reply = ("⚠️ Google's AI servers are busy right now — please press send again in a few seconds."
                             if any(k in str(e) for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
                             else f"⚠️ AI error — {type(e).__name__}: {str(e)[:350]}")
            st.markdown(reply)
        st.session_state.msgs.append(("assistant", reply))
        if S.events and S.events[-1] == "booked":
            st.balloons()
        S.events.clear()
        st.rerun()

# ---------- tap-to-book UI
def slot_grid(slots, prefix, selected=None, cols=6):
    clicked = None
    if not slots:
        st.warning("No free slots that day for this service - try another date.")
        return None
    cs = st.columns(cols)
    for i, t in enumerate(slots):
        if cs[i % cols].button(t, key=f"{prefix}{t}", type="primary" if t == selected else "secondary", use_container_width=True):
            clicked = t
    return clicked

def price_txt(p):
    return "Free" if p == 0 else f"₹{p:,}"

def quick_book():
    ss = st.session_state
    ss.setdefault("qb_svc", None); ss.setdefault("qb_slot", None)
    st.markdown("#### 1 · Tap a service")
    cs = st.columns(3)
    for i, (name, dur) in enumerate(SERVICES.items()):
        label = f"{ICONS.get(name,'💇')} {name}  \n{dur} min · {price_txt(DEMO_PRICES.get(name,0))}"
        if cs[i % 3].button(label, key=f"svc{i}", type="primary" if ss.qb_svc == name else "secondary", use_container_width=True):
            ss.qb_svc, ss.qb_slot = name, None
    if not ss.qb_svc:
        st.info("Select a service to see its duration and quote.")
        return
    d = st.date_input("2 · Pick a date", S.now.date() + timedelta(days=1), min_value=S.now.date(), key="qb_date")
    slots = S.free_slots(d.isoformat(), SERVICES[ss.qb_svc])["free_start_times"]
    st.markdown(f"#### 2 · Tap a time  \n<span style='opacity:.7'>{len(slots)} free slots on {d:%A %d %b}</span>", unsafe_allow_html=True)
    sel = slot_grid(slots, "qs", ss.qb_slot)
    if sel:
        ss.qb_slot = sel; st.rerun()
    if ss.qb_slot and ss.qb_slot not in slots:
        ss.qb_slot = None
    start = datetime.strptime(f"{d} {ss.qb_slot}", "%Y-%m-%d %H:%M") if ss.qb_slot else None
    q = quote(ss.qb_svc, start)
    st.markdown(f"""<div class='out' style='font-size:15px'><b>{ICONS.get(ss.qb_svc,'💇')} {q['service']}</b><br>
⏱ Duration: <b>{q['minutes']} min</b> · 💰 Quote: <b>{price_txt(q['price'])}</b> <i>(demo price)</i>
{f"<br>🕒 {ss.qb_slot} → {q['ends']} on {d:%a %d %b}" if start else "<br>Pick a time to see your end time."}</div>""", unsafe_allow_html=True)
    if ss.qb_slot:
        clients = sorted({a["client"] for a in S.appts})
        c = st.selectbox("3 · Who is booking?", clients, index=clients.index("Client_064"))
        if S.no_shows(c) >= 2:
            st.warning("This client has 2+ no-shows: a deposit link will be sent with the confirmation.")
        if st.button("✅ Confirm booking", type="primary"):
            r = S.book(c, ss.qb_svc, d.isoformat(), ss.qb_slot)
            if r["ok"]:
                st.success(f"Booked {r['service']} · {r['when']} (ref {r['ref']}). Confirmation + 24h reminder are in the Outbox.")
                st.balloons(); ss.qb_slot = None
            else:
                st.error(r["error"])

def my_appts():
    ss = st.session_state
    clients = sorted({a["client"] for a in S.appts})
    c = st.selectbox("Client", clients, index=clients.index("Client_064"), key="ma_client")
    ups = S.upcoming(c)
    if not ups:
        st.info("No upcoming appointments for this client. Use Quick book to create one.")
    for a in ups:
        c1, c2, c3 = st.columns([4, 1.3, 1.5])
        c1.markdown(f"**{ICONS.get(a['service'],'💇')} {a['service']}** · {a['start']:%a %d %b %H:%M} ({a['dur']} min)  \n<span style='opacity:.6'>ref {a['ref']}</span>", unsafe_allow_html=True)
        if c2.button("Cancel", key="c" + a["ref"]):
            r = S.cancel(a["ref"]); pass
            ss.ma_msg = r["policy"]; st.rerun()
        if c3.button("Reschedule", key="r" + a["ref"]):
            ss.rs_ref = a["ref"]
    if ss.get("ma_msg"):
        st.warning("Cancelled. " + ss.pop("ma_msg"))
    ref = ss.get("rs_ref"); a = S._get(ref) if ref else None
    if a and a["status"] not in ("cancelled", "rescheduled", "no_show"):
        st.markdown(f"#### Move **{a['service']}** to…")
        d = st.date_input("New date", a["start"].date(), min_value=S.now.date(), key="rs_date")
        sel = slot_grid(S.free_slots(d.isoformat(), a["dur"])["free_start_times"], "rs")
        if sel:
            r = S.reschedule(ref, d.isoformat(), sel)
            if r["ok"]:
                ss.rs_ref = None; st.success(f"Moved to {r['new_time']}. Notice sent (see Outbox)."); st.balloons()
            else:
                st.error(r["error"])

mode = st.radio("Mode", ["📅 Quick book (tap)", "🗓️ My appointments", "💬 Chat with AI"], horizontal=True, label_visibility="collapsed")
if mode.startswith("📅"):
    quick_book()
elif mode.startswith("🗓️"):
    my_appts()
else:
    chat_ui()
