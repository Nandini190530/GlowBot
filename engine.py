"""Booking engine for GlowBot: availability, booking, cancel, reschedule, no-show, reminders.
All state lives in a Store; the Gemini tools in app.py are thin wrappers around these methods."""
import re
from datetime import datetime, timedelta
import pandas as pd

OPEN_MIN, CLOSE_MIN, STEP = 9 * 60, 18 * 60, 30
INACTIVE = {"cancelled", "rescheduled", "no_show"}
SERVICES = {  # name -> minutes (estimated from the dataset's service names)
    "Women's Haircut": 60, "Blowout/styling": 45, "All over color": 120,
    "Partial Custom Highlight": 150, "Full custom Highlight": 150,
    "Balayage/Foilayage": 180, "Signature Blonde Session": 180,
    "Gloss with haircut": 75, "Wax": 30, "Free Consultation": 30,
}
_KEYS = [("balayage", 180), ("blonde", 180), ("highlight", 150), ("all over", 120),
         ("correction", 120), ("gloss", 75), ("haircut", 60), ("wax", 30), ("blowout", 45)]


def _dur(s):
    s = s.lower()
    return next((v for k, v in _KEYS if k in s), 60)


def _svc(s):
    return re.sub(r"\s*\(.*?\)", "", s.split(",")[0]).replace("|", " ").strip()


def match_service(text):
    t = text.lower()
    for name in SERVICES:
        if name.lower() in t or t in name.lower():
            return name
    for k, name in [("hair", "Women's Haircut"), ("cut", "Women's Haircut"), ("balayage", "Balayage/Foilayage"),
                    ("foil", "Balayage/Foilayage"), ("blond", "Signature Blonde Session"),
                    ("highlight", "Partial Custom Highlight"), ("color", "All over color"),
                    ("colour", "All over color"), ("gloss", "Gloss with haircut"), ("blow", "Blowout/styling"),
                    ("wax", "Wax"), ("consult", "Free Consultation")]:
        if k in t:
            return name
    return None


class Store:
    def __init__(self, csv_path, now="2025-11-03 09:00"):
        df = pd.read_csv(csv_path)
        self.appts = []
        for i, r in enumerate(df.itertuples()):
            st = "booked" if r.status in ("completed", "checkout") else r.status
            self.appts.append(dict(ref=r.appointment_hash[:8], client=r.client_pid,
                                   start=datetime.strptime(r.appt_datetime_utc[:16], "%Y-%m-%dT%H:%M"),
                                   dur=_dur(r.services), status=st, service=_svc(r.services)))
        self.base = len(self.appts)
        self.now = datetime.strptime(now, "%Y-%m-%d %H:%M")
        self.outbox, self.events, self.n = [], [], 0

    # ---- helpers
    def _msg(self, kind, client, text, when="sent now"):
        self.outbox.insert(0, dict(kind=kind, to=client, text=text, when=when))

    def _get(self, ref):
        return next((a for a in self.appts if a["ref"].lower() == ref.strip().lower()), None)

    def _parse(self, date, time):
        try:
            return datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        except ValueError:
            return None

    def _fmt(self, d):
        return d.strftime("%a %d %b %Y, %H:%M")

    def no_shows(self, client):
        return sum(a["client"] == client and a["status"] == "no_show" for a in self.appts)

    def upcoming(self, client):
        return sorted([a for a in self.appts if a["client"] == client and a["start"] > self.now
                       and a["status"] not in INACTIVE], key=lambda a: a["start"])

    def _is_free(self, start, dur, ignore=None):
        end = start + timedelta(minutes=dur)
        if start <= self.now or start.hour * 60 + start.minute < OPEN_MIN or \
                start.hour * 60 + start.minute + dur > CLOSE_MIN or start.minute % STEP:
            return False
        return not any(a is not ignore and a["status"] not in INACTIVE and a["start"] < end
                       and a["start"] + timedelta(minutes=a["dur"]) > start for a in self.appts)

    # ---- actions (all return plain dicts for the LLM)
    def free_slots(self, date, dur=60):
        d = datetime.strptime(date, "%Y-%m-%d")
        out = [(d + timedelta(minutes=m)).strftime("%H:%M") for m in range(OPEN_MIN, CLOSE_MIN, STEP)
               if self._is_free(d + timedelta(minutes=m), dur)]
        return dict(date=date, duration_min=dur, free_start_times=out)

    def book(self, client, service, date, time):
        svc = match_service(service)
        if not svc:
            return dict(ok=False, error="Unknown service", services=list(SERVICES))
        s = self._parse(date, time)
        if not s:
            return dict(ok=False, error="Bad date/time format; use YYYY-MM-DD and HH:MM")
        if not self._is_free(s, SERVICES[svc]):
            return dict(ok=False, error="Slot not available", alternatives=self.free_slots(date, SERVICES[svc]))
        self.n += 1
        a = dict(ref=f"N{self.n:03d}", client=client, start=s, dur=SERVICES[svc], status="booked", service=svc)
        self.appts.append(a)
        dep = self.no_shows(client) >= 2
        self._msg("Confirmation", client, f"Hi! Your {svc} is booked for {self._fmt(s)}. Free changes up to 24h before."
                  + (" A deposit link is required to secure this slot." if dep else ""))
        rem = s - timedelta(hours=24)
        self._msg("Reminder (24h)", client, f"Reminder: {svc} tomorrow at {s:%H:%M}. Reply C confirm / R reschedule / X cancel.",
                  f"scheduled {self._fmt(rem)}" if rem > self.now else "sent now (<24h away)")
        self.events.append("booked")
        return dict(ok=True, ref=a["ref"], service=svc, when=self._fmt(s), deposit_required=dep,
                    reminder_queued=True)

    def my_appointments(self, client):
        return [dict(ref=a["ref"], service=a["service"], when=self._fmt(a["start"])) for a in self.upcoming(client)]

    def cancel(self, ref):
        a = self._get(ref)
        if not a or a["status"] in INACTIVE:
            return dict(ok=False, error="Appointment not found or already inactive")
        late = a["start"] - self.now < timedelta(hours=24)
        a["status"] = "cancelled"
        self._msg("Cancellation", a["client"], f"Your {a['service']} on {self._fmt(a['start'])} is cancelled.")
        self.events.append("cancelled")
        return dict(ok=True, late_cancellation=late, policy="Late cancellation (<24h): fee may apply" if late
                    else "Free cancellation (24h+ notice)", slot_released=True)

    def reschedule(self, ref, new_date, new_time):
        a = self._get(ref)
        if not a or a["status"] in INACTIVE:
            return dict(ok=False, error="Appointment not found or already inactive")
        s = self._parse(new_date, new_time)
        if not s or not self._is_free(s, a["dur"], ignore=a):
            return dict(ok=False, error="New slot unavailable", alternatives=self.free_slots(new_date, a["dur"]) if s else None)
        late = a["start"] - self.now < timedelta(hours=24)
        old = a["start"]; a["start"] = s
        self._msg("Reschedule", a["client"], f"Your {a['service']} moved from {self._fmt(old)} to {self._fmt(s)}.")
        self.events.append("rescheduled")
        return dict(ok=True, new_time=self._fmt(s), late_change=late)

    def mark_no_show(self, ref):
        a = self._get(ref)
        if not a:
            return dict(ok=False, error="Appointment not found")
        a["status"] = "no_show"
        n = self.no_shows(a["client"])
        pol = {1: "1st no-show: friendly warning + rebooking link",
               2: "2nd no-show: deposit required for future bookings"}.get(n, "3+ no-shows: manual approval for future bookings")
        self._msg("No-show follow-up", a["client"], {1: "We missed you today! Reply B to rebook. Repeated no-shows may need a deposit.",
                  2: "We missed you again. A deposit is now required for future bookings.",
                  }.get(n, "Please call the salon to arrange your next booking."))
        self.events.append("no_show")
        return dict(ok=True, client=a["client"], no_show_count=n, policy_applied=pol)


# DEMO price list (the dataset has no prices) - replace with the salon's real rates.
DEMO_PRICES = {"Women's Haircut": 800, "Blowout/styling": 600, "All over color": 2500,
               "Partial Custom Highlight": 3500, "Full custom Highlight": 4500,
               "Balayage/Foilayage": 6000, "Signature Blonde Session": 7000,
               "Gloss with haircut": 1800, "Wax": 500, "Free Consultation": 0}
ICONS = {"Women's Haircut": "✂️", "Blowout/styling": "💨", "All over color": "🎨", "Partial Custom Highlight": "✨",
         "Full custom Highlight": "🌟", "Balayage/Foilayage": "🌅", "Signature Blonde Session": "👱‍♀️",
         "Gloss with haircut": "💎", "Wax": "🪶", "Free Consultation": "💬"}


def quote(service, start=None):
    d = SERVICES[service]
    q = dict(service=service, minutes=d, price=DEMO_PRICES.get(service, 0))
    if start:
        q["ends"] = (start + timedelta(minutes=d)).strftime("%H:%M")
    return q
