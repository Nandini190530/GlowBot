# GlowBot – Salon Appointment Booking Assistant (Streamlit + Gemini)

Chatbot for an appointment-booking use case: slot availability, confirmation + 24h reminder,
reschedule / cancel flow, and no-show handling. Powered by Google Gemini function calling over a
booking engine built on `appointments_anonymized.csv` (186 appointments, 105 clients, 1 stylist).

## Run locally
```
pip install -r requirements.txt
export GEMINI_API_KEY=your_key      # Windows PowerShell: $env:GEMINI_API_KEY="your_key"
streamlit run app.py
```
Free key: https://aistudio.google.com/apikey

## Deploy free on Streamlit Community Cloud (shareable link)
1. Create a GitHub repo and upload ALL files in this folder (app.py, engine.py, requirements.txt, the CSV).
2. Go to https://share.streamlit.io -> New app -> pick the repo, main file `app.py`.
3. Advanced settings -> Secrets, paste:  `GEMINI_API_KEY = "your_key"`
4. Deploy. Share the URL.  (Never commit the key to GitHub.)

## Files
- `app.py` – UI (animated header, chat, staff panel, simulated SMS outbox) + Gemini agent with 6 tools
- `engine.py` – availability, booking, cancel, reschedule, reminders, no-show policy
- `appointments_anonymized.csv` – dataset

## Policies implemented
Free cancel/reschedule with 24h+ notice; <24h = late cancellation; reminder 24h before;
no-show: 1st warning -> 2nd deposit required -> 3rd manual approval.
