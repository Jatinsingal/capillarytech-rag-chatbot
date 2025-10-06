import json
import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="CapillaryTech Docs Chatbot", page_icon="💬", layout="wide")

# Professional theme + layout CSS
st.markdown(
	"""
	<style>
		:root {
			--bg:#0b0f14; --panel:#0e141b; --panel2:#0f1822; --muted:#9fb3c8; --text:#e6edf3;
			--primary:#2f81f7; --primary-700:#1f6feb; --border:#1b2733; --glow:#2f81f740;
		}
		/* page */
		header, .block-container {padding-top: 0 !important;}
		section.main > div {padding-top:0}
		/* gradient header */
		.hero {
			background: linear-gradient(135deg, #0d1117 0%, #0b1b34 60%, #0a2447 100%);
			border-bottom:1px solid var(--border);
			position:sticky; top:0; z-index:5;
			padding: 22px 0 18px 0; margin-bottom: 10px;
			box-shadow: 0 8px 24px -12px #00000088;
		}
		.hero h1{ color:var(--text); font-weight:700; letter-spacing:.3px; margin:0 }
		.subtle{ color:var(--muted); font-size:14px }
		.container{ max-width: 1100px; margin:0 auto; padding:0 16px }

		/* chat area */
		.chat {max-width: 900px; margin: 0 auto 90px auto;}
		.msg { display:block; padding:13px 16px; margin:10px 0; line-height:1.45; }
		.user { background:var(--primary); color:#fff; border-radius:16px; border-top-right-radius:6px; }
		.assistant { background:var(--panel2); color:var(--text); border:1px solid var(--border); border-radius:16px; border-top-left-radius:6px; box-shadow:0 6px 18px -10px var(--glow) }
		.citations{ font-size:13px; color:var(--muted) }
		.divider{border-top:1px solid var(--border); margin:18px 0}

		/* suggestion chips */
		.chips { display:flex; gap:10px; flex-wrap:wrap; }
		.chip { background:transparent; color:var(--text); border:1px solid var(--border); padding:8px 12px; border-radius:10px; cursor:pointer }
		.chip:hover{ border-color:var(--primary); box-shadow:0 0 0 2px var(--glow) inset }

		/* sticky input */
		.footer { position:fixed; bottom:0; left:0; right:0; background:linear-gradient(180deg, transparent, #0b0f14 35%); padding:14px 0 18px;}
		.input-wrap {max-width: 900px; margin:0 auto; display:flex; gap:10px}
		input[type="text"], textarea{ border-radius:10px !important; border:1px solid var(--border) !important; background:#0c131b !important; color:var(--text) !important; }
		.send-btn button{ height:42px; border-radius:10px; background:var(--primary-700) }

		/* sidebar */
		[data-testid="stSidebar"] {border-right:1px solid var(--border)}
		[data-testid="stSidebar"] .stButton>button{width:100%; border-radius:10px}
	</style>
	""",
	unsafe_allow_html=True,
)

# Hero header
st.markdown("""
<div class='hero'>
  <div class='container'>
    <h1>CapillaryTech Docs Chatbot</h1>
    <div class='subtle'>Ask product and capability questions. Cited from public resources.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='chat container'><div class='chat-inner'>", unsafe_allow_html=True)

if "messages" not in st.session_state:
	st.session_state.messages = []

with st.sidebar:
	st.markdown("**Controls**")
	st.caption("API: " + API_URL)
	col1, col2 = st.columns(2)
	with col1:
		refresh = st.button("Clear Chat")
	with col2:
		ingest = st.button("Rebuild Index")
	if refresh:
		st.session_state.messages = []
		st.rerun()
	if ingest:
		with st.spinner("Scraping and indexing (first run may take time)..."):
			try:
				resp = requests.post(f"{API_URL}/ingest", timeout=600)
				st.success(resp.json().get("status", "done"))
			except Exception as e:
				st.error(f"Ingest failed: {e}")

# History
for m in st.session_state.messages:
	klass = "user" if m["role"] == "user" else "assistant"
	st.markdown(f"<div class='msg {klass}'>{m['content']}</div>", unsafe_allow_html=True)

# Suggestions row (static, non-clickable)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
examples = [
	"What does CapillaryTech do?",
	"List core products and solutions",
	"Tell me about loyalty features",
	"How to get started quickly?",
]
st.markdown(
	"<div class='chips'>" + "".join([f"<div class='chip' style='pointer-events:none;opacity:.85'>{e}</div>" for e in examples]) + "</div>",
	unsafe_allow_html=True,
)

prompt = st.chat_input("Ask anything about CapillaryTech docs...")
if prompt:
	st.session_state.messages.append({"role": "user", "content": prompt})
	st.markdown(f"<div class='msg user'>{prompt}</div>", unsafe_allow_html=True)
	with st.spinner("Answering..."):
		try:
			resp = requests.post(
				f"{API_URL}/chat",
				json={"query": prompt},
				timeout=45,
			)
			data = resp.json()
			answer = data.get("answer", "")
			cites = data.get("citations", [])
			st.markdown(f"<div class='msg assistant'>{answer}</div>", unsafe_allow_html=True)
			if cites:
				with st.expander("Citations"):
					for c in cites:
						st.write(f"- {c.get('meta','')} (score {round(c.get('score',0),3)})")
			st.session_state.messages.append({"role": "assistant", "content": answer})
		except Exception as e:
			st.error(f"Request failed: {e}")

st.markdown("</div></div>", unsafe_allow_html=True)

# Sticky footer spacing
st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)


