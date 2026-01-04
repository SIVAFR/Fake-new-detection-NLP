import streamlit as st
import torch



# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
css = """
<style>
/* Full background */
[data-testid="stAppViewContainer"] {
    background: #050608;
    color: #f9fafb;
}

/* Reduce top padding */
.block-container {
    padding-top: 2.5rem;
}

/* Title + subtitle */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #9ca3af;
    margin-bottom: 24px;
}

/* Input label */
.input-label {
    font-size: 15px;
    color: #e5e7eb;
    margin-bottom: 6px;
}

/* Textarea styling */
textarea {
    background: #020617 !important;
    color: #f9fafb !important;
    border-radius: 10px !important;
    border: 1px solid #374151 !important;
}

/* Result boxes */
.result-box {
    margin-top: 20px;
    padding: 16px;
    border-radius: 14px;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
}
.fake-box {
    background: rgba(248, 113, 113, 0.10);
    border: 1px solid #f87171;
    color: #fecaca;
}
.real-box {
    background: rgba(34, 197, 94, 0.10);
    border: 1px solid #22c55e;
    color: #bbf7d0;
}
.confidence {
    font-size: 14px;
    color: #e5e7eb;
    margin-top: 4px;
    text-align: center;
}

/* Footer */
.footer {
    position: fixed;
    bottom: 10px;
    width: 100%;
    text-align: center;
    font-size: 13px;
    color: #6b7280;
    opacity: 0.9;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<div class='title'>📰 Fake News Detection System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Quickly check if a headline or post looks REAL or FAKE.</div>",
    unsafe_allow_html=True,
)

# -----------------------------
# INPUT AREA
# -----------------------------
st.markdown("<div class='input-label'>Enter any news headline, social media post, or claim:</div>",
            unsafe_allow_html=True)

text = st.text_area(
    label="",
    height=180,
    placeholder="Example: Government announces new policy on renewable energy..."
)

analyze_clicked = st.button("Analyze", use_container_width=True)

# -----------------------------
# PREDICTION
# -----------------------------
# Make model LESS aggressive about calling FAKE
THRESHOLD = 0.85   # 0.85 = only very high fake-prob gets called fake

if analyze_clicked:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        fake_prob = probs[0].item()
        real_prob = probs[1].item()

        # Only say FAKE if fake_prob is very high AND clearly bigger than real_prob
        if fake_prob > THRESHOLD and fake_prob > real_prob:
            result_html = "<div class='result-box fake-box'>🚫 FAKE NEWS</div>"
        else:
            result_html = "<div class='result-box real-box'>✅ REAL NEWS</div>"

        st.markdown(result_html, unsafe_allow_html=True)
        st.markdown(
            f"<div class='confidence'>Fake probability: {fake_prob:.2f} • Real probability: {real_prob:.2f}</div>",
            unsafe_allow_html=True,
        )

# -----------------------------
# FOOTER WITH YOUR NAME
# -----------------------------
footer_html = """
<div class="footer">
    Developed by <strong>Sivanandan KP</strong>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
