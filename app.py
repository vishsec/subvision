import streamlit as st

import serve

BG = "https://raw.githubusercontent.com/vishsec/subvision/main/gradcam/bg_knee.png"

st.set_page_config(page_title="subvision - knee OA pre-read", layout="centered")

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(rgba(255,255,255,0.94), rgba(255,255,255,0.97)),
            url("{BG}") center/cover no-repeat fixed;
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    .block-container {{ padding-top: 2rem; max-width: 760px; }}

    h1, p, label, .stMarkdown {{ color: #16181d; }}

    .hero {{
        font-size: 1.9rem; font-weight: 700; letter-spacing: 0.42em;
        text-align: center; color: #000; margin-bottom: 0.1em;
        text-indent: 0.42em;
    }}
    .hero span {{
        display: block; font-size: 0.68rem; font-weight: 400;
        letter-spacing: 0.58em; color: #707070; margin-top: 0.5em;
        text-indent: 0.58em;
    }}
    .tag {{
        text-align: center; color: #555; font-size: 0.7rem;
        letter-spacing: 0.18em; text-transform: uppercase;
        border-top: 1px solid #e0e0e0; border-bottom: 1px solid #e0e0e0;
        padding: 0.55em 0; margin: 0.8em 0 1.5em 0;
    }}

    div[data-testid="stFileUploader"] section {{
        border: 1px dashed #b0b0b0 !important;
        background: #fafafa; border-radius: 4px;
    }}
    div[data-testid="stFileUploader"] section:hover {{ border-color: #000; }}
    div[data-testid="stFileUploader"] label {{ color: #16181d !important; }}

    .verdict {{
        border: 2px solid #000; padding: 0.8em 1em; text-align: center;
        letter-spacing: 0.2em; font-weight: 700; font-size: 1.02rem;
        margin: 1em 0 0.4em 0; border-radius: 4px; background: #fff; color: #000;
    }}
    .verdict.pos {{ background: #000; color: #fff; }}
    .verdict small {{
        display: block; font-size: 0.6rem; font-weight: 600; letter-spacing: 0.3em;
        margin-top: 0.45em; opacity: 0.8;
    }}
    .flag {{
        background: #fff4d6; border: 1px solid #e0b040;
        color: #7a4d00; text-align: center; font-weight: 700; letter-spacing: 0.16em;
        padding: 0.55em; margin: 0.45em 0 0.15em 0; border-radius: 4px; font-size: 0.82rem;
    }}

    .prob {{
        text-align: center; font-size: 2.5rem; font-weight: 700;
        color: #000; letter-spacing: 0.04em; margin: 0.4em 0 0.05em 0;
    }}
    .prob-pct {{
        font-size: 1.15rem; font-weight: 600; color: #666;
        letter-spacing: 0.02em; margin-left: 0.35em;
    }}
    .prob-sub {{
        text-align: center; color: #666; font-size: 0.68rem;
        letter-spacing: 0.32em; text-transform: uppercase; margin-bottom: 1.1em;
    }}
    .grid {{ display: flex; gap: 8px; justify-content: center; margin-bottom: 1.1em; }}
    .cell {{
        flex: 1; border: 1px solid #e0e0e0; background: #fff;
        padding: 0.55em 0.35em; text-align: center; border-radius: 4px;
    }}
    .cell b {{ display: block; color: #000; font-size: 0.92rem; word-break: break-all; }}
    .cell span {{
        display: block; color: #888; font-size: 0.6rem;
        letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.25em;
    }}

    .disc {{
        border: 1px solid #e2e2e2; background: #f6f6f6;
        color: #444; font-size: 0.76rem; letter-spacing: 0.05em;
        padding: 0.7em 0.9em; margin-top: 1.3em; border-radius: 4px;
    }}

    [data-testid="stExpander"] {{
        border: 1px solid #e4e4e4; background: #fafafa; border-radius: 4px;
    }}
    [data-testid="stExpander"] summary {{ color: #16181d; letter-spacing: 0.1em; }}

    .stButton > button {{
        background: #000; color: #fff; border: 1px solid #000; border-radius: 4px;
        letter-spacing: 0.15em;
    }}
    .stButton > button:hover {{ background: #fff; color: #000; border: 1px solid #000; }}

    [data-testid="stSpinner"] > div {{ border-top-color: #000; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero">SUBVISION<span>KNEE OA PRE-READ</span></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="tag">Pre-read AID only &middot; not a diagnosis &middot; requires radiologist sign-off</p>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload knee X-ray",
    type=["png", "jpg", "jpeg", "dcm"],
    help="PNG/JPG or DICOM (.dcm)",
)

if uploaded is not None:
    raw = uploaded.getvalue()
    with st.spinner("Running model..."):
        result = serve.predict_raw(raw)

    prob = result["prob_kl2plus"]
    pos = result["label"].startswith("KL>=")
    border = result.get("uncertainty") == "borderline"
    label = result["label"].replace(">=", "\u2265").replace("<", "\u003c")
    conf = (prob if pos else 1 - prob) * 100  # confidence of the PREDICTED class

    st.markdown(
        f'<div class="verdict {"pos" if pos else "neg"}">{label}'
        f"<small>{conf:.1f}% CONFIDENCE</small></div>",
        unsafe_allow_html=True,
    )
    if border:
        st.markdown(
            '<div class="flag">BORDERLINE &mdash; QUEUE FOR RADIOLOGIST SIGN-OFF</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="prob">{prob:.4f}<span class="prob-pct">({prob * 100:.1f}%)</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="prob-sub">probability of osteoarthritis (KL &ge; 2) &middot; threshold {result["threshold"]}</div>',
        unsafe_allow_html=True,
    )

    lat = result["latency_ms"]
    fmt = "DICOM" if result["input_format"] == "DICOM" else "PNG / JPG"
    st.markdown(
        f"""
        <div class="grid">
          <div class="cell"><b>{fmt}</b><span>input</span></div>
          <div class="cell"><b>{lat["total"]:.0f} ms</b><span>latency</span></div>
          <div class="cell"><b>{result.get("tta", "-")}</b><span>tta</span></div>
          <div class="cell"><b>{result.get("model_version", "?")}</b><span>model</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Latency breakdown & request"):
        st.write(lat)
        st.caption(f"request `{result.get('request_id', '?')}`")

    st.markdown(
        '<div class="disc">Pre-read aid only. Not a diagnosis. Requires radiologist sign-off. '
        "Single-view, OAI-derived data &mdash; needs local validation before any real use.</div>",
        unsafe_allow_html=True,
    )

with st.expander("Model card"):
    st.markdown(
        """
- **Task**: knee osteoarthritis grading, KL 0-1 (negative) vs KL 2-4 (positive)
- **Architecture**: DenseNet121, exported to ONNX (28.2 MB), CPU inference, **hflip TTA** (avg of base + mirrored pass)
- **Test set**: n=1,656 — AUC 0.950, accuracy 86.8%, sensitivity 0.81, specificity 0.91 (with TTA; bootstrap CI excludes zero)
- **Input**: PNG/JPG or DICOM, resized to 380x380, ImageNet normalization
- **Known limits**: single-view X-rays; weakest on borderline grades (KL1/KL2, 87% of errors); needs local validation and radiologist sign-off before any clinical use
- **Repo**: this app is the same code path as the tested FastAPI service (pytest: health, PNG/DICOM parity, determinism)
"""
    )
