import streamlit as st

import serve

BG = "https://raw.githubusercontent.com/vishsec/subvision/main/gradcam/bg_knee.png"

st.set_page_config(page_title="subvision - knee OA pre-read", layout="centered")

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(rgba(0,0,0,0.87), rgba(0,0,0,0.93)),
            url("{BG}") center/cover no-repeat fixed;
        filter: grayscale(1);
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    .block-container {{ padding-top: 2rem; max-width: 760px; }}

    h1, p, label, .stMarkdown {{ color: #f2f2f2; }}

    .hero {{
        font-size: 1.9rem; font-weight: 700; letter-spacing: 0.42em;
        text-align: center; color: #fff; margin-bottom: 0.1em;
        text-indent: 0.42em;
    }}
    .hero span {{
        display: block; font-size: 0.68rem; font-weight: 400;
        letter-spacing: 0.58em; color: #9a9a9a; margin-top: 0.5em;
        text-indent: 0.58em;
    }}
    .tag {{
        text-align: center; color: #8f8f8f; font-size: 0.7rem;
        letter-spacing: 0.18em; text-transform: uppercase;
        border-top: 1px solid #2c2c2c; border-bottom: 1px solid #2c2c2c;
        padding: 0.55em 0; margin: 0.8em 0 1.5em 0;
    }}

    div[data-testid="stFileUploader"] section {{
        border: 1px dashed #5a5a5a !important;
        background: rgba(0,0,0,0.55); border-radius: 2px;
    }}
    div[data-testid="stFileUploader"] section:hover {{ border-color: #fff; }}
    div[data-testid="stFileUploader"] label {{ color: #fff !important; }}

    .verdict {{
        border: 1px solid #fff; padding: 0.8em 1em; text-align: center;
        letter-spacing: 0.2em; font-weight: 700; font-size: 1.02rem;
        margin: 1em 0 0.4em 0; border-radius: 2px;
    }}
    .verdict.pos {{ background: #fff; color: #000; }}
    .verdict.neg {{ background: rgba(0,0,0,0.72); color: #fff; }}
    .flag {{
        background: repeating-linear-gradient(45deg, #fff, #fff 10px, #cfcfcf 10px, #cfcfcf 20px);
        color: #000; text-align: center; font-weight: 700; letter-spacing: 0.16em;
        padding: 0.55em; margin: 0.45em 0 0.15em 0; border-radius: 2px; font-size: 0.82rem;
    }}

    .prob {{
        text-align: center; font-size: 2.5rem; font-weight: 700;
        color: #fff; letter-spacing: 0.04em; margin: 0.4em 0 0.05em 0;
    }}
    .prob-sub {{
        text-align: center; color: #8f8f8f; font-size: 0.68rem;
        letter-spacing: 0.32em; text-transform: uppercase; margin-bottom: 1.1em;
    }}
    .grid {{ display: flex; gap: 8px; justify-content: center; margin-bottom: 1.1em; }}
    .cell {{
        flex: 1; border: 1px solid #333; background: rgba(0,0,0,0.6);
        padding: 0.55em 0.35em; text-align: center; border-radius: 2px;
    }}
    .cell b {{ display: block; color: #fff; font-size: 0.92rem; word-break: break-all; }}
    .cell span {{
        display: block; color: #7d7d7d; font-size: 0.6rem;
        letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.25em;
    }}

    .disc {{
        border: 1px solid #3a3a3a; background: rgba(0,0,0,0.66);
        color: #bdbdbd; font-size: 0.76rem; letter-spacing: 0.05em;
        padding: 0.7em 0.9em; margin-top: 1.3em; border-radius: 2px;
    }}

    [data-testid="stExpander"] {{
        border: 1px solid #2e2e2e; background: rgba(0,0,0,0.55); border-radius: 2px;
    }}
    [data-testid="stExpander"] summary {{ color: #dcdcdc; letter-spacing: 0.1em; }}

    .stButton > button {{
        background: #000; color: #fff; border: 1px solid #fff; border-radius: 2px;
        letter-spacing: 0.15em;
    }}
    .stButton > button:hover {{ background: #fff; color: #000; }}

    [data-testid="stSpinner"] > div {{ border-top-color: #fff; }}
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

    st.markdown(
        f'<div class="verdict {"pos" if pos else "neg"}">{label}</div>',
        unsafe_allow_html=True,
    )
    if border:
        st.markdown(
            '<div class="flag">BORDERLINE &mdash; QUEUE FOR RADIOLOGIST SIGN-OFF</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="prob">{prob:.4f}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="prob-sub">P(KL &ge; 2) &middot; threshold {result["threshold"]}</div>',
        unsafe_allow_html=True,
    )

    lat = result["latency_ms"]
    st.markdown(
        f"""
        <div class="grid">
          <div class="cell"><b>{result["input_format"]}</b><span>input</span></div>
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
