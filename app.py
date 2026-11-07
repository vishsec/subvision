import streamlit as st

import serve

st.set_page_config(page_title="subvision - knee OA pre-read", layout="centered")

st.title("Knee OA pre-read (KL >= 2)")
st.caption("Pre-read AID only. Not a diagnosis. Requires radiologist sign-off.")

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
    if result["label"].startswith("KL>="):
        st.success(f"**{result['label']}**")
    else:
        st.info(f"**{result['label']}**")

    if result.get("uncertainty") == "borderline":
        st.warning(
            f"**Borderline** (p in {result['band']}) — queue for radiologist sign-off."
        )

    st.metric("P(KL >= 2)", f"{prob:.4f}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Input", result["input_format"])
    c2.metric("Threshold", result["threshold"])
    c3.metric("Latency", f"{result['latency_ms']['total']:.0f} ms")

    c4, c5 = st.columns(2)
    c4.caption(f"model `{result.get('model_version', '?')}`")
    c5.caption(f"request `{result.get('request_id', '?')}`")

    with st.expander("Latency breakdown"):
        st.write(result["latency_ms"])

    st.warning(result["disclaimer"])

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
