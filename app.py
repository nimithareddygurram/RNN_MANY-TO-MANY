import streamlit as st
from src.predict import tag_sentence

st.set_page_config(page_title="TagFlow AI", page_icon="🏷️", layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 10% 10%,rgba(14,165,233,.14),transparent 30%),linear-gradient(135deg,#07111f,#111827,#090d18)}
.hero{text-align:center;font-size:52px;font-weight:900;background:linear-gradient(90deg,#22d3ee,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.card{padding:25px;border-radius:20px;border:1px solid rgba(148,163,184,.2);background:rgba(15,23,42,.75)}
.stButton>button{width:100%;height:48px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🏷️ TagFlow AI")
    page = st.radio("Navigation", ["🏠 Home","🏷️ POS Tagger","🧠 How It Works","🏗️ Architecture","📚 Project Details"])
    st.info("Many-to-Many\n\nMany input words → Many POS tags\n\nRecurrent layer: SimpleRNN only")

st.markdown('<div class="hero">TagFlow AI</div>', unsafe_allow_html=True)
st.caption("Many-to-Many Part-of-Speech Tagging using Simple RNN")

if page == "🏠 Home":
    st.markdown('<div class="card"><h2>Sequence Labeling</h2><p>Enter a sentence. The Simple RNN processes every word and returns a POS tag for each position.</p></div>', unsafe_allow_html=True)
    st.code("Word₁ Word₂ Word₃ ... → SimpleRNN(return_sequences=True) → Tag₁ Tag₂ Tag₃ ...")
elif page == "🏷️ POS Tagger":
    sentence = st.text_area("Enter a sentence", "the dog runs quickly")
    if st.button("🚀 Tag Sentence"):
        if sentence.strip():
            try:
                result = tag_sentence(sentence)
                st.success("Tagging completed")
                for word, tag in result:
                    st.write(f"**{word}** → `{tag}`")
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Enter a sentence.")
elif page == "🧠 How It Works":
    st.markdown("### Flow\n1. Each word becomes an integer ID.\n2. Padding makes sequences equal length.\n3. Embedding creates vectors.\n4. `SimpleRNN(return_sequences=True)` returns an output for every timestep.\n5. `TimeDistributed(Dense)` predicts a POS tag for every word.")
elif page == "🏗️ Architecture":
    st.code("""Many Input Words
      ↓
Word IDs + Padding
      ↓
Embedding
      ↓
SimpleRNN (64)
return_sequences=True
      ↓
TimeDistributed Dense
      ↓
Softmax at each timestep
      ↓
Many POS Tags""")
elif page == "📚 Project Details":
    st.markdown("### Objective\nDemonstrate **Many-to-Many aligned sequence labeling** with Simple RNN.\n\n### Example\n`the dog runs` → `DET NOUN VERB`\n\n### Evaluation\nToken-level accuracy is reported.\n\n> The bundled dataset is intentionally tiny so the project runs immediately. Replace it with a large annotated POS dataset for meaningful generalization.")
