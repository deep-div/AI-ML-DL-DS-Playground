import streamlit as st
import torch
import logging
import time
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

# Configure page
st.set_page_config(page_title="🌐 Translator", page_icon="🌐")

# Device detection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    logging.warning("⚠️ GPU not found — using CPU (translation may be slow).")

# Language mapping
lang_id = {
    "Afrikaans": "af", "Amharic": "am", "Arabic": "ar", "Asturian": "ast",
    "Azerbaijani": "az", "Bashkir": "ba", "Belarusian": "be", "Bulgarian": "bg",
    "Bengali": "bn", "Breton": "br", "Bosnian": "bs", "Catalan": "ca",
    "Cebuano": "ceb", "Czech": "cs", "Welsh": "cy", "Danish": "da",
    "German": "de", "Greeek": "el", "English": "en", "Spanish": "es",
    "Estonian": "et", "Persian": "fa", "Fulah": "ff", "Finnish": "fi",
    "French": "fr", "Western Frisian": "fy", "Irish": "ga", "Gaelic": "gd",
    "Galician": "gl", "Gujarati": "gu", "Hausa": "ha", "Hebrew": "he",
    "Hindi": "hi", "Croatian": "hr", "Haitian": "ht", "Hungarian": "hu",
    "Armenian": "hy", "Indonesian": "id", "Igbo": "ig", "Iloko": "ilo",
    "Icelandic": "is", "Italian": "it", "Japanese": "ja", "Javanese": "jv",
    "Georgian": "ka", "Kazakh": "kk", "Central Khmer": "km", "Kannada": "kn",
    "Korean": "ko", "Luxembourgish": "lb", "Ganda": "lg", "Lingala": "ln",
    "Lao": "lo", "Lithuanian": "lt", "Latvian": "lv", "Malagasy": "mg",
    "Macedonian": "mk", "Malayalam": "ml", "Mongolian": "mn", "Marathi": "mr",
    "Malay": "ms", "Burmese": "my", "Nepali": "ne", "Dutch": "nl",
    "Norwegian": "no", "Northern Sotho": "ns", "Occitan": "oc", "Oriya": "or",
    "Panjabi": "pa", "Polish": "pl", "Pushto": "ps", "Portuguese": "pt",
    "Romanian": "ro", "Russian": "ru", "Sindhi": "sd", "Sinhala": "si",
    "Slovak": "sk", "Slovenian": "sl", "Somali": "so", "Albanian": "sq",
    "Serbian": "sr", "Swati": "ss", "Sundanese": "su", "Swedish": "sv",
    "Swahili": "sw", "Tamil": "ta", "Thai": "th", "Tagalog": "tl",
    "Tswana": "tn", "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur",
    "Uzbek": "uz", "Vietnamese": "vi", "Wolof": "wo", "Xhosa": "xh",
    "Yiddish": "yi", "Yoruba": "yo", "Chinese": "zh", "Zulu": "zu",
}

# Cache model/tokenizer loading
@st.cache_resource
def load_model():
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B").to(device)
    model.eval()
    return tokenizer, model

# Title
st.title("🌍 M2M100 Language Translator")
st.markdown("🔁 Translate text between **100+ languages** using Facebook's `M2M100` multilingual model.")

# Text input
user_input = st.text_area(
    "✏️ Enter your text below:",
    height=200,
    max_chars=5120,
    placeholder="E.g. Hello, how are you?"
)

# Language selections (default: English → Hindi)
col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("🌐 Source Language", sorted(lang_id.keys()), index=list(lang_id.keys()).index("English"))
with col2:
    target_lang = st.selectbox("🔁 Target Language", sorted(lang_id.keys()), index=list(lang_id.keys()).index("Hindi"))

# Translate Button
if st.button("🚀 Translate", disabled=(not user_input.strip())):
    with st.spinner("Translating... Please wait"):
        start = time.time()
        tokenizer, model = load_model()

        src = lang_id[source_lang]
        tgt = lang_id[target_lang]

        tokenizer.src_lang = src
        with torch.no_grad():
            encoded = tokenizer(user_input, return_tensors="pt").to(device)
            output = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(tgt)
            )
            result = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        end = time.time()
        st.success("✅ Translation complete!")
        st.markdown("### 📝 Translated Text")
        st.text_area("Output", value=result, height=150, disabled=True)
        st.caption(f"⏱️ Time taken: {round(end - start, 2)} seconds")

# Optional reset
st.markdown("---")
if st.button("🔄 Reset"):
    st.experimental_rerun()
