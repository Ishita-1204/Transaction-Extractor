import io
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader, PdfWriter

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

st.set_page_config(page_title="PDF Transaction Extractor", layout="wide")


# -----------------------------
# PDF HELPERS
# -----------------------------
def unlock_pdf_bytes(pdf_bytes: bytes, password: str = "") -> bytes:
    """
    If password is provided and PDF is encrypted, decrypt it and return unlocked bytes.
    Otherwise return original bytes.
    """
    input_stream = io.BytesIO(pdf_bytes)
    reader = PdfReader(input_stream)

    if reader.is_encrypted:
        if not password:
            raise ValueError("This PDF is password protected. Enter the password.")

        decrypt_result = reader.decrypt(password)
        if decrypt_result == 0:
            raise ValueError("Incorrect PDF password.")

        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        output_stream = io.BytesIO()
        writer.write(output_stream)
        return output_stream.getvalue()

    return pdf_bytes


def extract_text_preview(pdf_bytes: bytes) -> str:
    """
    Optional PDF text preview for debugging.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        chunks = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            chunks.append(f"\n--- PAGE {i} ---\n{text}")
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def get_total_pages(pdf_bytes: bytes, password: str = "") -> int:
    unlocked_pdf = unlock_pdf_bytes(pdf_bytes, password)
    reader = PdfReader(io.BytesIO(unlocked_pdf))
    return len(reader.pages)


# -----------------------------
# GEMINI HELPERS
# -----------------------------
def get_gemini_client():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in Streamlit secrets or .env file.")
    return genai.Client(api_key=GEMINI_API_KEY)


def extract_json_block(text: str):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start_array = text.find("[")
    end_array = text.rfind("]")
    if start_array != -1 and end_array != -1 and end_array > start_array:
        candidate = text[start_array:end_array + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = text[start_obj:end_obj + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Could not parse JSON returned by Gemini.")


def normalize_date(value):
    if value is None:
        return ""
    value = str(value).strip()
    if value.lower() in {"nan", "none", "null", ""}:
        return ""

    patterns = [
        r"\b\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
    ]
    for pat in patterns:
        m = re.search(pat, value)
        if m:
            return m.group(0)
    return value


def normalize_amount(value):
    """
    Preserve amount as shown:
    - keep commas
    - keep decimals
    - keep Cr
    - remove minus sign
    - ignore plain integers like reward points
    - support Indian number format like 1,88,800.00
    """
    if value is None:
        return ""

    original = str(value).strip()
    if original.lower() in {"nan", "none", "null", ""}:
        return ""

    has_cr = bool(re.search(r"\bCr\b", original, flags=re.IGNORECASE))

    cleaned = (
        original.replace("₹", "")
        .replace("Rs.", "")
        .replace("Rs", "")
        .strip()
    )

    cleaned_no_cr = re.sub(r"\bCr\b", "", cleaned, flags=re.IGNORECASE).strip()

    match = re.search(r"-?(?:\d{1,3}(?:,\d{2,3})+|\d+)\.\d{2}", cleaned_no_cr)
    if match:
        amount = match.group(0).replace("-", "")
        return f"{amount} Cr" if has_cr else amount

    return ""


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = [
        "Transaction Type",
        "Date",
        "Transaction Description",
        "Amount (in Rs.)",
    ]

    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[expected_cols].copy()

    df["Transaction Type"] = df["Transaction Type"].astype(str).str.strip()
    df["Date"] = df["Date"].apply(normalize_date)
    df["Transaction Description"] = (
        df["Transaction Description"]
        .astype(str)
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["Amount (in Rs.)"] = df["Amount (in Rs.)"].apply(normalize_amount)

    df = df[
        ~(
            (df["Date"] == "")
            & (df["Transaction Description"] == "")
            & (df["Amount (in Rs.)"] == "")
        )
    ].reset_index(drop=True)

    return df


def parse_with_gemini(pdf_bytes: bytes):
    client = get_gemini_client()

    prompt = """
You are extracting transactions from the ENTIRE credit-card statement PDF.

Return ONLY valid JSON with this exact schema:
{
  "domestic": [
    {
      "Transaction Type": "Domestic",
      "Date": "",
      "Transaction Description": "",
      "Amount (in Rs.)": ""
    }
  ],
  "international": [
    {
      "Transaction Type": "International",
      "Date": "",
      "Transaction Description": "",
      "Amount (in Rs.)": ""
    }
  ]
}

Rules:
1. Read the ENTIRE PDF, all pages.
2. Extract rows from Domestic and International sections separately across all pages.
3. Extract only:
   - Transaction Type
   - Date
   - Transaction Description
   - Amount (in Rs.)
4. Preserve amount exactly as shown in the statement.
5. Preserve commas, decimals, and Cr.
6. Handle Indian comma format exactly, such as 1,88,800.00.
7. Never change 28,500.00 to 28,000.00.
8. Never use reward points as amount.
9. Ignore headers, summaries, totals, reward points, footers, and page labels.
10. If a row spans multiple lines, merge description into one row.
11. If the same transaction continues on later pages, keep proper row continuity.
12. Do not invent rows.
13. If a number is unclear, prefer the visually exact rightmost amount in the row.

Return JSON only.
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            {
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": pdf_bytes,
                }
            },
        ],
    )

    raw_text = response.text
    parsed = extract_json_block(raw_text)

    domestic = parsed.get("domestic", []) if isinstance(parsed, dict) else []
    international = parsed.get("international", []) if isinstance(parsed, dict) else []

    domestic_df = clean_dataframe(pd.DataFrame(domestic))
    international_df = clean_dataframe(pd.DataFrame(international))

    if not domestic_df.empty:
        domestic_df["Transaction Type"] = "Domestic"
    if not international_df.empty:
        international_df["Transaction Type"] = "International"

    combined_df = pd.concat([domestic_df, international_df], ignore_index=True)
    combined_df = clean_dataframe(combined_df)

    return domestic_df, international_df, combined_df, raw_text


# -----------------------------
# FILE HELPERS
# -----------------------------
def read_existing_uploaded_file(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()

    if ext == ".csv":
        return pd.read_csv(uploaded_file)

    if ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(uploaded_file)
        if "Combined" in xls.sheet_names:
            return pd.read_excel(uploaded_file, sheet_name="Combined")
        return pd.read_excel(uploaded_file)

    raise ValueError("Only CSV or Excel files are supported.")


def combine_and_deduplicate(old_df: pd.DataFrame, new_df: pd.DataFrame):
    old_df = clean_dataframe(old_df) if old_df is not None and not old_df.empty else pd.DataFrame()
    new_df = clean_dataframe(new_df) if new_df is not None and not new_df.empty else pd.DataFrame()

    if old_df.empty:
        combined = new_df.copy()
    else:
        combined = pd.concat([old_df, new_df], ignore_index=True)

    dedup_cols = [
        "Transaction Type",
        "Date",
        "Transaction Description",
        "Amount (in Rs.)",
    ]

    combined = combined.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)
    return combined


def build_excel_bytes(combined_df: pd.DataFrame, domestic_df: pd.DataFrame, international_df: pd.DataFrame):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        combined_df.to_excel(writer, index=False, sheet_name="Combined")
        domestic_df.to_excel(writer, index=False, sheet_name="Domestic")
        international_df.to_excel(writer, index=False, sheet_name="International")
    output.seek(0)
    return output.getvalue()


def build_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# PIPELINE
# -----------------------------
def run_pipeline(pdf_bytes: bytes, pdf_password: str = ""):
    progress_bar = st.progress(0)
    log_box = st.empty()

    log_box.markdown("**🔓 Opening PDF...**")
    unlocked_pdf = unlock_pdf_bytes(pdf_bytes, pdf_password)
    total_pages = get_total_pages(pdf_bytes, pdf_password)
    progress_bar.progress(35)

    log_box.markdown(f"**📄 Reading full PDF preview text ({total_pages} pages)...**")
    preview_text = extract_text_preview(unlocked_pdf)
    progress_bar.progress(65)

    log_box.markdown("**🤖 Processing full PDF with Gemini...**")
    domestic_df, international_df, combined_df, raw_model_output = parse_with_gemini(
        unlocked_pdf
    )
    progress_bar.progress(100)

    log_box.markdown("**✅ Extraction completed!**")
    return domestic_df, international_df, combined_df, preview_text, raw_model_output, total_pages


# -----------------------------
# UI
# -----------------------------
st.title("PDF Domestic / International Transaction Extractor")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in Streamlit secrets or .env file")
    st.stop()

with st.expander("Setup"):
    st.write("Run locally with:")
    st.code("python -m streamlit run app.py", language="bash")
    st.write("Gemini model used in code:")
    st.code(GEMINI_MODEL)

pdf_password = st.text_input("PDF Password (optional)", type="password")
uploaded_pdf = st.file_uploader("Upload PDF statement", type=["pdf"])

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.read()

    try:
        total_pages = get_total_pages(pdf_bytes, pdf_password)
    except Exception as e:
        st.error(f"Could not open PDF: {e}")
        st.stop()

    try:
        domestic_df, international_df, combined_df, preview_text, raw_model_output, total_pages = run_pipeline(
            pdf_bytes,
            pdf_password,
        )
    except Exception as e:
        st.error(f"Extraction failed: {e}")
        st.stop()

    st.success("Done.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total pages", int(total_pages))
    col2.metric("Domestic rows", len(domestic_df))
    col3.metric("International rows", len(international_df))
    col4.metric("Combined rows", len(combined_df))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Domestic", "International", "Combined", "PDF Text Preview", "Raw Gemini Output"]
    )

    with tab1:
        if domestic_df.empty:
            st.warning("No Domestic transactions found.")
        else:
            st.dataframe(domestic_df, use_container_width=True)

    with tab2:
        if international_df.empty:
            st.warning("No International transactions found.")
        else:
            st.dataframe(international_df, use_container_width=True)

    with tab3:
        if combined_df.empty:
            st.warning("No transactions found.")
        else:
            st.dataframe(combined_df, use_container_width=True)

    with tab4:
        if preview_text.strip():
            st.text(preview_text[:30000])
        else:
            st.info("No text preview available.")

    with tab5:
        st.text(raw_model_output[:30000])

    st.markdown("---")
    st.subheader("Save / Append Output")

    save_mode = st.radio(
        "Choose save mode",
        ["Create new output file", "Append to existing CSV/Excel"],
        horizontal=True,
    )

    output_type = st.radio(
        "Choose output type",
        ["CSV", "Excel"],
        horizontal=True,
    )

    final_df = combined_df.copy()

    if save_mode == "Append to existing CSV/Excel":
        existing_file = st.file_uploader(
            "Upload existing CSV or Excel file to append into",
            type=["csv", "xlsx", "xls"],
            key="existing_file",
        )

        if existing_file is not None:
            try:
                old_df = read_existing_uploaded_file(existing_file)
                final_df = combine_and_deduplicate(old_df, combined_df)
                st.info(f"Existing rows: {len(old_df)} | Final rows after append/deduplicate: {len(final_df)}")
                st.dataframe(final_df, use_container_width=True)
            except Exception as e:
                st.error(f"Could not read existing file: {e}")
                st.stop()
        else:
            st.warning("Upload an existing CSV/Excel file to append.")

    output_name = st.text_input("Output file name (without extension)", value="transactions_output").strip()
    if output_name == "":
        output_name = "transactions_output"

    left_col, right_col = st.columns(2)

    with left_col:
        if output_type == "CSV":
            csv_bytes = build_csv_bytes(final_df)
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name=f"{output_name}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            excel_bytes = build_excel_bytes(final_df, domestic_df, international_df)
            st.download_button(
                label="Download Excel",
                data=excel_bytes,
                file_name=f"{output_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with right_col:
        st.write("Preview of final file content:")
        st.dataframe(final_df, use_container_width=True)

else:
    st.info("Upload a PDF to begin.")