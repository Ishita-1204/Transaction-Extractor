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


DATE_TIME_PATTERN = re.compile(
    r"\b\d{2}/\d{2}/\d{4}(?:\s+\d{2}:\d{2}:\d{2})?\b"
)
AMOUNT_CR_PATTERN = re.compile(
    r"[-+]?\d[\d,]*\.\d{2}(?:\s*Cr)?",
    re.IGNORECASE
)
INTEGER_PATTERN = re.compile(r"^\d+$")

def group_boxes_into_rows(boxes, y_threshold=14):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b["cy"], b["x1"]))
    rows = []
    current_row = [boxes[0]]
    current_y = boxes[0]["cy"]

    for box in boxes[1:]:
        if abs(box["cy"] - current_y) <= y_threshold:
            current_row.append(box)
            current_y = (current_y + box["cy"]) / 2
        else:
            rows.append(sorted(current_row, key=lambda b: b["x1"]))
            current_row = [box]
            current_y = box["cy"]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b["x1"]))

    return rows


def is_header_or_noise_row(row_text: str) -> bool:
    low = row_text.lower()
    bad_words = [
        "date",
        "transaction description",
        "feature reward",
        "points",
        "amount",
        "amount (in rs",
        "domestic transactions",
        "international transactions",
        "statement",
        "page ",
        "customer care",
        "reward summary",
        "transaction summary",
        "payment summary",
        "opening balance",
        "closing balance",
        "available credit",
        "important",
        "disclaimer",
        "hdfc bank",
        "gstin",
        "card ending",
    ]
    return any(word in low for word in bad_words)


def split_boxes_by_section(images):
    domestic_boxes = []
    international_boxes = []
    current_section = None

    for img in images:
        page_boxes = extract_ocr_boxes_from_image(img)
        page_rows = group_boxes_into_rows(page_boxes)

        for row in page_rows:
            row_text = " ".join(b["text"] for b in row).strip().lower()

            if "domestic transactions" in row_text:
                current_section = "domestic"
                continue

            if "international transactions" in row_text:
                current_section = "international"
                continue

            if current_section == "domestic":
                domestic_boxes.extend(row)
            elif current_section == "international":
                international_boxes.extend(row)

    return domestic_boxes, international_boxes


def parse_rows_from_boxes(boxes, tx_type="Domestic"):
    rows_grouped = group_boxes_into_rows(boxes)
    parsed_rows = []
    pending = None

    for row in rows_grouped:
        row_text = " ".join(b["text"] for b in row).strip()

        if not row_text or is_header_or_noise_row(row_text):
            continue

        date_tokens = []
        desc_tokens = []
        reward_tokens = []
        amount_tokens = []

        for b in row:
            x = b["cx"]
            txt = b["text"].strip()

            if x < 170:
                date_tokens.append(txt)
            elif x < 600:
                desc_tokens.append(txt)
            elif x < 740:
                reward_tokens.append(txt)
            else:
                amount_tokens.append(txt)

        date_text = " ".join(date_tokens).strip()
        desc_text = " ".join(desc_tokens).strip()
        reward_text = " ".join(reward_tokens).strip()
        amount_text = " ".join(amount_tokens).strip()

        date_match = DATE_TIME_PATTERN.search(date_text)
        date_val = date_match.group(0) if date_match else ""

        amount_match = AMOUNT_CR_PATTERN.search(amount_text)
        amount_val = amount_match.group(0) if amount_match else ""

        reward_val = ""
        for token in reward_text.split():
            token_clean = token.replace(",", "").strip()
            if INTEGER_PATTERN.fullmatch(token_clean):
                reward_val = token_clean
                break

        if date_val and amount_val:
            if pending:
                parsed_rows.append(pending)

            pending = {
                "Transaction Type": tx_type,
                "Date": date_val,
                "Transaction Description": desc_text.strip(),
                "Feature Reward Points": reward_val,
                "Amount (in Rs.)": amount_val.strip(),
            }
        else:
            if pending and desc_text:
                pending["Transaction Description"] = (
                    pending["Transaction Description"] + " " + desc_text
                ).strip()

        # if row has reward only and no desc, attach reward to pending if empty
        if pending and reward_val and not pending.get("Feature Reward Points"):
            pending["Feature Reward Points"] = reward_val

    if pending:
        parsed_rows.append(pending)

    df = pd.DataFrame(parsed_rows)

    if not df.empty:
        df["Transaction Description"] = (
            df["Transaction Description"]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["Feature Reward Points"] = (
            df["Feature Reward Points"].astype(str).replace("nan", "").str.strip()
        )
        df["Amount (in Rs.)"] = (
            df["Amount (in Rs.)"]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    return df


def normalize_amount_column(df: pd.DataFrame):
    if "Amount (in Rs.)" in df.columns:
        df["Amount (in Rs.)"] = (
            df["Amount (in Rs.)"]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    return df


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
    if old_df is None or old_df.empty:
        combined = new_df.copy()
    else:
        combined = pd.concat([old_df, new_df], ignore_index=True)

    dedup_cols = [
        c
        for c in [
            "Transaction Type",
            "Date",
            "Transaction Description",
            "Feature Reward Points",
            "Amount (in Rs.)",
        ]
        if c in combined.columns
    ]

    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")

    combined = combined.reset_index(drop=True)
    return combined


def build_excel_bytes(
    combined_df: pd.DataFrame, domestic_df: pd.DataFrame, international_df: pd.DataFrame
):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        combined_df.to_excel(writer, index=False, sheet_name="Combined")
        domestic_df.to_excel(writer, index=False, sheet_name="Domestic")
        international_df.to_excel(writer, index=False, sheet_name="International")

    output.seek(0)
    return output.getvalue()


def build_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


def extract_transactions_from_pdf(pdf_bytes: bytes):
    images = pdf_to_images(pdf_bytes)

    progress_bar = st.progress(0)
    log_box = st.empty()

    total_pages = len(images)
    page_texts = []

    for i, img in enumerate(images, start=1):
        log_box.markdown(f"**📄 OCR page {i}/{total_pages}...**")
        page_boxes = extract_ocr_boxes_from_image(img)
        preview_text = " ".join(
            [b["text"] for b in sorted(page_boxes, key=lambda x: (x["cy"], x["x1"]))]
        )
        page_texts.append((i, preview_text[:30000]))
        progress_bar.progress(i / total_pages)

    log_box.markdown("**🔍 Splitting Domestic and International sections...**")
    domestic_boxes, international_boxes = split_boxes_by_section(images)

    log_box.markdown("**📊 Parsing Domestic transactions...**")
    domestic_df = parse_rows_from_boxes(domestic_boxes, tx_type="Domestic")
    domestic_df = normalize_amount_column(domestic_df)

    log_box.markdown("**🌍 Parsing International transactions...**")
    international_df = parse_rows_from_boxes(international_boxes, tx_type="International")
    international_df = normalize_amount_column(international_df)

    if not domestic_df.empty or not international_df.empty:
        combined_df = pd.concat([domestic_df, international_df], ignore_index=True)
    else:
        combined_df = pd.DataFrame(
            columns=[
                "Transaction Type",
                "Date",
                "Transaction Description",
                "Feature Reward Points",
                "Amount (in Rs.)",
            ]
        )

    log_box.markdown("**✅ Extraction completed!**")
    return domestic_df, international_df, combined_df, page_texts


st.title("PDF Domestic / International Transaction Extractor")
st.write(
    "Upload a PDF statement, extract the transactions table, preview it, and save it as a new CSV/Excel or append it to an existing file."
)

with st.expander("Important setup note for Windows"):
    st.write("Run using:")
    st.code("python -m streamlit run app.py", language="bash")
    st.write("Poppler path used in code:")
    st.code(POPPLER_PATH)

uploaded_pdf = st.file_uploader("Upload PDF statement", type=["pdf"])

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.read()

    with st.spinner("Starting extraction..."):
        try:
            domestic_df, international_df, combined_df, page_texts = extract_transactions_from_pdf(
                pdf_bytes
            )
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()

    st.success("Extraction completed.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Domestic rows", len(domestic_df))
    col2.metric("International rows", len(international_df))
    col3.metric("Combined rows", len(combined_df))

    tab1, tab2, tab3, tab4 = st.tabs(["Domestic", "International", "Combined", "Raw OCR"])

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
        for page_no, text in page_texts:
            with st.expander(f"Page {page_no} OCR text"):
                st.text(text)

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
                st.info(
                    f"Existing rows: {len(old_df)} | Final rows after append/deduplicate: {len(final_df)}"
                )
                st.dataframe(final_df, use_container_width=True)
            except Exception as e:
                st.error(f"Could not read existing file: {e}")
                st.stop()
        else:
            st.warning("Upload an existing CSV/Excel file to append.")

    output_name = st.text_input(
        "Output file name (without extension)",
        value="transactions_output",
    ).strip()

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