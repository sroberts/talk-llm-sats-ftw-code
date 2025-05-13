import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal
import json
import requests
import tempfile
import os
from PyPDF2 import PdfReader
import tiktoken
import time

st.sidebar.markdown("### About")
st.sidebar.write("This app is a simple demonstration of the Key Assumptions Check Strucutred Analytic Technique using OpenAI's GPT-4 model.")

st.sidebar.markdown("### API Key")
api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

st.sidebar.markdown("### License")
st.sidebar.write("This app is for educational purposes only. Please refer to OpenAI's terms of service.")

st.sidebar.markdown("### Disclaimer")
st.sidebar.write("The answers provided by this tool using the OpenAI GPT4o model are generated based on the input topic and may not be accurate. Use at your own risk.")

st.sidebar.markdown("### Contact")
st.sidebar.write("For any questions or feedback, please reach out to us. https://taurus.blue")


st.title("LLM SATS FTW - Key Assumption Check")
st.write("This app uses OpenAI's GPT-4 to do a Key Assumption Check on a PDF.")

st.subheader("📝 Input")
pdf_url = st.text_input("🔗 PDF URL", placeholder="e.g. https://arxiv.org/pdf/1234.5678.pdf")

if api_key and pdf_url:
    st.info("Step 1: Downloading and extracting text from PDF...")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_pdf_path = tmp_file.name
        reader = PdfReader(tmp_pdf_path)
        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        os.remove(tmp_pdf_path)
        st.success("PDF downloaded and text extracted.")
        st.write(pdf_text[:2000] + ("..." if len(pdf_text) > 2000 else ""))
    except Exception as e:
        st.error(f"Failed to download or extract PDF: {e}")
        pdf_text = None

    if pdf_text:
        st.info("Step 2: Extracting key assumptions from the document...")
        # Define a simple output schema for assumptions
        class AssumptionsOutput(BaseModel):
            assumptions: List[str] = Field(..., description="A list of key assumptions the document is making")
        parser = PydanticOutputParser(pydantic_object=AssumptionsOutput)
        format_instructions = parser.get_format_instructions()
        prompt = PromptTemplate(
            template="Read the following document and return a list of the key assumptions the text is making in JSON format.\n{format_instructions}\n\nDocument: {document}",
            input_variables=["document"],
            partial_variables={"format_instructions": format_instructions}
        )
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)

        # Tokenize and chunk the text
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = enc.encode(pdf_text)
        chunk_size = 7500
        chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
        st.info(f"Document split into {len(chunks)} chunk(s) for processing.")
        all_assumptions = set()
        for idx, chunk in enumerate(chunks):
            st.info(f"Processing chunk {idx+1} of {len(chunks)}...")
            chunk_text = enc.decode(chunk)
            messages = prompt.format_prompt(document=chunk_text).to_messages()
            response = llm.invoke(messages)
            result = parser.parse(response.content)
            all_assumptions.update([a.strip() for a in result.assumptions])
            # Add a delay to avoid hitting OpenAI rate limits
            time.sleep(10)
        st.success(f"Extracted {len(all_assumptions)} unique key assumptions from the document.")
        st.write(sorted(all_assumptions))