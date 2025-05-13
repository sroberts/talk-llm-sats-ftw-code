import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator
from typing import List

class AchHypoGenAnswerOutput(BaseModel):
    topic: str = Field(..., description="The original topic")
    answer: List[str] = Field(..., description="Generate a list of hypotheses for the topic")


class AchEvidenceOutput(BaseModel):
    topic: str = Field(..., description="The original topic")
    hypothesis: str = Field(..., description="A specific hypothesis related to the topic")
    evidence: List[str] = Field(..., description="Generate a list of evidence for the hypothesis")


class ACHScore(BaseModel):
    topic: str = Field(..., description="The central topic or issue being analyzed")
    hypothesis: str = Field(..., description="A specific hypothesis related to the topic")
    evidence: str = Field(..., description="A piece of evidence relevant to the hypothesis")
    score: int = Field(..., description="Score from -5 (Very Unlikely) to 5 (Very Likely)")

    @field_validator("score")
    def score_must_be_in_range(cls, v):
        if v < -5 or v > 5:
            raise ValueError("Score must be between -5 and 5")
        return v


st.sidebar.markdown("### About")
st.sidebar.write("This app is a simple demonstration of the Analysis of Competing Hypothesis Strucutred Analytic Technique using OpenAI's GPT-4 model.")

st.sidebar.markdown("### API Key")
api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

st.sidebar.markdown("### License")
st.sidebar.write("This app is for educational purposes only. Please refer to OpenAI's terms of service.")

st.sidebar.markdown("### Disclaimer")
st.sidebar.write("The answers provided by this tool using the OpenAI GPT4o model are generated based on the input topic and may not be accurate. Use at your own risk.")

st.sidebar.markdown("### Contact")
st.sidebar.write("For any questions or feedback, please reach out to us. https://taurus.blue")


st.title("LLM SATS FTW - ACH")
st.write("This app uses OpenAI's GPT-4 to do an Analysis of Competing Hypothesis on a given topic in a structured JSON format.")

st.subheader("📝 Input")
topic = st.text_input("❓ Your Scenario", placeholder="e.g. What is the attribution of the XZ Backdoor?")


if api_key and topic:
    st.info("Step 1: Generating hypotheses...")
    # Step 1: Generate hypotheses
    hypo_parser = PydanticOutputParser(pydantic_object=AchHypoGenAnswerOutput)
    hypo_format = hypo_parser.get_format_instructions()
    hypo_prompt = PromptTemplate(
        template="Generate a list of 3-5 plausible hypotheses for the following topic and return the result in JSON format.\n{format_instructions}\n\nTopic: {topic}",
        input_variables=["topic"],
        partial_variables={"format_instructions": hypo_format}
    )
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)
    hypo_messages = hypo_prompt.format_prompt(topic=topic).to_messages()
    hypo_response = llm.invoke(hypo_messages)
    hypo_result = hypo_parser.parse(hypo_response.content)
    hypotheses = hypo_result.answer
    st.success(f"Generated {len(hypotheses)} hypotheses.")
    st.write(hypotheses)

    st.info("Step 2: Generating evidence for each hypothesis...")
    # Step 2: Generate evidence for each hypothesis
    evidence_parser = PydanticOutputParser(pydantic_object=AchEvidenceOutput)
    evidence_format = evidence_parser.get_format_instructions()
    evidence_dict = {}
    for h in hypotheses:
        st.write(f"Generating evidence for hypothesis: {h}")
        evidence_prompt = PromptTemplate(
            template="Given the topic and hypothesis, generate a list of 3-5 pieces of evidence (for or against) and return the result in JSON format.\n{format_instructions}\n\nTopic: {topic}\nHypothesis: {hypothesis}",
            input_variables=["topic", "hypothesis"],
            partial_variables={"format_instructions": evidence_format}
        )
        evidence_messages = evidence_prompt.format_prompt(topic=topic, hypothesis=h).to_messages()
        evidence_response = llm.invoke(evidence_messages)
        evidence_result = evidence_parser.parse(evidence_response.content)
        evidence_dict[h] = evidence_result.evidence
        st.write(evidence_result.evidence)

    # Gather all unique evidence across all hypotheses
    all_evidence = sorted(set(ev for ev_list in evidence_dict.values() for ev in ev_list))

    st.info("Step 3: Scoring each (hypothesis, evidence) pair...")
    # Step 3: Score each (hypothesis, evidence) pair
    score_parser = PydanticOutputParser(pydantic_object=ACHScore)
    score_format = score_parser.get_format_instructions()
    score_matrix = {}
    for h in hypotheses:
        score_matrix[h] = {}
        st.write(f"\n**Hypothesis:** {h}")
        for e in all_evidence:
            score_prompt = PromptTemplate(
                template="Given the topic, hypothesis, and evidence, assign a score from -5 (Very Unlikely) to 5 (Very Likely) for how strongly the evidence supports or disproves the hypothesis. Negative scores mean the evidence disproves the hypothesis. Return the result in JSON format.\n{format_instructions}\n\nTopic: {topic}\nHypothesis: {hypothesis}\nEvidence: {evidence}",
                input_variables=["topic", "hypothesis", "evidence"],
                partial_variables={"format_instructions": score_format}
            )
            score_messages = score_prompt.format_prompt(topic=topic, hypothesis=h, evidence=e).to_messages()
            score_response = llm.invoke(score_messages)
            score_result = score_parser.parse(score_response.content)
            score_matrix[h][e] = score_result.score
            st.markdown(f"- <span style='color:gray'><b>Evidence:</b></span> <i>{e}</i> <span style='color:gray'>|</span> <b>Score:</b> <span style='color:{'green' if score_result.score > 0 else 'red' if score_result.score < 0 else 'black'}'>{score_result.score}</span>", unsafe_allow_html=True)

    st.info("Step 4: Displaying the ACH matrix...")
    # Step 4: Display the matrix
    st.subheader("📦 ACH Matrix")
    st.write("Rows: Hypotheses, Columns: Evidence, Values: Score")
    import pandas as pd
    matrix_data = []
    for h in hypotheses:
        row = [score_matrix[h].get(e, None) for e in all_evidence]
        matrix_data.append(row)
    df = pd.DataFrame(matrix_data, index=hypotheses, columns=all_evidence)
    st.dataframe(df)

    st.info("Step 5: Calculating and highlighting top results...")
    # Step 5: Call out highest scoring hypothesis and evidence
    # Highest scoring hypothesis: sum of scores across all evidence
    hypo_scores = {h: sum([s for s in score_matrix[h].values() if isinstance(s, int)]) for h in hypotheses}
    top_hypo = max(hypo_scores, key=hypo_scores.get)
    st.success(f"Highest scoring hypothesis: {top_hypo} (Total score: {hypo_scores[top_hypo]})")

    # Highest scoring evidence: sum of scores across all hypotheses
    evidence_scores = {e: sum([score_matrix[h].get(e, 0) for h in hypotheses if isinstance(score_matrix[h].get(e, 0), int)]) for e in all_evidence}
    top_evidence = max(evidence_scores, key=evidence_scores.get)
    st.info(f"Highest scoring piece of evidence: {top_evidence} (Total score: {evidence_scores[top_evidence]})")

    # Step 6: Downloadable CSV of the matrix
    st.info("Step 6: Download the ACH Matrix as CSV")
    csv = df.to_csv()
    st.download_button(
        label="Download ACH Matrix as CSV",
        data=csv,
        file_name="ach_matrix.csv",
        mime="text/csv"
    )