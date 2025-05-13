import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

class StarburstAnswerOutput(BaseModel):
    topic: str = Field(..., description="The original topic")
    answer_who: List[str] = Field(..., description="List of entities involved")
    answer_what: List[str] = Field(..., description="List of actions or events")
    answer_when: List[str] = Field(..., description="List of times or time ranges")
    answer_where: List[str] = Field(..., description="List of locations")
    answer_why: List[str] = Field(..., description="List of motivations or explanations")


st.sidebar.markdown("### About")
st.sidebar.write("This app is a simple demonstration of the Starburst Strucutred Analytic Technique using OpenAI's GPT-4 model.")

st.sidebar.markdown("### API Key")
api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

st.sidebar.markdown("### License")
st.sidebar.write("This app is for educational purposes only. Please refer to OpenAI's terms of service.")

st.sidebar.markdown("### Disclaimer")
st.sidebar.write("The answers provided by this tool using the OpenAI GPT4o model are generated based on the input topic and may not be accurate. Use at your own risk.")

st.sidebar.markdown("### Contact")
st.sidebar.write("For any questions or feedback, please reach out to us. https://taurus.blue")


st.title("LLM SATS FTW - Starburst")
st.write("This app uses OpenAI's GPT-4 to answer topics in a structured JSON format.")

st.subheader("📝 Input")
topic = st.text_input("❓ Your Scenario", placeholder="e.g. A ransomware attack on a hospital")


if api_key and topic:
    parser = PydanticOutputParser(pydantic_object=StarburstAnswerOutput)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template="Brainstorm a series of 3-5 questions under who, what, when, where, why about the topic and return the result in JSON format.\n{format_instructions}\nTopic: {topic}",
        input_variables=["topic"],
        partial_variables={"format_instructions": format_instructions}
    )

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)
        messages = prompt.format_prompt(topic=topic).to_messages()
        response = llm.invoke(messages)
        result = parser.parse(response.content)

        st.subheader("📦 Output")
        st.json(result.model_dump())
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")