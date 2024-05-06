from pandasai.llm.local_llm import LocalLLM
import streamlit as st 
import pandas as pd 
from pandasai import SmartDataframe

model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="llama3_latest"
)

st.title("Data analysis with PandasAI")

# if you want to give the option for the user to use his own . parquet file
# uploaded_file = st.file_uploader("Upload a parquet file", type=['parquet'])

# if uploaded_file is not None:
#     data = pd.read_parquet(uploaded_file)
data= pd.read_parquet(".parquet")
st.write(data.head(3))

df = SmartDataframe(data, config={"llm": model})
prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            st.write(df.chat(prompt))
