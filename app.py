import streamlit as st
import pickle
from kendra_bedrock_query import kendraSearch
from query_against_openSearch import answer_query
from snowflake_bedrock_query import snowflake_answer
from PIL import Image
from image_generation import image_generator

model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

st.markdown("<h1 style='text-align: center; color: gold;'>One Chat-Bot to Rule Them All</h1>", unsafe_allow_html=True)

kendra_image = Image.open("images/kendra.png")
opensearch_image = Image.open("images/opensearch.png")
snowflake_image = Image.open("images/snowflake.png")
image_gen = Image.open("images/image_generation.png")
architecture = Image.open("images/architecture.png")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.balloons()

if prompt := st.chat_input("Ask about Harry Potter, Amazon Documentation, the MOMA database or create an image!"):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user",
                                      "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        model_prediction = model.predict(vectorizer.transform([prompt]))
        print(model_prediction)
        if model_prediction[0] == 'AWS Documentation':
            answer = kendraSearch(prompt)
            message_placeholder.markdown(f""" Answer:
                {answer} """)
            with st.sidebar:
                st.header("Architecture Diagram:")
                st.markdown(f""":rainbow[This is an {model_prediction[0]} Question. 
                        All of your AWS Documentation is stored in Amazon Kendra.]""")
                st.image(kendra_image, caption='Kendra RAG Classification')
                st.markdown("<h1 style='text-align: center; color: red;'>Model Used: anthropic.claude-v2 </h1>",
                            unsafe_allow_html=True)
        elif model_prediction[0] == 'Harry Potter':
            answer = answer_query(prompt)
            message_placeholder.markdown(f""" Answer:
                        {answer} """)
            with st.sidebar:
                st.header("Architecture Diagram:")
                st.markdown(f""":rainbow[This is an {model_prediction[0]} Question. 
                        All of your Harry Potter information is stored in Amazon OpenSearch.]""")
                st.image(opensearch_image, caption='OpenSearch RAG Classification')
                st.markdown("<h1 style='text-align: center; color: red;'>Model Used: anthropic.claude-v2 </h1>", unsafe_allow_html=True)
        elif model_prediction[0] == 'MOMA SQL Database':
            answer = snowflake_answer(prompt)
            message_placeholder.markdown(f""" Answer:
                {answer[1]}
                The SQL command to get this answer was:""")
            st.code(answer[0], language="sql")
            with st.sidebar:
                st.header("Architecture Diagram:")
                st.markdown(f""" :rainbow[This is an {model_prediction[0]} Question. 
                                All of your MOMA Database information is stored in Snowflake.]""")
                st.image(snowflake_image, caption='Snowflake RAG Classification')
                st.markdown("<h1 style='text-align: center; color: red;'>Model Used: amazon.titan-text-express-v1 </h1>",
                            unsafe_allow_html=True)
        elif model_prediction[0] == 'Image Generation':
            answer = "Generated Image"
            image_path = image_generator(prompt)
            generated_image = Image.open(image_path)
            st.image(generated_image, caption="Generated Image")
            with st.sidebar:
                st.header("Architecture Diagram:")
                st.markdown(f""":rainbow[This is an {model_prediction[0]} Ask.]""")
                st.image(image_gen, caption="Image Generator Classification")
                st.markdown(
                    "<h1 style='text-align: center; color: red;'>Model Used: stability.stable-diffusion-xl-v0 </h1>",
                    unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant",
                                      "content": answer})


