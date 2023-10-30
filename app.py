import streamlit as st
import pickle
from kendra_bedrock_query import kendraSearch
from query_against_openSearch import opensearch_query
from snowflake_bedrock_query import snowflake_answer
from PIL import Image
from image_generation import image_generator
# loading in the classification model we created
model = pickle.load(open('model/model.pkl', 'rb'))
# loading the vectorizer we created
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
# Creating a header on the frontend of the application
st.markdown("<h1 style='text-align: center; color: gold;'>One Chat-Bot to Rule Them All</h1>", unsafe_allow_html=True)
# importing all the architecture diagram images
kendra_image = Image.open("images/kendra.png")
opensearch_image = Image.open("images/opensearch.png")
snowflake_image = Image.open("images/snowflake.png")
image_gen = Image.open("images/image_generation.png")
architecture = Image.open("images/architecture.png")
# configuring values for session state
if "messages" not in st.session_state:
    st.session_state.messages = []
# writing the message that is stored in session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# adding some special effects from the UI perspective
st.balloons()
# evaluating st.chat_input and determining if a question has been input
if question := st.chat_input("Ask about Harry Potter, Amazon Documentation, the MOMA database or create an image!"):
    # with the user icon, write the question to the front end
    with st.chat_message("user"):
        st.markdown(question)
    # append the question and the role (user) as a message to the session state
    st.session_state.messages.append({"role": "user",
                                      "content": question})
    # respond as the assistant with the answer
    with st.chat_message("assistant"):
        # making sure there are no messages present when generating the answer
        message_placeholder = st.empty()
        # using the classification model we created to classify the request to ensure it is routed to the appropriate "micro-service"
        model_prediction = model.predict(vectorizer.transform([question]))
        # if the classification model classifies the question as an AWS Documentation question send the question to Kendra
        if model_prediction[0] == 'AWS Documentation':
            # call the Kendra Search Function as this is where our AWS Documentation is stored, and pass in users question
            answer = kendraSearch(question)
            # format the answer and display on the frontend
            message_placeholder.markdown(f""" Answer:
                {answer} """)
            # have a sidebar pop up to highlight the architecture diagram, and the specific model and data store used to answer the question
            with st.sidebar:
                st.header("Architecture Diagram:")
                st.markdown(f""":rainbow[This is an {model_prediction[0]} Question. 
                        All of your AWS Documentation is stored in Amazon Kendra.]""")
                st.image(kendra_image, caption='Kendra RAG Classification')
                st.markdown("<h1 style='text-align: center; color: red;'>Model Used: anthropic.claude-v2 </h1>",
                            unsafe_allow_html=True)
        # if the classification model classifies the question as an Harry Potter question send the question to Amazon OpenSearch
        elif model_prediction[0] == 'Harry Potter':
            # call the opensearch_query Function as this is where our Harry Potter data is stored, and pass in users question
            answer = opensearch_query(question)
            # format the answer and display on the frontend
            message_placeholder.markdown(f""" Answer:
                        {answer} """)
            # have a sidebar pop up to highlight the architecture diagram, and the specific model and data store used to answer the question
            with st.sidebar:
                st.header("Architecture Diagram:")
                st.markdown(f""":rainbow[This is an {model_prediction[0]} Question. 
                        All of your Harry Potter information is stored in Amazon OpenSearch.]""")
                st.image(opensearch_image, caption='OpenSearch RAG Classification')
                st.markdown("<h1 style='text-align: center; color: red;'>Model Used: anthropic.claude-v2 </h1>", unsafe_allow_html=True)
        # if the classification model classifies the question as an SQL question send the question to Snowflake
        elif model_prediction[0] == 'MOMA SQL Database':
            # call the snowflake_answer Function as this is where our SQL MOMA data is stored, and pass in users question
            answer = snowflake_answer(question)
            # format the answer and display on the frontend
            message_placeholder.markdown(f""" Answer:
                {answer[1]}
                The SQL command to get this answer was:""")
            # display the SQL query used to retrieve data in code format
            st.code(answer[0], language="sql")
            # have a sidebar pop up to highlight the architecture diagram, and the specific model and data store used to answer the question
            with st.sidebar:
                st.header("Architecture Diagram:")
                st.markdown(f""" :rainbow[This is an {model_prediction[0]} Question. 
                                All of your MOMA Database information is stored in Snowflake.]""")
                st.image(snowflake_image, caption='Snowflake RAG Classification')
                st.markdown("<h1 style='text-align: center; color: red;'>Model Used: amazon.titan-text-express-v1 </h1>",
                            unsafe_allow_html=True)
        # if the classification model classifies the question as an image generation question send the question to the image generator
        elif model_prediction[0] == 'Image Generation':
            # stored generated image in the session state
            answer = "Generated Image"
            # call the image_generator Function as this is where our image generator service is, and pass in users question
            image_path = image_generator(question)
            # open the path of the generated image
            generated_image = Image.open(image_path)
            # display the image to the front end
            st.image(generated_image, caption="Generated Image")
            # have a sidebar pop up to highlight the architecture diagram, and the specific model used to answer the question
            with st.sidebar:
                st.header("Architecture Diagram:")
                st.markdown(f""":rainbow[This is an {model_prediction[0]} Ask.]""")
                st.image(image_gen, caption="Image Generator Classification")
                st.markdown(
                    "<h1 style='text-align: center; color: red;'>Model Used: stability.stable-diffusion-xl-v0 </h1>",
                    unsafe_allow_html=True)
    # storing the session state
    st.session_state.messages.append({"role": "assistant",
                                      "content": answer})