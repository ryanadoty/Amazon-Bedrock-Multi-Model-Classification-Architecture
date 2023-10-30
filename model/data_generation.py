import boto3
import json
import botocore.config

# setting up default session with specific CLI profile name
boto3.setup_default_session(profile_name='default')
# extending the timeout period of a bedrock invocation
config = botocore.config.Config(connect_timeout=120, read_timeout=120)
# instantiating the bedrock client
bedrock = boto3.client('bedrock-runtime', 'us-east-1', endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com', config=config)

def prompt(prompt_data):
    """
    This function is used to invoke amazon bedrock and to create sample data that we can later leverage to train our
    classification model.
    :param prompt_data: The prompt we are passing in specifying the task we are asking the LLM to perform.
    :return: The output of the LLM, it is designed to be used to return a list of questions that relate to a specific task
    you are trying to perform with your application, or a task you would like to classify.
    """
    # setting the parameters of the request to amazon bedrock
    body = json.dumps({"prompt": prompt_data,
                       "max_tokens_to_sample": 8191,
                       "temperature": 0,
                       "top_k": 250,
                       "top_p": 0.5,
                       "stop_sequences": []
                       })
    # specifying the model ID of the specific model you would like to use
    # TODO: Change the modelID as you see fit
    modelId = 'anthropic.claude-v2'
    # Specifying the expected formatting of the data
    accept = 'application/json'
    contentType = 'application/json'
    # invoking amazon bedrock with the parameters we determined above^
    response = bedrock.invoke_model(body=body,
                                    modelId=modelId,
                                    accept=accept,
                                    contentType=contentType)
    # gathering the response from amazon Bedrock
    response_body = json.loads(response.get('body').read())
    # creating a variable to store the final answer generated by Amazon Bedrock
    answer = response_body.get('completion')
    # returning the final answer created by Amazon Bedrock
    return answer

# this variable contains the prompt that you can edit to ensure that sample data is created to mimick the task you
# are trying to accomplish
# TODO: EDIT THIS PROMPT TO GENERATE SAMPLE DATA ACCORDING TO YOUR USE CASE:
prompt_data = """\n\nHuman: Provide 10 unique questions related to the creation of an image:

It's very important that the questions are seperated by commas.

Here is an example:
<example>
H: <text>Provide 10 unique questions related to the creation of an image?</text>
A: <response>Here are 200 unique questions related to summarization tasks:
Can you create an image of a building?, Can you create a picture of a horse?, Can you create an image of a unicorn in space?,</response>
</example>
\n\nAssistant:
"""
# gathering the answer returned by the prompt function
answer = prompt(prompt_data)
# printing the answer to understand the structure of the data to help with parsing
print(answer)

# cleaning the data so that it is easier to work with, so we can later store in a CSV
answer = answer.strip("Here are 10 unique questions related to the creation of an image:\n\n")
# splitting the data and storing the generated questions in a list
questions = answer.split(", ")
# printing the final list of questions to ensure they are formatted as expected
print(questions)

# TODO: Uncomment the following code if you would like to store the generated questions as a CSV, to later use in model training.
# with open("image_generation_questions.csv", 'a') as csvfile:
#     writer = csv.writer(csvfile)
#
#     for question in questions:
#         writer.writerow([question, "Image Generation"])