import time

# import streamlit as st
# from streamlit_float import *



from google import genai
from google.genai import types


from tenacity import retry, wait_random_exponential

import helpercode
import logging
import helperbqfunction
import helperfinhub
import helperalphavantage

import gemini20functionfinhub
import gemini20functiongeneral
import gemini20functionalphavantage


import evaluationagent

PROJECT_ID = helpercode.get_project_id()
LOCATION = "us-central1"

MODELNAME = "gemini-2.0-flash-exp"

# helpercode.init_logging()
logger = logging.getLogger("MarketMind-async")

SYSTEM_INSTRUCTION = """You are a highly skilled financial analyst specializing in asset management. Your task is to conduct thorough financial analysis and generate detailed reports from an investor's perspective. Follow these guidelines meticulously:

                        **1. Symbol Identification and Lookup:**

                        *   **Primary Symbol Focus:** When multiple symbols exist for a company, prioritize the *primary* symbol, which typically does *not* contain a dot (".") in its name (e.g., "AAPL" instead of "AAPL.MX").
                        *   **Mandatory Symbol Lookup:** Before executing any other functions, always use the `symbol_lookup` function to identify and confirm the correct primary symbol for the company under analysis. Do not proceed without a successful lookup.
                        *   **Handle Lookup Failures:** If `symbol_lookup` fails to identify a symbol, inform the user and gracefully end the analysis.

                        **2. Date Handling:**

                        *   **Current Date Determination:** Use the `current_date` function to obtain the current date at the beginning of each analysis. This date is critical for subsequent time-sensitive operations.
                        *   **Default Year Range:** If a function call requires a date range and the user has not supplied one, calculate the start and end dates for the *current year* using the date obtained from `current_date`. Use these as the default start and end dates in the relevant function calls.

                        **3. Analysis Components:**

                        *   **Comprehensive Report:** Your report should be comprehensive and contain the following sections:
                            *   **Company Profile:**  Include a detailed overview of the company, its industry, and its business model.
                            *   **Company News:** Summarize the latest significant news impacting the company.
                            *   **Basic Financials:** Present key financial metrics and ratios for the company, covering recent periods (using current year as default period).
                            *   **Peer Analysis:** Identify and analyze the company's key competitors, comparing their financials and market performance (current year default).
                            *   **Insider Sentiment:**  Report on insider trading activity and overall sentiment expressed by company insiders.
                            *   **SEC Filings:**  Provide an overview of the company's recent SEC filings, highlighting any significant disclosures (current year as default).

                        **4. Data Handling and Error Management:**

                        *   **Data Completeness:** If a function requires date that is not present or unavailable, use the current year as the default period. Report missing data but don't let it stop you.
                        *   **Function Execution:** Execute functions carefully, ensuring you have the necessary data, especially dates and symbols, before invoking any function.
                        *   **Clear Output:** Present results in a clear and concise manner, suitable for an asset management investor.

                        **5. Analytical Perspective:**

                        *   **Asset Management Lens:** Conduct all analysis with an asset manager's perspective in mind. Evaluate the company as a potential investment, focusing on risk, return, and long-term prospects.

                        **Example Workflow (Implicit):**

                        1.  Get the current date using `current_date`.
                        2.  Use `symbol_lookup` to identify the primary symbol for the company provided by the user.
                        3.  If no symbol is found, end the process and report back.
                        4.  Calculate the start and end date by using the result of the current_date tool.
                        5.  Call the relevant functions to retrieve the company profile, news, financials, peers, insider sentiment, and SEC filings. Use the current year start and end date when required, or the date specified by the user.
                        6.  Assemble a detailed and insightful report that addresses each of the sections mentioned above.
                        """


market_query20_tool = types.Tool(
    function_declarations=[
        # geminifunctionsbq.sql_query_func,
        # geminifunctionsbq.list_datasets_func,
        # geminifunctionsbq.list_tables_func,
        # geminifunctionsbq.get_table_func,
        # geminifunctionsbq.sql_query_func,
        gemini20functionfinhub.symbol_lookup,
        gemini20functionfinhub.company_news,
        gemini20functionfinhub.company_profile,
        gemini20functionfinhub.company_basic_financials,
        gemini20functionfinhub.company_peers,
        gemini20functionfinhub.insider_sentiment,
        gemini20functionfinhub.financials_reported,
        gemini20functionfinhub.sec_filings,
        gemini20functiongeneral.current_date,
        # gemini20functionalphavantage.monthly_stock_price,
        # gemini20functionalphavantage.market_sentiment,
    ],
)

generate_config_20 = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    system_instruction=[types.Part.from_text(SYSTEM_INSTRUCTION)],
    tools= [market_query20_tool],
)


def handle_api_response(api_requests_and_responses, backend_details):
    backend_details += "- Function call:\n"
    backend_details += (
                        "   - Function name: ```"
                        + str(api_requests_and_responses[-1][0])
                        + "```"
                    )
    backend_details += "\n\n"
    backend_details += (
                        "   - Function parameters: ```"
                        + str(api_requests_and_responses[-1][1])
                        + "```"
                    )
    backend_details += "\n\n"
    backend_details += (
                        "   - API response: ```"
                        + str(api_requests_and_responses[-1][2])
                        + "```"
                    )
    backend_details += "\n\n"
    # with message_placeholder.container():
    #     st.markdown(backend_details)
    return backend_details


def handle_external_function(api_requests_and_responses, params, function_name):
    """This function handesl the call to the external function once Gemini has determined a function call is required"""
    if function_name in helpercode.function_handler.keys():
        logger.warning("General function found")
        api_response = helpercode.function_handler[function_name]()
        api_requests_and_responses.append(
                                [function_name, params, api_response]
                        )

    # if function_name in helperbqfunction.function_handler.keys():
    #     logger.warning("BQ function found")
    #     api_response = helperbqfunction.function_handler[function_name](st.session_state.client, params)
    #     api_requests_and_responses.append(
    #                             [function_name, params, api_response]
    #                     )

    if function_name in helperfinhub.function_handler.keys():
        logger.warning("finhub function found")
        api_response = helperfinhub.function_handler[function_name](params)
        api_requests_and_responses.append(
                                [function_name, params, api_response]
                        )
    
    if function_name in helperalphavantage.function_handler.keys():
        logger.warning("alpha vantage function found")
        api_response = helperalphavantage.function_handler[function_name](params)
        api_requests_and_responses.append(
                                [function_name, params, api_response]
                        )
                
    return api_response

def handel_gemini20_parallel_func(response, 
                                  api_requests_and_responses, backend_details, 
                                  functioncontent, handle_external_function, 
                                  generate_config_20, aicontent, messages, client, logger):
    logger.warning("Starting parallal function resonse loop")
    global stringoutputcount
    parts=[]
    function_parts = []
    for response in response.candidates[0].content.parts:
        logger.warning("Function loop starting")
        logger.warning(response)
        params = {}
        try:
            for key, value in response.function_call.args.items():
                params[key] = value
        except AttributeError:
            continue
                
        logger.warning("Prams processing done")
        logger.warning(response)
        logger.warning(f"""FunctionName: {response.function_call.name} Params: {params}""")
        # logger.warning(params)

        function_name = response.function_call.name

        api_response = handle_external_function(api_requests_and_responses, params, function_name)

        logger.warning("Function Response complete")
        stringoutputcount = stringoutputcount + len(str(api_response))
        logger.warning(f"""String output count is {stringoutputcount}""")
        logger.warning(api_response)
        function_parts.append(response)
        parts.append(types.Part.from_function_response(
                    name=function_name,
                    response={
                        "result": api_response,
                    },
                    ),
                )

        backend_details = handle_api_response(api_requests_and_responses, backend_details)

    logger.warning("Making gemini call for api response")

    functioncontent.append(function_parts)
    functioncontent.append(parts)
    response, messages = handle_gemini20_chat(functioncontent, logger, generate_config_20, client, messages)

    logger.warning(f"""Length of functions is {len(response.candidates[0].content.parts)}""")
    #testing
    # st.session_state.
    aicontent.append(response.candidates[0].content)
    #testing

    if len(response.candidates[0].content.parts) >1:
        response, backend_details, functioncontent, messages = handel_gemini20_parallel_func(response,
                                                                        api_requests_and_responses,
                                                                        backend_details, functioncontent,
                                                                        handle_external_function, generate_config_20, 
                                                                        aicontent, messages, client, logger)

            
    logger.warning("gemini api response completed")
    return response,backend_details, functioncontent, messages

def handle_gemini20_serial_func(response, 
                                api_requests_and_responses, 
                                backend_details, functioncontent, handle_external_function, 
                                generate_config_20, aicontent, messages, client, logger):
    response = response.candidates[0].content.parts[0]
    global stringoutputcount
    logger.warning(response)
    logger.warning("First Resonse done")

    function_calling_in_process = True
    while function_calling_in_process:
        try:
            logger.warning("Function loop starting")
            params = {}
            for key, value in response.function_call.args.items():
                params[key] = value
                    
            logger.warning("Prams processing done")
            logger.warning(response)
            logger.warning(f"""FunctionName: {response.function_call.name} Params: {params}""")
            # logger.warning(params)

            function_name = response.function_call.name

            api_response = handle_external_function(api_requests_and_responses, params, function_name)

            logger.warning("Function Response complete")
            stringoutputcount = stringoutputcount + len(str(api_response))
            logger.warning(f"""String output count is {stringoutputcount}""")
            logger.warning(api_response)
            logger.warning("Making gemini call for api response")
            
            part = types.Part.from_function_response(
                            name=function_name,
                            response={
                                "result": api_response,
                            },
            )

            functioncontent.append(response)
            functioncontent.append(part)
            response, messages = handle_gemini20_chat_single(functioncontent, logger, generate_config_20, client, messages)



            logger.warning("Function Response complete")


            backend_details = handle_api_response(api_requests_and_responses, backend_details)
                    
            logger.warning("gemini api response completed")
            logger.warning(response)
            logger.warning("next call ready")
            logger.warning(f"""Length of functions is {len(response.candidates[0].content.parts)}""")
            #testing
            # st.session_state.
            aicontent.append(response.candidates[0].content)
            #testing

            if len(response.candidates[0].content.parts) >1:
                response, backend_details, functioncontent, messages = handel_gemini20_parallel_func(response,
                                                                        api_requests_and_responses,
                                                                        backend_details, functioncontent, handle_external_function, 
                                                                        generate_config_20, aicontent, 
                                                                        messages, client, logger)
            else:
                response = response.candidates[0].content.parts[0]


        except AttributeError as e:
            logger.warning(e)
            function_calling_in_process = False
    return response,backend_details, functioncontent, messages


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def handle_gemini20_chat(functioncontent, logger, generate_config_20, client, messages):
    logger.warning("Making actual multi gemini call")
    # st.session_state.aicontent.append(function_parts)
    # st.session_state.aicontent.append(parts)
    # functioncontent.append(function_parts)
    # functioncontent.append(parts)
    try:
        response = client.models.generate_content(model=MODELNAME,
                                                            #   contents=st.session_state.aicontent,
                                                              contents=functioncontent,
                                                              config=generate_config_20)
    except Exception as e:
        logger.error(e)
        raise e
    logger.warning("Multi call succeeded")
    logger.warning(response)
    logger.warning(f"""Tokens in use: {response.usage_metadata}""")
    
    try:
        logger.warning("Adding messages to session state")
        full_response = response.text
        # st.session_state.
        messages.append({
            "role": "assistant",
            "content": full_response,
            "md5has" : helpercode.get_md5_hash(full_response)
        })
    except Exception as e:
        logger.error(e)
    logger.warning("sending response back")
    return response, messages

@retry(wait=wait_random_exponential(multiplier=1, max=60))
def handle_gemini20_chat_single(functioncontent, logger, generate_config_20, client, messages):
    logger.warning("Making actual single gemini call")
    # st.session_state.aicontent.append(response)
    # st.session_state.aicontent.append(part)
    # functioncontent.append(response)
    # functioncontent.append(part)
    try:
        response = client.models.generate_content(model=MODELNAME,
                                                            #   contents=st.session_state.aicontent,
                                                              contents=functioncontent,
                                                              config=generate_config_20)
    except Exception as e:
        logger.error(e)
        raise e
    logger.warning("Single call succeeded")
    logger.warning(response)
    logger.warning(f"""Tokens in use: {response.usage_metadata}""")
    
    try:
        logger.warning("Adding messages to session state")
        full_response = response.text
        # st.session_state.
        messages.append({
            "role": "assistant",
            "content": full_response,
            "md5has" : helpercode.get_md5_hash(full_response)
        })
    except Exception as e:
        logger.error(e)
    logger.warning("sending response back")
    return response, messages

@retry(wait=wait_random_exponential(multiplier=1, max=60))
def handel_initial_gemini20_chat(generate_config_20, aicontent, client, logger):
    try:
        response =client.models.generate_content(model=MODELNAME,
                                                            contents=aicontent,
                                                            config=generate_config_20)
    except Exception as e:
        logger.error(e)
        raise e                                                    
    return response

def handle_gemini20(prompt, aicontent, logger, PROJECT_ID, LOCATION):
    logger.warning("Starting Gemini 2.0")
    global stringoutputcount

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )

    # if "chat" not in st.session_state:
    #     st.session_state.chat = client
    
    # if "aicontent" not in st.session_state:
    #     st.session_state.aicontent = []
    
    stringoutputcount = 0

    # if prompt := st.chat_input("What is up?"):

    #     # # Display user message in chat message container
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #st.session_state.messages is being translated into below:
    messages = []

    # st.session_state.aicontent is being translated into below:
    # aicontent = []


    # Add user message to chat history
    logger.warning(f"""Model is: {MODELNAME}, Prompt is: {prompt}""")


    # st.session_state.
    # messages.append({"role": "user", "content": prompt})

    # with st.chat_message("assistant"):
    # message_placeholder = st.empty()
    full_response = ""


    logger.warning("Configuring prompt")
    # st.session_state.
    aicontent.append(types.Content(role='user', parts=[types.Part(text=prompt)]))
    functioncontent = []
    functioncontent.append(types.Content(role='user', parts=[types.Part(text=prompt)]))

    # evaluationagent.evaluation_agent(prompt, MODELNAME)

    logger.warning("Conversation history start")
    logger.warning(aicontent)
    logger.warning("Conversation history end")
    logger.warning("Prompt configured, calling Gemini...")
    response = handel_initial_gemini20_chat(generate_config_20, aicontent, client, logger)

    logger.warning("Gemini called, This is the start")
    logger.warning(response)
    logger.warning(f"""Tokens in use: {response.usage_metadata}""")
    logger.warning("The start is done")

    logger.warning(f"""Length of functions is {len(response.candidates[0].content.parts)}""")

    api_requests_and_responses = []
    backend_details = ""
    #testing
    # st.session_state.
    aicontent.append(response.candidates[0].content)
    #testing
    if len(response.candidates[0].content.parts) >1:
        response, backend_details, functioncontent = handel_gemini20_parallel_func(response, 
                                                                                    api_requests_and_responses, 
                                                                                    backend_details, functioncontent, 
                                                                                    handle_external_function, generate_config_20,
                                                                                    client, logger)


    else:
        response, backend_details, functioncontent = handle_gemini20_serial_func(response, 
                                                                                    api_requests_and_responses, 
                                                                                    backend_details, functioncontent, 
                                                                                    handle_external_function, generate_config_20, 
                                                                                    client, logger)

    time.sleep(3)
    
    logger.warning(f"Full response generated: {full_response}")
    full_response = response.text
    # st.session_state.aicontent.append(types.Content(role='model', parts=[types.Part(text=full_response)]))
    # with message_placeholder.container():
    #     st.markdown(full_response.replace("$", r"\$"))  # noqa: W605
    #     with st.expander("Function calls, parameters, and responses:"):
    #         st.markdown(backend_details)

    # st.session_state.
    # messages.append(
    #     {
    #         "role": "assistant",
    #         "content": full_response,
    #         "backend_details": backend_details,
    #         "md5has" : helpercode.get_md5_hash(full_response)
    #     }
    # )
    logger.warning(f"""Total string output count is {stringoutputcount}""")
    logger.warning(aicontent)
    logger.warning("This is the end of Gemini 2.0")
    return full_response
