import functions_framework
import logging
import base64
import json
import helpercode
from google import genai
from google.genai import types
import gemini20handler


PROJECT_ID = helpercode.get_project_id()
LOCATION = "us-central1"

@functions_framework.cloud_event
def hello_gemini(cloud_event):
    session_id = cloud_event.data["message"]["attributes"]["session_id"]
    helpercode.init_logging(session_id)
    logger = logging.getLogger("MarketMind-async")
    logger.warning("Processing request")
    model = cloud_event.data["message"]["attributes"]["model"]
    prompt = cloud_event.data["message"]["attributes"]["prompt"]
    aicontent = []
    message_aicontentstring = base64.b64decode(cloud_event.data["message"]["data"])
    message_aicontent = json.loads(message_aicontentstring)
    for item in message_aicontentstring:
        aicontent.append(types.Content(role=item["role"], parts=[types.Part(text=item["content"])]))

    logger.warning(f"Gemini request context ready: {aicontent}")
    
    if model == "gemini-2.0-flash-exp":
        gemini20handler.handle_gemini20(prompt, aicontent, logger, PROJECT_ID, LOCATION)

