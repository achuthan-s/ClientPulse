import json
import os
import random
import requests
from pydantic import BaseModel, Field
from typing import List

# Define the structure we want the LLM to return
class MessageExtraction(BaseModel):
    sentiment: float = Field(
        description="Sentiment score from -1.0 (very negative) to 1.0 (very positive)"
    )
    topics: List[str] = Field(
        description="List of 1-3 main topics discussed (e.g. 'billing', 'outage', 'kickoff')"
    )
    scope_dispute_signal: bool = Field(
        description="True if the client is complaining about missing features, delays, or what was promised vs delivered"
    )
    reason: str = Field(
        description="Short 1-sentence reason for the scope_dispute_signal value"
    )

def mock_llm_extraction(text: str) -> MessageExtraction:
    """
    Since we might not have an API key in this environment, this function
    mocks the LLM's structured output based on simple keyword matching.
    In a real app, this would be: `client.models.generate_content(..., response_schema=MessageExtraction)`
    """
    text_lower = text.lower()
    
    # 1. Estimate sentiment
    sentiment = 0.0
    if any(word in text_lower for word in ["great", "fantastic", "thanks", "good", "excited", "loved", "smooth", "smoothly"]):
        sentiment = min(1.0, sentiment + 0.6)
    if any(word in text_lower for word in ["delay", "missing", "unacceptable", "frustrated", "error", "down", "disaster", "cancel", "outage", "disruption", "critical", "vulnerability"]):
        sentiment = max(-1.0, sentiment - 0.7)
    if "threatened" in text_lower or "breach" in text_lower or "pausing all payments" in text_lower or "migrating to a competitor" in text_lower:
        sentiment = -1.0
        
    # 2. Extract topics
    topics = []
    if "password" in text_lower or "login" in text_lower: topics.append("authentication")
    if "report" in text_lower or "dashboard" in text_lower: topics.append("reporting")
    if "delay" in text_lower or "schedule" in text_lower or "timeline" in text_lower: topics.append("timeline")
    if "scope" in text_lower or "endpoint" in text_lower: topics.append("deliverables")
    if "outage" in text_lower or "500" in text_lower or "down" in text_lower or "locked up" in text_lower: topics.append("system_stability")
    if "security" in text_lower or "vulnerability" in text_lower or "infosec" in text_lower: topics.append("security")
    if not topics: topics.append("general_update")
        
    # 3. Detect scope disputes
    scope_dispute = False
    reason = "No scope issues detected in this message."
    
    if "out of scope" in text_lower or ("explicitly discussed" in text_lower and "missing" in text_lower) or "agreed upon scope" in text_lower:
        scope_dispute = True
        reason = "Client explicitly mentioned missing deliverables or disputed the agreed scope."
    elif "delay" in text_lower and "unacceptable" in text_lower:
        scope_dispute = True
        reason = "Client is highly frustrated about delivery timelines."
        
    return MessageExtraction(
        sentiment=sentiment,
        topics=topics[:3],
        scope_dispute_signal=scope_dispute,
        reason=reason
    )

def extract_features_from_llm(text: str) -> MessageExtraction:
    """
    This is the wrapper that would call the actual LLM.
    NVIDIA is preferred when configured; otherwise use Gemini or the local mock.
    """
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    api_key = os.environ.get("GEMINI_API_KEY")

    if nvidia_api_key:
        try:
            model = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
            response = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {nvidia_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Extract the requested client-message features. "
                                "Return only a JSON object with sentiment (number from -1 to 1), "
                                "topics (array of 1-3 strings), scope_dispute_signal (boolean), "
                                "and reason (string)."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Analyze this client message:\n\n{text}",
                        },
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return MessageExtraction.model_validate_json(content)
        except Exception as e:
            print(f"NVIDIA LLM call failed: {e}. Falling back to mock.")
            return mock_llm_extraction(text)

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"Analyze this client message and extract the requested fields:\n\nMessage: {text}",
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': MessageExtraction,
                    'temperature': 0.1,
                },
            )
            # Parse the JSON string back into our Pydantic model
            return MessageExtraction.model_validate_json(response.text)
        except Exception as e:
            print(f"LLM call failed: {e}. Falling back to mock.")
            return mock_llm_extraction(text)
    else:
        return mock_llm_extraction(text)

def main():
    input_path = "data/client_messages.json"
    output_path = "data/extracted_messages.json"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run Phase 1 first.")
        return
        
    with open(input_path, "r") as f:
        messages = json.load(f)
        
    print(f"Extracting features for {len(messages)} messages...")
    
    # We will enrich the original message dictionary
    for i, msg in enumerate(messages):
        extraction = extract_features_from_llm(msg["text"])
        
        # Merge the extracted Pydantic fields into the message dict
        msg.update(extraction.model_dump())
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(messages)} messages.")
            
    with open(output_path, "w") as f:
        json.dump(messages, f, indent=2)
        
    print(f"Finished. Saved enriched data to {output_path}")
    
    # Print a sample to verify
    print("\nSample Extraction (Declining Account):")
    sample = next(m for m in messages if m["scope_dispute_signal"] is True)
    print(json.dumps(sample, indent=2))

if __name__ == "__main__":
    main()
