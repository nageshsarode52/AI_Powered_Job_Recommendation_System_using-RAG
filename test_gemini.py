import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

print("GOOGLE_API_KEY present? ->", bool(os.environ.get("GOOGLE_API_KEY")))
print("API key preview:", (os.environ.get("GOOGLE_API_KEY") or "")[:8])

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.0")
    resp = llm.invoke([HumanMessage(content="Say hello in one short sentence.")])
    print("RESPONSE OK:", resp.content)
except Exception as e:
    print("LLM CALL ERROR:", repr(e))