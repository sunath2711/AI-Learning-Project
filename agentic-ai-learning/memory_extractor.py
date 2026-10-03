import json
import os
import urllib.request


URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-lightning:free"

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")


# conversation = """
# My name is Arnab.
# I work as a journalist at Republic Bharat.
# I enjoy trekking and playing table tennis.
# I recently started learning Agentic AI.
# I am huge fan of Classical music and I love to play the flute.
# My idol is cristiano ronaldo and I am a huge fan of football.
# """
conversation = """
My name is Arnab and I work as a journalist at Republic Bharat.
I enjoy trekking and playing table tennis.

My friend Rahul is a software engineer at Google and loves cricket.

I've been extremely busy this week because of a project at work.

I think Python is much better than Java for my work.

My colleague Priya is learning Agentic AI.

I recently started learning Agentic AI myself.

I am planning to buy a new laptop next month.

I went to Nepal for trekking last year.
"""

# prompt = f"""
# You are a memory extraction system.

# Read the conversation below and extract only stable,
# useful information about the user that could be useful
# in future conversations.

# Return ONLY valid JSON.
# Do not include explanations.

# Use this format:

# {{
#     "name": "...",
#     "profession": "...",
#     "workplace": "...",
#     "interests": []
# }}

# If a field is unknown, use null.
# If there are no interests, use an empty list.

# Conversation:
# {conversation}
# """
prompt = f"""
You are a strict user-memory extraction system.

Your job is to extract ONLY information that the user explicitly
states about themselves.

Rules:
1. Never infer information.
2. Never guess missing information.
3. Never add facts that are not explicitly stated.
4. Do not extract facts about other people.
5. Do not treat temporary situations as permanent user facts.
6. Do not convert opinions into factual attributes unless they are
   clearly useful as a stable preference.
7. Preserve the meaning of what the user said.
8. If a field is unknown, use null.
9. If there are no interests, use an empty list.
10. Return ONLY valid JSON. No explanation.

Use exactly this format:

{{
    "name": null,
    "profession": null,
    "workplace": null,
    "interests": []
}}

Conversation:
{conversation}
"""

messages = [
    {
        "role": "system",
        "content": "You extract structured user memory accurately."
    },
    {
        "role": "user",
        "content": prompt
    }
]


payload = {
    "model": MODEL,
    "messages": messages,
    "stream": False
}


data = json.dumps(payload).encode("utf-8")


request = urllib.request.Request(
    URL,
    data=data,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    },
    method="POST"
)


with urllib.request.urlopen(request) as response:
    result = json.loads(response.read().decode("utf-8"))


answer = result["choices"][0]["message"]["content"]
def validate_memory(data):
    required_fields = {
        "name": (str, type(None)),
        "profession": (str, type(None)),
        "workplace": (str, type(None)),
        "interests": (list,)
    }

    # Check for unexpected fields
    unexpected_fields = set(data.keys()) - set(required_fields.keys())

    if unexpected_fields:
        raise ValueError(
            f"Unexpected fields found: {unexpected_fields}"
        )

    # Check required fields and their types
    for field, allowed_types in required_fields.items():

        if field not in data:
            raise ValueError(
                f"Missing required field: {field}"
            )

        if not isinstance(data[field], allowed_types):
            raise ValueError(
                f"Invalid type for {field}: "
                f"{type(data[field]).__name__}"
            )

    # Check that every interest is a string
    for interest in data["interests"]:
        if not isinstance(interest, str):
            raise ValueError(
                "Every interest must be a string"
            )

    return True
try:
    memory = json.loads(answer)
except json.JSONDecodeError:
    raise ValueError("LLM returned invalid JSON")

validate_memory(memory)

with open("user_memory.json", "w", encoding="utf-8") as f:
    json.dump(memory, f, indent=4, ensure_ascii=False)

print("\nMemory saved to user_memory.json")

print("Raw model output:")
print(answer)
print("\nValidated memory:")
print(json.dumps(memory, indent=2))

