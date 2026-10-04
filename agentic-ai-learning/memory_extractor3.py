import json
import os
import urllib.request
import urllib.error


URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-lightning:free"

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY environment variable is not set."
    )


MEMORY_FILE = "user_memory.json"


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

    # Every interest must be a string
    for interest in data["interests"]:
        if not isinstance(interest, str):
            raise ValueError(
                "Every interest must be a string"
            )

    return True


def extract_memory(conversation):

    prompt = f"""
You are a strict user-memory extraction system.

Extract information that the user explicitly states about themselves
and that could reasonably be useful in future conversations.

IMPORTANT RULES:

1. Extract ONLY facts explicitly stated by the user about themselves.
2. NEVER infer or guess information.
3. NEVER extract information about other people.
4. If the user says "I", "my", or "me" and provides information,
   that information refers to the user.
5. Include explicit user interests, hobbies, activities, skills,
   learning topics, preferences, and ongoing areas of interest.
6. An interest does NOT need to be permanent or lifelong.
   If the user explicitly says they are currently learning,
   doing, or interested in something, it can be included.
7. Do not include temporary situations such as being busy this week.
8. Do not include future plans unless they represent a clear
   ongoing interest.
9. Do not convert opinions into interests unless the user explicitly
   describes them as a preference.
10. Preserve the meaning of the user's statement.
11. If a field is unknown, use null.
12. If there are no interests, use an empty list.
13. Return ONLY valid JSON. Do not return explanations.

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

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as e:
        print("Memory extraction HTTP error:", e.code)
        print(e.read().decode("utf-8"))
        raise

    if "choices" not in result:
        print("Unexpected memory extraction response:")
        print(json.dumps(result, indent=2))
        raise RuntimeError(
            "Memory extractor did not return a normal response."
        )

    answer = result["choices"][0]["message"]["content"]

    # Parse JSON returned by LLM
    try:
        memory = json.loads(answer)
    except json.JSONDecodeError:
        print("Raw invalid memory output:")
        print(answer)
        raise ValueError(
            "Memory extractor returned invalid JSON"
        )

    # Validate LLM output
    validate_memory(memory)

    return memory


def load_existing_memory():

    if not os.path.exists(MEMORY_FILE):
        return {
            "name": None,
            "profession": None,
            "workplace": None,
            "interests": []
        }

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_memory(existing, new):

    merged = existing.copy()

    # Update scalar fields only when new value is not null
    for field in ["name", "profession", "workplace"]:
        if new[field] is not None:
            merged[field] = new[field]

    # Add new interests without duplicates
    for interest in new["interests"]:
        if interest not in merged["interests"]:
            merged["interests"].append(interest)

    return merged


def save_memory(memory):

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            memory,
            f,
            indent=4,
            ensure_ascii=False
        )


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    conversation = """
    My name is Arnab and I work as a journalist at Republic Bharat.

    I enjoy trekking and playing table tennis.

    I recently started learning System Design.

    I am also learning to play Sitar online from YouTube.
    """

    print("Extracting memory...")

    new_memory = extract_memory(conversation)

    print("\nNew extracted memory:")
    print(json.dumps(new_memory, indent=2))

    existing_memory = load_existing_memory()

    merged_memory = merge_memory(
        existing_memory,
        new_memory
    )

    print("\nMerged memory:")
    print(json.dumps(merged_memory, indent=2))

    save_memory(merged_memory)

    print("\nMemory saved to user_memory.json")