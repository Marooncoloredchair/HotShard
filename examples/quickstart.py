"""Smallest possible example: wrap an OpenAI client.

Prereqs:
    pip install 'guarded-hotshard[openai]'

Run:
    OPENAI_API_KEY=sk-... python examples/quickstart.py
"""

from openai import OpenAI

from guarded_hotshard import wrap


client = OpenAI()  # reads OPENAI_API_KEY
client = wrap(
    client,
    mode="protected_lane",
    critical_users={"acme-prod-1"},
    concurrency=8,
)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hi in one word."}],
    user="acme-prod-1",  # premium tenant -> gets the protected lane
)
print(resp.choices[0].message.content)
