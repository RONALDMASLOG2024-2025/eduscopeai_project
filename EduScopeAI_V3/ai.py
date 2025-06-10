from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message)
