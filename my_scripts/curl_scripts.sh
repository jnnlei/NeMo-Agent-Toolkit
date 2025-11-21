# 方式 1: 使用 input_message + 额外字段

# 基本用法 - 只有 input_message
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "Is 4 + 4 greater than the current hour of the day?"
  }'

# 带额外字段 
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "Is 4 + 4 greater than the current hour of the day?",
    "use_knowledge_base": true,
    "temperature": 0.7,
    "max_tokens": 512
  }'

# 方式 2: 使用 messages (OpenAI 兼容格式)

# 标准 messages 格式
curl --request POST \
  --url http://localhost:8000/chat \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [
      {
        "role": "user",
        "content": "Is 4 + 4 greater than the current hour of the day?"
      }
    ]
  }'

# messages 格式 + 额外字段
curl --request POST \
  --url http://localhost:8000/chat \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [
      {
        "role": "user",
        "content": "Is 4 + 4 greater than the current hour of the day?"
      }
    ],
    "use_knowledge_base": true,
    "temperature": 0.7,
    "model": "meta/llama-3.1-70b-instruct"
  }'

# 方式 3: 流式响应

# 使用 generate_stream 端点
curl --request POST \
  --url http://localhost:8000/generate_stream \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "Tell me a short story",
    "stream": true
  }'

# 使用 chat_stream 端点
curl --request POST \
  --url http://localhost:8000/chat_stream \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [{"role": "user", "content": "Tell me a short story"}],
    "stream": true,
    "temperature": 0.8
  }'