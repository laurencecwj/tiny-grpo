#!/bin/bash

MSG='{ "model": "tgi", "messages": [ { "role": "system", "content": "You are a helpful assistant." }, { "role": "user", "content": "####" } ], "stream": false, "max_tokens": 4096, "temperature": 0.01 }'

msg=$(echo $MSG | sed -e "s/####/$1/")
curl localhost:7777/v1/chat/completions -X POST \
                -d "$msg" \
                -H 'Content-Type: application/json' | jq '.choices[0].message.content'