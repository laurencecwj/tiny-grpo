#!/bin/bash

MSG='{ "model": "tgi", "messages": [ { "role": "system", "content": "You are a helpful assistant." }, { "role": "user", "content": "####" } ], "stream": false, "max_tokens": 4096, "temperature": 0.01 }'

INFILE="$1"

# Read the input file line by line
while read -r LINE
do
    msg=$(echo $MSG | sed -e "s/####/$LINE/")
    curl localhost:7777/v1/chat/completions -X POST \
                    -d "$msg" \
                    -H 'Content-Type: application/json' 2>/dev/null | jq '.choices[0].message.content' >> "$INFILE".output
done < "$INFILE"
