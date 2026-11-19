#!/bin/sh
# Start Ollama and make sure the chat and embedding models are present.
# Models are stored in the ollama volume, so they are downloaded only on the
# first start; later starts skip the pull and work offline.
set -e

ollama serve &
server_pid=$!

# Forward docker stop (SIGTERM) to the server so it shuts down cleanly.
trap 'kill -TERM "$server_pid"; wait "$server_pid"; exit 0' TERM INT

until ollama list >/dev/null 2>&1; do
    sleep 1
done

for model in "$LLM_MODEL" "$EMBEDDING_MODEL"; do
    if ollama show "$model" >/dev/null 2>&1; then
        echo "Model $model already present."
    else
        echo "Pulling $model..."
        ollama pull "$model"
    fi
done

echo "Models ready."
wait "$server_pid"
