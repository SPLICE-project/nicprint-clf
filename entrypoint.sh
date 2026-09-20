#!/usr/bin/env bash
if ! whoami &>/dev/null; then
    echo "evaluator:x:$(id -u):0:evaluator:/home/evaluator:/bin/bash" >> /etc/passwd
    export HOME=/home/evaluator
fi
exec "$@"