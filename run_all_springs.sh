#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tail -n +2 "$script_dir/springs.txt" | while IFS=$'\t' read -r part_number _rest; do
    key="${part_number,,}"
    echo "Running $key..."
    python "$script_dir/spring.py" "$key" > "$script_dir/outputs/${key}.txt"
done

echo "Done."
