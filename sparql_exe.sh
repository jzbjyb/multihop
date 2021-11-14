#!/usr/bin/env bash
set -e

world_size=20

for (( rank = 0; rank < ${world_size}; ++rank ))
do
  python -m dataset.sparql_log \
    --output wikidata_sparql/execution/organic.jsonl.${world_size}.${rank} \
    --world_size ${world_size} \
    --rank ${rank} &> wikidata_sparql/execution/organic.out.${world_size}.${rank} &
done
wait
