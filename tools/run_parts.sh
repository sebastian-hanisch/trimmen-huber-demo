#!/bin/sh
# Startet alle Messreihen-Teile (tools/make_results.py --part <name>) mit höchstens 4 Prozessen gleichzeitig und fügt sie danach zusammen.
#     sh tools/run_parts.sh
cd "$(dirname "$0")/.." || exit 1
PY="${THU_PYTHON:-C:/Users/sebas/Claude Code Projects/_venvs/test/Scripts/python.exe}"
"$PY" -I -X utf8 tools/make_results.py --list < /dev/null | tr -d '\r' | tr ' ' '\n' | xargs -P 4 -I{} "$PY" -I -X utf8 tools/make_results.py --part {}
"$PY" -I -X utf8 tools/make_results.py --merge < /dev/null
