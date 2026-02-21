#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
#  Aurora Ops — Engine Diagnostic
#  Run from project root: bash diagnose.sh
# ═══════════════════════════════════════════════════════

set -o pipefail

RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[0;33m'
CYN='\033[0;36m'
RST='\033[0m'

ok()   { echo -e "  ${GRN}✓${RST} $*"; }
fail() { echo -e "  ${RED}✖${RST} $*"; }
warn() { echo -e "  ${YEL}⚠${RST} $*"; }
info() { echo -e "  ${CYN}→${RST} $*"; }

echo ""
echo "═══════════════════════════════════════════"
echo "  Aurora Ops — Engine Diagnostic"
echo "═══════════════════════════════════════════"
echo ""

# ─── 1. Find Python ───
echo "1) Python Detection"

VENV_PY=""
if [ -f "myenv/bin/python" ]; then
  VENV_PY="myenv/bin/python"
elif [ -f ".venv/bin/python" ]; then
  VENV_PY=".venv/bin/python"
fi

if [ -n "$VENV_PY" ]; then
  ok "Venv found: $VENV_PY"
  PY="$VENV_PY"
  PY_VER=$("$PY" --version 2>&1)
  info "$PY_VER"
elif command -v python3 &>/dev/null; then
  PY="python3"
  warn "No venv — using system python3"
  PY_VER=$("$PY" --version 2>&1)
  info "$PY_VER"
elif command -v python &>/dev/null; then
  PY="python"
  warn "No venv — using system python"
  PY_VER=$("$PY" --version 2>&1)
  info "$PY_VER"
else
  fail "Python not found!"
  exit 1
fi

echo ""

# ─── 2. Check engine.py exists ───
echo "2) Engine File Check"

if [ -f "engine.py" ]; then
  ok "engine.py found"
  LINES=$(wc -l < engine.py)
  info "$LINES lines"
else
  fail "engine.py NOT FOUND in $(pwd)"
  echo "     This is the problem. Make sure engine.py is in your project root."
  exit 1
fi

# Check for ghost files
for ghost in enhance.py adjust.py analyzer.py; do
  if [ -f "$ghost" ]; then
    warn "OLD file still exists: $ghost (should be deleted)"
  fi
done

echo ""

# ─── 3. Test imports ───
echo "3) Python Import Test"

IMPORT_OUT=$("$PY" -c "
import sys, json
try:
    import cv2
    print(json.dumps({'cv2': cv2.__version__}))
except ImportError as e:
    print(json.dumps({'error': f'cv2 import failed: {e}'}))
try:
    import numpy as np
    print(json.dumps({'numpy': np.__version__}))
except ImportError as e:
    print(json.dumps({'error': f'numpy import failed: {e}'}))
" 2>&1)

if echo "$IMPORT_OUT" | grep -q '"error"'; then
  fail "Import errors:"
  echo "$IMPORT_OUT"
else
  ok "cv2 + numpy OK"
  info "$IMPORT_OUT"
fi

echo ""

# ─── 4. Test engine.py startup (capabilities) ───
echo "4) Engine Startup (capabilities command)"

CAP_OUT=$(echo '{"command":"capabilities"}' | "$PY" engine.py 2>&1)
CAP_EXIT=$?

if [ $CAP_EXIT -ne 0 ]; then
  fail "engine.py crashed on startup (exit code $CAP_EXIT)"
  echo "     Raw output:"
  echo "$CAP_OUT"
  echo ""
  echo "     This usually means an import error in engine.py itself."
  echo "     Try: $PY engine.py  (with no stdin) to see the traceback"
  exit 1
fi

if echo "$CAP_OUT" | "$PY" -m json.tool &>/dev/null; then
  ok "Capabilities returned valid JSON"
  ENGINE_VER=$(echo "$CAP_OUT" | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(d.get('engine_version','?'))" 2>/dev/null)
  info "Engine version: $ENGINE_VER"
else
  fail "Capabilities returned invalid JSON:"
  echo "$CAP_OUT"
  exit 1
fi

echo ""

# ─── 5. Create a test image ───
echo "5) Creating Test Image"

TEST_IMG="/tmp/aurora_diag_test.jpg"
"$PY" -c "
import cv2, numpy as np
img = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)
cv2.imwrite('$TEST_IMG', img)
print('ok')
" 2>&1

if [ -f "$TEST_IMG" ]; then
  ok "Test image created: $TEST_IMG"
else
  fail "Could not create test image"
  exit 1
fi

echo ""

# ─── 6. Test each command ───
echo "6) Testing Engine Commands"

test_cmd() {
  local label="$1"
  local json_input="$2"
  local expect_temp="$3"  # "yes" if we expect temp_path

  RESULT=$(echo "$json_input" | "$PY" engine.py 2>/tmp/aurora_diag_stderr.txt)
  EXIT=$?
  STDERR=$(cat /tmp/aurora_diag_stderr.txt 2>/dev/null)

  if [ $EXIT -ne 0 ]; then
    fail "$label — crashed (exit $EXIT)"
    [ -n "$STDERR" ] && echo "     stderr: ${STDERR:0:200}"
    return 1
  fi

  if echo "$RESULT" | grep -q '"error"'; then
    ERR=$(echo "$RESULT" | "$PY" -c "import sys,json; print(json.load(sys.stdin).get('error',''))" 2>/dev/null)
    fail "$label — error: $ERR"
    return 1
  fi

  if [ "$expect_temp" = "yes" ]; then
    TEMP=$(echo "$RESULT" | "$PY" -c "import sys,json; print(json.load(sys.stdin).get('temp_path',''))" 2>/dev/null)
    if [ -n "$TEMP" ] && [ -f "$TEMP" ]; then
      ok "$label → $TEMP"
    else
      fail "$label — no temp file produced"
      echo "     output: ${RESULT:0:200}"
      return 1
    fi
  else
    ok "$label"
  fi
  return 0
}

# Analyze
test_cmd "analyze" \
  "{\"command\":\"analyze\",\"image\":\"$TEST_IMG\"}" "no"

# Adjust (basic)
test_cmd "adjust (exposure+contrast)" \
  "{\"command\":\"adjust\",\"image\":\"$TEST_IMG\",\"params\":{\"exposure\":1.0,\"contrast\":30}}" "yes"

# Adjust (all sliders)
test_cmd "adjust (all 15 sliders)" \
  "{\"command\":\"adjust\",\"image\":\"$TEST_IMG\",\"params\":{\"exposure\":0.5,\"contrast\":20,\"highlights\":-15,\"shadows\":30,\"whites\":10,\"blacks\":-10,\"temperature\":25,\"tint\":-10,\"vibrance\":40,\"saturation\":15,\"clarity\":50,\"dehaze\":20,\"sharpness\":40,\"grain\":10,\"vignette\":-30}}" "yes"

# Adjust with X-ray
for xr in structure depth frequency thermal bones reveal bright occlusion; do
  test_cmd "xray: $xr" \
    "{\"command\":\"adjust\",\"image\":\"$TEST_IMG\",\"params\":{\"xray\":\"$xr\",\"xray_blend\":100}}" "yes"
done

# Adjust with X-ray blend
test_cmd "xray blend 50%" \
  "{\"command\":\"adjust\",\"image\":\"$TEST_IMG\",\"params\":{\"exposure\":0.5,\"xray\":\"thermal\",\"xray_blend\":50}}" "yes"

# Enhance modes
for mode in pro color smooth light portrait; do
  test_cmd "enhance: $mode" \
    "{\"command\":\"enhance\",\"image\":\"$TEST_IMG\",\"modes\":[\"$mode\"]}" "yes"
done

# Enhance layer stack
test_cmd "enhance: stacked (smooth+pro+color)" \
  "{\"command\":\"enhance\",\"image\":\"$TEST_IMG\",\"modes\":[\"smooth\",\"pro\",\"color\"]}" "yes"

# Auto-enhance
test_cmd "auto_enhance" \
  "{\"command\":\"auto_enhance\",\"image\":\"$TEST_IMG\"}" "yes"

# Save
LAST_TEMP=$(echo "{\"command\":\"adjust\",\"image\":\"$TEST_IMG\",\"params\":{\"exposure\":1}}" | "$PY" engine.py 2>/dev/null | "$PY" -c "import sys,json; print(json.load(sys.stdin).get('temp_path',''))" 2>/dev/null)
if [ -n "$LAST_TEMP" ] && [ -f "$LAST_TEMP" ]; then
  test_cmd "save" \
    "{\"command\":\"save\",\"temp_path\":\"$LAST_TEMP\",\"save_path\":\"/tmp/aurora_diag_saved.jpg\"}" "no"
fi

echo ""

# ─── 7. Test from Electron's perspective ───
echo "7) Simulating Electron's callEngine()"
info "This tests the exact spawn+stdin pattern your main process uses"

NODE_TEST=$(node -e "
const { spawn } = require('child_process');
const child = spawn('$PY', ['engine.py']);
let stdout = '', stderr = '';
child.stdout.on('data', d => stdout += d);
child.stderr.on('data', d => stderr += d);
child.on('close', code => {
  try {
    const s = stdout.indexOf('{'), e = stdout.lastIndexOf('}');
    if (s === -1) throw new Error('No JSON in stdout');
    const result = JSON.parse(stdout.substring(s, e+1));
    if (result.error) console.log('FAIL: ' + result.error);
    else if (result.status) console.log('OK: ' + JSON.stringify(result).substring(0,120));
    else console.log('UNEXPECTED: ' + stdout.substring(0,120));
  } catch(e) {
    console.log('PARSE_FAIL: ' + e.message);
    console.log('stdout: ' + stdout.substring(0,200));
    console.log('stderr: ' + stderr.substring(0,200));
  }
});
child.stdin.write(JSON.stringify({command:'analyze',image:'$TEST_IMG'}));
child.stdin.end();
" 2>&1)

if echo "$NODE_TEST" | grep -q "^OK:"; then
  ok "Node spawn+stdin works: ${NODE_TEST:0:100}"
else
  fail "Node spawn+stdin failed:"
  echo "     $NODE_TEST"
fi

echo ""

# ─── 8. Summary ───
echo "═══════════════════════════════════════════"
echo "  Diagnostic Complete"
echo "═══════════════════════════════════════════"
echo ""

# Cleanup
rm -f /tmp/aurora_diag_test.jpg /tmp/aurora_diag_stderr.txt /tmp/aurora_diag_saved.jpg

echo "If everything above is green, your engine works."
echo "If something is red, copy this output and send it to me."
echo ""
echo "Reminder: delete these old files if they still exist:"
echo "  rm -f enhance.py adjust.py analyzer.py"
echo ""