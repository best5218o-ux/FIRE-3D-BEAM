#!/usr/bin/env bash
# 내부 PC용 오프라인 설치판 만들기 — 인터넷 되는 PC(리눅스/WSL/맥)에서 실행.
#   bash offline/build_bundle.sh            # → offline/build/FIRE-3D-BEAM_offline/ + FIRE-3D-BEAM_offline.zip + .zip.sha256
# 환경변수:
#   PYFULL=3.11.9            Windows embeddable Python 버전(3.11.x 마지막 바이너리 배포판)
#   TESS_VER=5.4.0.20240606  UB Mannheim Tesseract 빌드
#   WITH_SYNTH=1             채점용 가상 정보카드(data/, 약 11MB) 포함 (0 이면 제외 — 자체시험 표본 samples/ 는 항상 포함)
#   SAMPLE=data_std2/hall_1404  자체시험 표본(가상 표준양식 카드 + 진압작전도 + 대장·부지). 정답 gt.json 은 넣지 않음
#   WITH_V3=0                예전 STR 모델(str_seg_v3.onnx, 비교·되돌리기용) 포함 — 기본 제외(기본 모델 = GPU 모델 str_seg.onnx)
#   ZIP=1                    FIRE-3D-BEAM_offline.zip 도 만들기
#   LINUX_TEST=1             같은 고정 버전의 리눅스 휠로 '오프라인 설치 → 자체시험(변환·AI 교차검증·검수 반영)'을 이 PC에서 재현
# 넣지 않는 것: synth/(가상 생성 — python-docx·LibreOffice·playwright), strseg/·학습 스크립트(torch), AI허브 자료·정답, data_hold/ out*/ 등
#              개발 산출물, 실제 정보카드, selftest_review.py(playwright — 개발 PC 전용)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; APP="$(dirname "$HERE")"
PYFULL="${PYFULL:-3.11.9}"; PYV="${PYFULL%.*}"; PYTAG="${PYV/./}"
TESS_VER="${TESS_VER:-5.4.0.20240606}"
WITH_SYNTH="${WITH_SYNTH:-1}"; ZIP="${ZIP:-1}"; LINUX_TEST="${LINUX_TEST:-0}"; WITH_V3="${WITH_V3:-0}"; SAMPLE="${SAMPLE:-data_std2/hall_1404}"
BUILD="$HERE/build"; OUT="$BUILD/FIRE-3D-BEAM_offline"; CACHE="$BUILD/cache"
PY="${PYTHON:-python3}"; T="$HERE/bundle_tools.py"
mkdir -p "$CACHE"; rm -rf "$OUT"; mkdir -p "$OUT"/{wheelhouse,installers/tessdata}
fetch() { # url dest
  [ -s "$2" ] && return 0
  echo "  받기: $1"; curl -fSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

echo "■ 1) 3D 뷰어 번들(viewer/dist/app.js) — 내부 PC에는 node 없음"
if [ -x "$APP/viewer/node_modules/.bin/esbuild" ]; then
  (cd "$APP" && env -u FIRE3DBEAM_PREBUILT "$PY" -c "import build_viewer; build_viewer.bundle()")
fi
[ -s "$APP/viewer/dist/app.js" ] || { echo "viewer/dist/app.js 없음 (cd viewer && npm install 후 다시)"; exit 1; }

echo "■ 2) wheelhouse: win_amd64 / Python $PYV 휠만(--only-binary)"
"$PY" -m pip download --disable-pip-version-check -q --platform win_amd64 --python-version "$PYV" --implementation cp --abi "cp$PYTAG" \
  --only-binary=:all: -d "$OUT/wheelhouse" -r "$HERE/requirements-win.txt" pip
echo "  검증: wheelhouse 만으로(인터넷 없이) 전부 풀리는지"
"$PY" -m pip install --disable-pip-version-check -q --dry-run --ignore-installed --no-index --find-links "$OUT/wheelhouse" \
  --platform win_amd64 --python-version "$PYV" --implementation cp --abi "cp$PYTAG" --only-binary=:all: --target "$BUILD/_dryrun" \
  -r "$HERE/requirements-win.txt" --report "$BUILD/resolve_report.json"
"$PY" "$T" report "$BUILD/resolve_report.json"
rm -rf "$BUILD/_dryrun"

echo "■ 3) Windows embeddable Python $PYFULL"
fetch "https://www.python.org/ftp/python/$PYFULL/python-$PYFULL-embed-amd64.zip" "$CACHE/python-$PYFULL-embed-amd64.zip" || \
  echo "  ! python.org 에서 못 받음 → 설치판에 python\\ 없음(install.bat 가 PC의 Python 3.11 로 venv 를 만듦 — 설치안내.md 참고)"
[ -s "$CACHE/python-$PYFULL-embed-amd64.zip" ] && "$PY" "$T" pth "$CACHE/python-$PYFULL-embed-amd64.zip" "$OUT/python" "$PYTAG"

echo "■ 4) Tesseract OCR $TESS_VER (UB Mannheim) + 언어팩(tessdata_fast: kor, eng — 개발 PC와 같은 모델)"
TESS_EXE="tesseract-ocr-w64-setup-$TESS_VER.exe"
fetch "https://github.com/UB-Mannheim/tesseract/releases/download/v$TESS_VER/$TESS_EXE" "$CACHE/$TESS_EXE" || \
  fetch "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.0.20221214.exe" "$CACHE/$TESS_EXE" || echo "  ! Tesseract 설치파일을 못 받음 → 설치안내.md 참고해 installers/ 에 직접 넣기"
for L in kor eng; do fetch "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/$L.traineddata" "$CACHE/$L.traineddata"; cp "$CACHE/$L.traineddata" "$OUT/installers/tessdata/"; done
if [ -s "$CACHE/$TESS_EXE" ] && command -v 7z >/dev/null; then
  echo "  7z 로 설치파일을 풀어 '관리자 권한 없이 쓰는' 휴대용 tesseract\\ 폴더 구성"
  rm -rf "$BUILD/_tess"; 7z x -y -o"$BUILD/_tess" "$CACHE/$TESS_EXE" >/dev/null
  mkdir -p "$OUT/tesseract"
  (cd "$BUILD/_tess" && find . -maxdepth 1 ! -name . ! -name '$PLUGINSDIR' ! -name 'uninstall*' -exec cp -r {} "$OUT/tesseract/" \;)
  rm -rf "$BUILD/_tess"
  [ -f "$OUT/tesseract/tesseract.exe" ] || { echo "  ! 풀기 실패 → 설치파일을 installers/ 에 둠"; rm -rf "$OUT/tesseract"; cp "$CACHE/$TESS_EXE" "$OUT/installers/"; }
  if [ -d "$OUT/tesseract" ]; then
    mkdir -p "$OUT/tesseract/tessdata"; cp "$OUT"/installers/tessdata/*.traineddata "$OUT/tesseract/tessdata/"
    "$PY" "$T" tess-prune "$OUT/tesseract"     # 학습 도구·문서·Java, tesseract.exe 가 쓰지 않는 DLL 제거(pefile)
    # 디버그 정보만 제거(libtesseract-5.dll 97→4 MB, 폴더 132→38 MB). 코드 영역은 그대로 — Wine 에서 원본과 OCR 결과(TSV) 바이트 동일 확인(2026-10-03)
    if [ "${STRIP_TESS:-1}" = 1 ] && objdump -i 2>/dev/null | grep -q pe-x86-64; then
      for f in "$OUT"/tesseract/*.dll "$OUT"/tesseract/tesseract.exe; do strip --strip-debug "$f" 2>/dev/null || true; done
      echo "  디버그 정보 제거: $(du -sh "$OUT/tesseract" | cut -f1)"
    fi
    rm -rf "$OUT/installers"                   # 휴대용이 있으면 설치파일·언어팩 사본 불필요
  fi
elif [ -s "$CACHE/$TESS_EXE" ]; then
  echo "  7z 없음 → 설치파일을 installers/ 에 둠(install.bat 가 실행)"; cp "$CACHE/$TESS_EXE" "$OUT/installers/"
fi

echo "■ 5) 프로그램(app/) — 실행에 필요한 것만(+ 금지 파일 검사)"
"$PY" "$T" app "$APP" "$OUT/app" "$WITH_SYNTH" "$WITH_V3" "$SAMPLE"
echo "  실행 코드 import ↔ wheelhouse 대조"
"$PY" "$T" imports "$OUT/app" "$OUT/wheelhouse"

echo "■ 6) 실행 파일(.bat, CRLF) + 설치안내·시범운용 절차서 + MANIFEST + SHA256SUMS"
for f in "$HERE"/win/*.bat; do sed 's/\r$//; s/$/\r/' "$f" > "$OUT/$(basename "$f")"; done
cp "$HERE/설치안내.md" "$HERE/시범운용_절차서.md" "$OUT/"
"$PY" "$T" manifest "$OUT" "$PYFULL" "$TESS_VER"
( cd "$OUT" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt )

echo "■ 7) 크기"
du -sh "$OUT/wheelhouse" "$OUT/python" "$OUT/app" "$OUT/app/models" "$OUT"/tesseract "$OUT/installers" 2>/dev/null || true
du -sh "$OUT"
if [ "$ZIP" = 1 ]; then
  rm -f "$BUILD/FIRE-3D-BEAM_offline.zip"
  "$PY" "$T" zip "$OUT" "$BUILD/FIRE-3D-BEAM_offline.zip"
  ( cd "$BUILD" && sha256sum FIRE-3D-BEAM_offline.zip > FIRE-3D-BEAM_offline.zip.sha256 && cat FIRE-3D-BEAM_offline.zip.sha256 )
fi

if [ "$LINUX_TEST" = 1 ]; then
  echo "■ 8) 재현 시험(리눅스): 같은 고정 버전을 인터넷 없이 새 가상환경에 설치 → 자체시험(변환·AI 교차검증·검수 반영)"
  LW="$BUILD/linux_wheelhouse"; mkdir -p "$LW"
  grep -v '^msvc-runtime' "$HERE/requirements-win.txt" > "$BUILD/requirements-linux-test.txt"      # msvc-runtime 은 Windows 전용
  "$PY" -m pip download --disable-pip-version-check -q --only-binary=:all: -d "$LW" -r "$BUILD/requirements-linux-test.txt" pip
  rm -rf "$BUILD/_venv" "$BUILD/_home"; "$PY" -m venv --without-pip "$BUILD/_venv"
  PIPWHL="$(ls "$LW"/pip-*.whl | head -1)"
  "$BUILD/_venv/bin/python" "$PIPWHL/pip" install -q --no-index --find-links "$LW" pip          # install.bat 과 같은 부트스트랩
  "$BUILD/_venv/bin/python" -m pip install -q --no-index --find-links "$LW" -r "$BUILD/requirements-linux-test.txt"
  mkdir -p "$BUILD/_home"
  FIRE3DBEAM_HOME="$BUILD/_home" FIRE3DBEAM_NO_OPEN=1 "$BUILD/_venv/bin/python" -X utf8 "$OUT/app/offline/launch.py" selftest
fi
echo "완료: $OUT"
