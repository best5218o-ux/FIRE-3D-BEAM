# FIRE-3D BEAM

**소방활동 정보카드(PDF) → 3D 지휘화면(HTML 1개) + IFC4 건물 모델 자동 변환 도구**

소방관서가 이미 가지고 있는 소방활동 정보카드(표 + 층별 평면도 + 진압작전도)를 넣으면,
층별 벽·문·계단·승강기·소방시설을 읽어 출동 중에도 바로 열어 볼 수 있는 3D 지휘화면과 IFC4 파일을 만듭니다.
인터넷 없이 동작하고(내부망 설치판 포함), 결과 HTML 은 설치 없이 크롬·엣지에서 열립니다.

> ⚠ **참고용 도구입니다.** 도면과 현장은 다를 수 있습니다. 화면의 「확인 필요」 항목은 현장에서 확인하고,
> 호스 본수·공기호흡기 귀환 한계·인명검색 순서 같은 계산값은 **가정 기반 참고값**입니다(`firebim/sop.py` 의 값은 모두 임시값 — 기관 SOP 로 바꿔 쓰세요).
> 투입·철수 판단은 현장지휘관이 합니다.

---

## 1. 무엇을 하나

| 단계 | 내용 | 주요 파일 |
|---|---|---|
| 정보카드 해석 | 표준양식(①~⑪) 자동 판별, 건물개황·층별 용도·소방시설·위험물 항목 읽기, 스캔본 OCR, **관계인 성명·연락처 자동 가림** | `firebim/card.py`, `card_ocr.py` |
| 평면도 인식 | 벽·문·창·계단·승강기·실, 범례 기호(비상구·소화전·수신기 등), 치수로 축척 계산 | `firebim/geom.py`, `vision.py`, `realcard.py`, `realplan.py`, `str_seg.py`, `spa_seg.py` |
| 층간 정합 | 계단·승강기·벽 겹침으로 층을 맞춰 쌓기 | `firebim/align.py`, `pipeline.py` |
| 교차검증 | 정보카드 ↔ 도면 ↔ 건축물대장(건축HUB 항목명) 3자 대조, 다르면 「확인 필요」 | `firebim/daejang.py`, `xcheck.py` |
| 소방 기능 | 수신기 경계구역 추정(NFTC 203), 부지·소방차 접근·사다리차 부서 후보, 진압작전도 인식, 진입로·피난로, 인명검색 우선순위 | `firebim/zones.py`, `site.py`, `sitemap.py`, `evac.py`, `search.py` |
| 출력 | 3D 지휘화면(three.js, HTML 1개) + IFC4(IfcSpace·IfcZone·소방시설 형식) | `firebim/export.py`, `viewer/` |
| 검수 | 의심도 순 검수 화면 → 고친 내용을 3D·IFC 에 반영 + 학습 라벨로 저장 | `review/` |
| 내부망 설치판 | Windows·인터넷 없음·관리자 권한 없음용 설치판 빌드 | `offline/` |

AI 모델(`models/*.onnx`)은 onnxruntime 만으로 CPU 에서 돌아갑니다(torch 불필요).

## 2. 설치 (개발 PC)

```bash
pip install -r requirements.txt
# Tesseract OCR + 한국어 언어팩 (Ubuntu: apt install tesseract-ocr tesseract-ocr-kor)
cd viewer && npm install && cd ..        # 3D 지휘화면 빌드(three.js, esbuild)
```
가상 건물 생성에 표준양식 docx→PDF 변환을 쓰면 LibreOffice 가 필요합니다. 학습(`strseg/`, `colab/`)에는 torch 가 필요합니다.

내부망(Windows, 인터넷 없음) PC 에는 `bash offline/build_bundle.sh` 로 설치판을 만들어 옮깁니다 → `offline/설치안내.md`, `offline/시범운용_절차서.md`.

## 3. 바로 써 보기 (가상 건물)

실제 정보카드 없이 시험할 수 있도록 **정답이 딸린 가상 건물·정보카드 생성기**가 들어 있습니다.

```bash
python -m synth.make 6                              # 가상 건물 18동(복도형·대공간형·코어형) → data/
python make_demo.py data/<건물>/card.pdf out/demo   # 정보카드 PDF → <건물>.html(3D) + <건물>.ifc
python run_all.py 6                                 # 전체 변환 + 자동 채점 → out/summary.json
python -m review.make out/<건물>.json               # 검수 화면
python -m review.apply out/<건물>.json 검수결과.json # 검수 반영 → 3D·IFC 재생성
```

선택 입력: `--daejang 대장.json|pdf`(건축물대장), `--site site.json`(부지), `--zones 경계구역일람도.json`.
정보카드 빈 양식: `templates/소방활동정보카드_표준양식.docx`.

### 회귀 기준 (이 저장소 그대로, `python run_all.py 6`)

| 지표 | 결과 |
|---|---|
| 층 정합 | 120/120 |
| 벽 재현율/정밀도 | 0.937 / 0.927 |
| 문 재현율/정밀도 | 0.700 / 0.895 |
| 계단 / 승강기 | 35/39 (오검출 0) / 33/34 |
| 정보카드 항목 / 교차검증 | 120/120 / 39/39 |
| 비상구·소화전·수신기·가스밸브 기호 | R 1.00 / P 1.00 |

가상 건물 기준이며 실제 도면 성능과는 다릅니다.

## 4. AI 모델과 학습 데이터

- `models/str_seg.onnx`(벽·창·문), `models/spa_seg.onnx`(실·계단·승강기): 한국지능정보사회진흥원 **AI허브 「건축 도면 데이터」** 로 학습했습니다.
- **원천 이미지·라벨은 이 저장소에 없고, 재배포하지 않습니다.** 다시 학습하려면 각자 AI허브에서 승인받아 내려받으세요:
  https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71465
- 내려받은 뒤: `tools/aihub/build_index.py`(라벨 색인) → `tools/aihub/prep.py`(긴 변 2048 축소) → `strseg/train.py` 또는 `colab/*.ipynb`(GPU).
  기본 경로는 `./aihub`, 다르면 환경변수 `AIHUB` / `AIHUB_ROOT` / `AIHUB_OBJ2`.
- 실제 도면 평가(`evalr/realplan_eval.py`, `evalr/realsym_eval.py`)의 손 정답 파일은 AI허브 원천에서 만든 것이라 넣지 않았습니다.

## 5. 보안·개인정보 수칙 (이 도구를 쓰는 기관에)

- 실제 소방활동 정보카드·건물 도면과 그 변환 결과는 **기관 내부 PC 밖으로 반출하지 말고, 이 저장소나 이슈·PR 에 올리지 마세요.** 문제 제보는 가상 건물로 재현해 주세요.
- 관계인 성명·연락처는 기본으로 결과에서 빠지며, 변환할 때마다 결과 전체를 검사합니다.
- 내부망 설치판은 실행 중 외부 연결을 코드에서 막아 둡니다.

## 6. 폴더 구성

```
firebim/   변환 엔진          viewer/    3D 지휘화면(three.js)   review/  검수 화면
synth/     가상 건물 생성기    evalr/     자동 채점               strseg/  모델 학습·내보내기
models/    ONNX 모델           offline/   내부망 설치판           tools/aihub/  AI허브 데이터 준비
colab/     GPU 학습 노트북     realplan/  실제 도면 전처리 설명   templates/    정보카드 빈 양식
```
기능별 상세 설명은 각 모듈 첫머리 주석과 `review/SCHEMA.md`, `realplan/README.md` 에 있습니다.

## 7. 라이선스

Copyright (C) 2026 김영민

이 프로그램은 **GNU Affero General Public License v3.0**(AGPL-3.0)으로 배포합니다 — [`LICENSE`](LICENSE).
고쳐 쓰는 것은 자유이며, 고친 프로그램을 배포하거나 네트워크 서비스로 제공하면 그 소스도 같은 조건으로 공개해야 합니다.
함께 쓰는 오픈소스와 데이터 출처는 [`NOTICE.md`](NOTICE.md) 를 보세요.
