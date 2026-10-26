# NOTICE

## 데이터 출처
- 이 저장소의 AI 모델(`models/*.onnx`)은 한국지능정보사회진흥원(NIA) **AI허브 「건축 도면 데이터」**를 활용하여 학습하였습니다.
  (https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71465)
  원천 이미지와 라벨은 AI허브 이용정책에 따라 이 저장소에 포함하지 않으며 재배포하지 않습니다.
- 건축물대장 항목명은 국토교통부 건축HUB 건축물대장 정보 서비스(공공데이터포털 15134735)의 항목명을 따릅니다.
- 정보카드 양식은 「소방활동 자료조사 등에 관한 규정」(소방청 훈령) 별지 서식을 바탕으로 만든 빈 양식입니다.
- 이 저장소에는 실제 건물의 정보카드·도면이 들어 있지 않습니다. 시험용 건물은 모두 `synth/` 가 만든 가상 건물입니다.

## 주요 오픈소스 의존성
| 이름 | 라이선스 |
|---|---|
| PyMuPDF | AGPL-3.0 |
| OpenCV | Apache-2.0 |
| NumPy, SciPy, scikit-image, Shapely, NetworkX | BSD 계열 |
| Pillow | MIT-CMU (HPND) |
| IfcOpenShell | LGPL-3.0 |
| ONNX Runtime | MIT |
| Tesseract OCR | Apache-2.0 |
| three.js | MIT |
| esbuild | MIT |
| PyTorch (학습 시에만) | BSD-3-Clause |
| Playwright (시험·화면 캡처 시에만) | Apache-2.0 |
| python-docx (가상 양식 생성 시에만) | MIT |
