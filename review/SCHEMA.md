# 검수 결과(교정) 파일 형식 — `firebim.review/1`

검수화면(`*_검수화면.html` / `X_review.html`)의 「검수 결과 저장」이 내려받는 `<건물id>_corrections.json`.
용도 ① `python -m review.apply <결과.json> <corrections.json>` 로 다시 적용 → 수정 결과·3D·IFC 재생성
용도 ② `python -m review.labels <corrections.json> --out 학습데이터` → 학습 라벨(COCO) + 검수 기반 성능(⑤ 학습 순환)

## 좌표 원칙
- 기호·문·벽은 **도면 이미지 px 좌표**(PDF에서 뽑은 평면도 원본 해상도, 원점 왼쪽 위, y 아래로 +)로 저장한다.
  엔진의 축척·층정합이 나중에 바뀌어도 같은 도면에 그대로 다시 적용되고, 학습 라벨로 바로 쓸 수 있다.
- 적용 시 전역 좌표(m)로 변환: `x = px / s + tx`, `y = ty − py / s` (`s, tx, ty` = 결과 JSON `plans[i]`).
- 여러 층이 한 도면(기준층 평면도)을 쓰면 도면 단위로 한 번 고치고, 그 도면의 모든 층에 적용한다.

## 최상위
| 키 | 내용 |
|---|---|
| `schema` | `"firebim.review/1"` |
| `tool`, `created`, `updated` | 만든 도구, 처음/마지막 저장 시각(ISO) |
| `building` | `{id, name, use, result_sha, pdf_name, pdf_sha256}` — `result_sha` 로 어느 변환 결과에 대한 검수인지 확인 |
| `source` | `{result_json, pdf}` 내부 PC 경로(반출 금지) |
| `summary` | `{items_total, items_done, plans_total, plans_done, auto_total}` — 검수 부담: 자동인식 `auto_total` 중 사람이 본 `items_total` (+ 선택 `xcheck_total, xcheck_done, vertical_added`) |
| `plans` | `{ "<도면 i>": Plan }` |
| `flags` | `[{key:"kind|floor", kind, floor, decision:"confirm"|"reject"}]` 교차검증 판정(confirm = 불일치 맞음, reject = 문제 없음 → 3D 확인사항에서 제외) |
| `vertical` | `[{kind:"stairs"|"elevs", k, name, box_m:[x0,y0,x1,y1], floors, decision:"ok"|"reject"}]` 계단·승강기 판정(reject → 제거, 위치로 매칭) |
| `vertical_added` | (선택) 검수자가 추가한 계단·승강기 `[{id, kind:"stairs"|"elevs", name, plan, box_px:[x0,y0,x1,y1], box_m, floors}]` — 아래 절 |
| `xcheck` | (선택) 교차검증(규칙 ↔ AI 모델) 항목 판정 `[{id, kind:"xwall"|"xdoor", sub, plan, p, seg, loc, o, ref, tpx|wpx, filled, bridge, ext, decision}]` — 아래 절 |
| `items` | `[{id, kind, plan, ref, p, decision}]` 검수 필요 항목별 처리(`p` = 의심도, `decision` = yes/no/fixed/null; `xwall`/`xdoor` 는 model/rule/later/null) |
| `ops` | 작업 기록 `[{t, plan, op, ...}]` (감사·분석용, 적용에는 안 씀) |

## Plan
| 키 | 내용 |
|---|---|
| `floors`, `caption`, `img_w`, `img_h`, `s` | 층 목록, 도면 제목, 이미지 크기(px), 축척(px/m) |
| `status` | `{align:"ok"|"bad"|null, scale:"ok"|"bad"|null, done:bool}` — `done` 인 도면만 학습 정답으로 사용 |
| `shift_m` | `[dx, dy]` 검수자가 맞춘 층 위치 보정(m, 전역 좌표; y 는 북쪽 +) |
| `symbols` | **최종** 기호 목록 `[{id, type, px, py, box:[x0,y0,x1,y1], box_est, conf, src, verified, orig}]` |
| `symbols_deleted` | 삭제한 자동 기호(오검출) |
| `doors` | **최종** 문 목록 `[{id, o:"h"|"v", px, py, w(px), ext, src}]` |
| `doors_deleted` | 삭제한 자동 문 |
| `walls_deleted` | 삭제한 자동 벽 `[{id, o, c, a, b, t}]` (px; 가로벽 o=h: y=c, x=a..b) |
| `walls_added` | 추가한 벽(같은 형식, `src:"added"`) |

- `type`: `exit` 비상구 · `hose` 옥내소화전 · `panel` 수신기 · `gas` 가스밸브 · `ev` 승강기
- `src`: `auto`(자동 그대로) · `added`(검수자 추가) · `moved` · `retyped` · `moved+retyped` · 문은 `edited`
- `orig`: 수정 전 `{type, px, py}` — 오검출/위치오차 분석용
- `verified`: 검수자가 「맞음」으로 확인
- `box_est`: 상자 크기를 모르는 기호(표시용 추정 상자). 적용 시 결과 JSON 에는 상자를 넣지 않는다.

## 교차검증 판정 `xcheck` (모델 경로 결과에만 있음, 2026-09-30 추가)
검수 화면 `xwall`/`xdoor` 항목(규칙 추출기와 STR 모델이 다르게 본 벽·문 조각, 의심도 p)에 대한 판정. 기하는 **도면 px** 로 함께 저장한다.
- `sub`: `rule_only`(규칙만 벽으로 봄) · `model_only`(모델만 벽/문으로 봄) · `added`(모델 문을 엔진이 이미 보충함). `filled` = 모델 벽을 엔진이 이미 보충함, `bridge` = 규칙은 개구부로 끊음
- `seg` `[x0,y0,x1,y1]`, `o` 방향, `tpx` 벽 두께(px) / `wpx` 문 폭(px), `loc` 중심, `ref` 해당 자동 벽 id(있으면)
- `decision`: `model` 모델이 맞음 · `rule` 규칙이 맞음(자동 결과 유지) · `later` 나중에(적용·라벨 없음)

| 항목 | `model` 적용(review.apply) | `rule` 적용 | 학습 라벨(review.labels) |
|---|---|---|---|
| xwall rule_only | 규칙 벽의 그 구간 잘라냄(남는 조각 < 0.3 m 버림) | 유지 | model → `wall_rejected` 음성 / rule → 벽 `confirmed` |
| xwall model_only | 벽 추가(양끝 0.4 m 연장, 같은 선 기존 벽 중심에 맞춤) + 그 구간 위 문 삭제(bridge) | 없음 | model → 벽 `src:"xcheck"`·`confirmed` / rule → `wall_rejected` |
| xwall model_only + filled | 유지(보충 벽 확정) | 보충 벽 구간 잘라냄 | model → 벽 `confirmed` / rule → `wall_rejected` |
| xdoor added | 유지 | 가장 가까운 문(0.6 m 안) 삭제 | rule → `door_rejected` |
| xdoor model_only | 문 추가(`src:"added", via:"xcheck"` → 결과 `review:"xcheck"`) | 없음 | rule → `door_rejected` |

적용 순서: 도면 문 목록(px)에 문 연산 → 벽은 `walls_deleted`/`walls_added` 다음에 자르기·추가 → 외곽 면적 재계산. 공통 로직 `review/xdec.py`.
결과 JSON `review.stats` 에 `xwall_cut, xwall_added, xdoor_deleted, xdoor_added, xdoor_dropped`, `review.xcheck` 에 판정 집계(모델이 맞은 비율).

## 계단·승강기 추가 `vertical_added` (2026-09-30 추가)
검수 화면 「계단 추가(S)」: 도면에 상자를 끌어 그리고 종류(계단/승강기)·연결층을 고른다.
- `box_px` = 그린 도면(`plan`) px 상자, `box_m` = 전역 m(참고용; 적용은 `box_px` + 현재 결과의 축척·정합으로 다시 계산), `floors` = 연결층
- 적용: 엔진 층간 클러스터와 같은 규칙 — 연결층은 최저~최고층 사이를 채우고, 같은 종류가 중심 1.5 m 안에 이미 있으면 연결층만 합침(`review:"merged"`),
  아니면 `res.stairs|elevs` 에 `review:"added"` 로 추가 → 3D·IFC(IfcStair / IfcTransportElement, Pset_소방활동 근거·출처 = '검수자 추가(검수 화면)'), 계단 수 교차검증 재계산.
- 순서: 승강기 기호 재묶음 → 오검출(reject) 제거 → 추가.

## 학습 라벨(COCO) — `review.labels`
- `images[]`: `{id, file_name:"images/<건물>_p<i>.png", width, height, building, plan, floors, px_per_m, reviewed, align, pdf_sha256}`
- `categories`: 1 exit · 2 hose · 3 panel · 4 gas · 5 ev · 6 door · 7 wall · 8 wall_rejected · 9 door_rejected · 10 stair · 11 elevator_shaft (8~11 추가, 1~7 번호 불변)
- `annotations[]`: `{id, image_id, category_id, bbox:[x,y,w,h], area, iscrowd:0, attributes}`
  - 기호: `attributes = {src, conf, verified, box_estimated, center}`
  - 문: bbox = 폭 × 벽두께(0.3m), `attributes = {src, orientation, external, segment}`
  - 벽: bbox + `segmentation`(두께 사각형), `attributes = {src, segment, thickness_px, confirmed?}` — `src:"xcheck"` = 교차검증 판정으로 추가, `confirmed` = 불일치 항목에서 검수자가 벽으로 확정
  - `wall_rejected`: 검수자가 '벽 아님'으로 판정한 띠(두께 max(벽 두께, 0.4 m)), `door_rejected`: '문 아님' 상자 — `attributes = {src:"xcheck", item, decision}`
    → `strseg.review_import` 는 이 영역에서 모델이 벽(문)으로 본 화소를 무시(255)가 아니라 **배경 0** 으로 학습(명시적 음성). 확정 벽·문이 겹치면 확정 우선, 모델 '창' 화소는 계속 무시.
  - `stair` / `elevator_shaft`: 자동(오검출 판정 제외) + 검수자 추가 상자(연결층이 겹치는 모든 도면에 투영), `attributes = {src:"auto"|"added", floors, name}` (STR 캐시에는 안 씀)
- `review_metrics.json`: 건물별 자동 vs 확정 정밀도·재현율(기호 종류별·문·벽), 플래그 확인/기각 수, 정합 정상/불량, 검수 부담,
  `xcheck`(교차검증 판정 집계·모델이 맞은 비율), `vertical`(오검출·추가 계단/승강기 수).
  정답 없는 실제 건물에서의 성능 지표로 `채점현황.html` 「검수 기반 성능」에 표시된다.
