# -*- coding: utf-8 -*-
"""기능 12·18·24 + 활동계획(인명검색·호스·공기호흡기·훈련 채점) 수치 기준 — 한 곳에서 관리.

- LAW  : 법령·화재안전기술기준에 적힌 값(출처 표기). 기준 개정 시에만 바꾼다.
- SOP  : 기관 SOP 값으로 설정 필요 — 차량 제원·부서 원칙·호스 길이 등은 본부·서마다 다르다.
         아래 기본값은 시험용 가정값이며, 실제 운용 전 기관 SOP 값으로 바꿔야 한다.
- TOL  : 교차검증 허용 오차(내부 판단 기준).
사용: from firebim.sop import CFG  /  CFG 를 복사해 값만 바꿔 각 함수에 cfg= 로 넘길 수 있다.
"""

LAW = {
    # NFTC 203(자동화재탐지설비 및 시각경보장치) 2.1.1 경계구역
    'zone_max_area_m2': 600.0,            # 하나의 경계구역 면적 600㎡ 이하
    'zone_max_side_m': 50.0,              # 한 변의 길이 50m 이하
    'zone_open_hall_area_m2': 1000.0,     # 단서: 주된 출입구에서 내부 전체가 보이는 것은 한 변 50m 범위 내 1,000㎡ 이하
    'zone_two_floor_area_m2': 500.0,      # 단서: 500㎡ 이하 범위 안에서는 2개 층을 하나의 경계구역으로 할 수 있음
    # NFTC 203 2.1.2 계단·경사로·승강로 등은 별도 경계구역, 수직거리 45m 이하마다
    'zone_vertical_max_m': 45.0,
    # 건축법 시행령 제51조④ · 건축물방화구조규칙 — 소방관 진입창
    'rescue_window_max_floor': 11,        # 2층 이상 11층 이하인 층에 설치
    'rescue_window_spacing_m': 40.0,      # 벽면 끝에서 수평거리 40m 이내마다
    'rescue_window_sill_m': 0.8,          # 창 아랫부분이 실내 바닥면으로부터 80cm 이내
}

SOP = {
    # ── 아래 값은 모두 "기관 SOP 값으로 설정 필요" (시험용 가정값) ──
    'floor_height_m': 3.3,                # 층고 기본값(건축물대장 '높이'가 있으면 높이÷지상층수로 대체)
    'aerial_ladder_len_m': 53.0,          # 고가사다리차 사다리 최대 길이(차량 제원)
    'aerial_max_outreach_m': 25.0,        # 사다리 전개(작업) 반경 — 정격하중 기준 수평 도달거리
    'aerial_turntable_h_m': 3.0,          # 턴테이블(사다리 기점) 높이
    'aerial_min_road_width_m': 6.0,       # 아웃트리거 전개에 필요한 도로(공지) 폭
    'aerial_max_positions': 2,            # 추천할 사다리차 부서 위치 수
    'aerial_min_gap_m': 8.0,              # 사다리차·펌프차 부서 위치 간 최소 간격
    'engine_min_road_width_m': 4.0,       # 소방펌프차 진입 가능 도로 폭
    'engine_max_setback_m': 20.0,         # 외벽면에서 이 거리 이내에 부서 가능하면 '접근 가능' 면
    'hose_len_m': 15.0,                   # 소방호스 1본 길이(65mm·40mm 공통 가정)
    'site_sample_m': 2.0,                 # 도로 위 후보 위치 간격
}

TOL = {
    'daejang_area_rel': 0.05,             # 대장↔정보카드 층 면적 허용 오차
    'daejang_gfa_rel': 0.03,              # 대장 연면적↔정보카드 연면적
    'drawing_area_rel': 0.12,             # 도면 추정 면적이 한쪽과 '맞는다'고 볼 오차(도면은 축척 오차 포함)
    'water_dist_rel': 0.30,               # 정보카드 소방용수 거리 ↔ 부지 계산 거리
    'water_dist_abs_m': 40.0,
    'site_reg_scale_rel': 0.15,           # 부지 건물외곽 ↔ 도면 외곽 크기 차 허용
}

CFG = dict(LAW, **SOP, **TOL)

# ── 활동계획(3D 지휘화면 화점 입력 후 산출) — 모두 임시값(기관 SOP 로 설정 필요). export.build 가 D["sop_search"]·D["sop_act"]·D["sop_train"] 로 뷰어에 넘김
SEARCH = {
    '_note': '인명검색 우선순위 기준값 — 모두 임시값(기관 SOP로 설정 필요)',
    'night_hours': [22, 6],                      # 야간 = 22시~06시(취침 재실 가정)
    # 범주별 가중치 [주간, 야간] — 등급 경계 bands 로 높음/보통/낮음
    'cat_weight': {
        'crit': [1.0, 1.0],      # 거동 불가 실(중환자실·수술실·격리병실·신생아실 등) — 실명이 있을 때만
        'sleep': [0.6, 1.0],     # 취침 가능 실(침실·병실·객실·세대·생활실)
        'occupied': [0.7, 0.3],  # 주간 재실 실(사무·교실·진료·판매·로비)
        'hall': [0.5, 0.2],      # 대공간(작업장·보관장)
        'corr': [0.3, 0.3],      # 복도·홀(검색 경로상 통과하며 확인)
        'svc': [0.1, 0.1],       # 설비·창고·주차·화장실
        'unk': [0.45, 0.45],     # 모름
    },
    'mobility_bonus': 0.2,                       # 정보카드 ⑩ 층 용도가 병동·노유자·요양 등(거동 곤란 재실) → 가중치 가산
    'bands': [0.8, 0.45],                        # 가중치 ≥0.8 높음 · ≥0.45 보통 · 그 밖 낮음
    'class_order_fire': ['iso', 'fire', 'shared', 'near', 'detour', 'ok'],
    'class_order_upper': ['none', 'above', 'hard', 'ok'],
    'floors_above': 2,                           # 화재층 위로 몇 개 층까지 순서를 매길지(직상층 = 1)
    'above_radius_m': 6.0,                       # 위층 '화점 직상부' 판정 반경(대피로 화점 차단 반경과 같게)
    'fire_cross_factor': 5.0,                    # 검색 거리 계산 시 화점 반경 안 격자 비용 배수(통과는 가능, 억제)
    'dist_ref': 'stair',                         # 'stair' = 공격계단 기준 · 'fire' = 화점 기준(옛 방식)
    'walk_mps': 0.8,                             # 장비 착용 보행속도(공기호흡기 진입 시간과 같은 가정)
    'stair_s_per_floor': 25.0,                   # 계단 1개 층 이동 시간(가정)
    'markers_fire': 12, 'markers_upper': 6,      # 3D 번호 표시 수(층별)
    'route_n': 6,                                # 화재층 검색 순서 경로(공격계단 → 1 → 2 …)를 그릴 실 수(0 = 안 그림)
}

ACT = {
    '_note': '호스·공기호흡기·진입 시간 기준값 — 모두 임시값(기관 SOP로 설정 필요).',
    'hose_len_m': 15.0,                          # 호스 1본 길이(SOP['hose_len_m'] 와 같게)
    'hose_slack_m': 0.0,                         # 여유 길이 α — N = ceil((L + α) / L_h). 0 이면 이전 결과와 같음
    'hose_carry': 2,                             # 기본 적재(휴대) 본수 — 넘으면 '추가 호스 필요'
    'walk_mps': 0.8,                             # 장비 착용 보행속도 v
    'stair_s_per_floor': 25.0,                   # 계단 1개 층 이동 시간 τ
    'scba_cyl_l': 6.8,                           # 공기호흡기 용기 용량 V (L)
    'scba_fill_bar': 300.0,                      # 충전압 P_f
    'scba_return_bar': 55.0,                     # 귀환 기준 잔압 P_r
    'scba_lpm': 55.0,                            # 활동 강도별 분당 소모량 q (L/min, 대기압 환산)
    'scba_margin_min': 5.0,                      # 안전여유 t_m (분)
    # 공격계단 점수(작을수록 우선) = 진입층 거리 + 화재층 거리 − 방수구 가점 + 피난 수요 벌점
    'atk_w_standpipe_m': 0.0,                    # 화재층 계단 둘레 standpipe_r_m 안에 연결송수관 방수구(도면 위치·규칙 추정)가 있으면 이만큼(m) 가점(추정 '후보'는 절반). 0 = 끔(기본 — '거리 합 최소' 그대로), 예 10
    'standpipe_r_m': 5.0,                        # NFPC 502 제6조 — 방수구는 계단으로부터 5 m 이내
    'atk_w_demand_m': 0.0,                       # 그 계단으로 대피하는 높은 등급 실(인명검색 범주) 1곳당 벌점(m) — 0 = 끔(기본)
}

TRAIN = {
    '_note': '훈련 모드 채점 배점·허용 범위 — 모두 임시값(교육 훈련 지침으로 설정 필요)',
    'pts_stair': 25, 'pts_route': 25, 'pts_hose': 15, 'pts_search': 25, 'pts_time': 10,
    'route_ratio_zero': 1.5,                     # 경로 길이 비(훈련자/산출안) 1.0 → 만점, 이 값 이상 → 0점(선형)
    'hose_tol': 1,                               # 호스 본수 ±1 이내 → 만점
    'search_k': 6,                               # 검색 순서: 훈련자가 화재층에서 고르는 실 수(산출안 상위 k 와 비교)
    'time_full_s': 120, 'time_zero_s': 300,      # 결정 시간 ≤120 s 만점, ≥300 s 0점(선형)
}


# ── 화재 확산 가정 시뮬레이션 · 대원 이동(3D 지휘화면 v3 시간축) — 모두 예시·임시값(기관 SOP·공학 검토로 설정 필요). export.build 가 D["sop_sim"] 로 뷰어(viewer/src/sim.js·crew.js)에 넘김
# 모델: 실(대피 분석 자동 분할)·계단실·승강로(층마다)·바깥을 마디로, 문·맞닿은 벽(문 미검출)·외벽·계단실 문·승강로 위아래를 가지로 한 단순 구획 확산.
#   연기 지수 s = 연기량 ÷ 실 부피(0~1). 화원 실 연기 발생 = smoke_max_m3s × min(1, 경과/grow_s). 가지 흐름 = C × (s_진한 − s_옅은) (계단·승강로는 위로 C_up, 아래로 C_dn).
#   s ≥ untenable 또는 불 붙은 지 burn_untenable_s 초 → '진입·대피 곤란'. 불은 같은 층 이웃 실로 문 spread_door_s · 문 미검출 경계 spread_wall_s 뒤 번짐.
#   실제 연기·열 거동(CFAST·FDS 등 공학 모델)과 다르며 현장 판단을 대신하지 않는다.
SIM = {
    '_note': '화재 확산 가정 시뮬레이션 — 모두 예시·임시값(기관 SOP·공학 검토로 설정 필요)',
    't_max_min': 15, 'dt_s': 2, 'frame_s': 5,            # 재생 범위(분)·계산 간격·저장 간격(초)
    'smoke_max_m3s': 6.0, 'grow_s': 240,                 # 화원 연기 발생 최대(㎥/s)·성장 시간(초)
    'untenable': 0.5, 'burn_untenable_s': 30,            # 곤란 판정: 연기 지수 · 불 붙은 뒤 시간
    'c_door': 0.5, 'c_wall': 0.12, 'c_ext': 0.04, 'c_extdoor': 0.3,      # 전도도(㎥/s): 문(검출) · 문 미검출 경계(닫힌 문 누설 가정) · 외벽 누설 · 외부 출입문
    'c_stair_door': 0.15, 'c_stair_door_sp': 0.04, 'c_ev_door': 0.06,   # 계단실 문 · 특별피난계단(부속실) · 승강기 문
    'c_stack_up': 2.5, 'c_stack_dn': 0.25, 'c_ev_up': 1.5, 'c_ev_dn': 0.15,   # 계단실·승강로 위/아래(굴뚝 효과)
    'spread_door_s': 300, 'spread_wall_s': 720,          # 불 번짐 시간: 문으로 이어진 실 · 맞닿은 실(문 미검출)
    'dispatch_s': 60, 'drive_kmh': 30, 'default_arrive_s': 360, 'setup_s': 60,   # 출동 준비·주행 속도(관할센터 거리 ÷)·거리 없을 때 도착·현장 전개
    'team2_delay_s': 90, 'rescue_delay_s': 30, 'search_dwell_s': 40,           # 진압2 뒤따름 · 구조팀 진입 지연 · 실 1개 검색 시간
}
