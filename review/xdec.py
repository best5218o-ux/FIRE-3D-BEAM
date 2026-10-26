# -*- coding: utf-8 -*-
"""교차검증(규칙 ↔ AI 모델) 검수 판정 → 벽·문 편집 연산. review.apply(전역 m)와 review.labels(도면 px)가 같이 쓴다.

교정 JSON `xcheck` 목록(형식: review/SCHEMA.md) 항목마다 decision:
  'model' = 모델이 맞음, 'rule' = 규칙이 맞음(자동 결과 유지), 'later' = 나중에(아무것도 안 바꿈)
판정 → 연산(모두 도면 px 좌표; 길이 허용값은 m 로 두고 단위 변환 계수 u(1 m 당 단위)로 환산)
  xwall rule_only  + model → cut(규칙 벽의 그 구간 삭제)            · 라벨: wall_rejected(명시적 '벽 아님')
  xwall rule_only  + rule  → (변경 없음)                            · 라벨: 벽 확정(confirmed)
  xwall model_only + model → add(모델 벽 조각을 벽으로 추가, 양끝 0.4 m 연장) — 이미 보충(filled)된 것은 변경 없음
                              + 그 구간 위의 문(규칙이 개구부로 본 자리) 삭제          · 라벨: 벽 확정
  xwall model_only + rule  → 보충(filled)된 벽이면 cut, 아니면 변경 없음 · 라벨: wall_rejected
  xdoor added      + rule  → del_door(모델이 보충한 문 삭제)        · 라벨: door_rejected
  xdoor model_only + model → add_door                              · 라벨: 문(검수 문 목록에 들어감)
"""
import math

EXT_M = 0.4          # 추가 벽 양끝 연장(이웃 벽과 겹쳐 가짜 틈 방지) = firebim.xcheck CFG['tol_model']
SNAP_M = 0.3         # 추가 벽 중심선을 같은 선의 기존 벽에 맞추는 거리
TOL_RULE_M = 0.1     # 규칙 벽 조각 자르기: 같은 선 허용
TOL_FILL_M = 0.3     # 보충 벽 자르기: 보충 시 중심선을 이웃 벽에 맞췄으므로 넓게
MIN_LEFT_M = 0.3     # 자르고 남은 조각이 이보다 짧으면 버림
DOOR_NEAR_M = 0.6    # 모델 보충 문 매칭 거리
NEG_T_M = 0.4        # 'wall_rejected' 라벨 띠 두께 최소(m) — 모델 벽 예측이 중심선에서 조금 어긋나도 덮도록
DECISIONS = ('model', 'rule', 'later')


def seg_wall(seg, o):
    """[x0,y0,x1,y1](px) + 방향 → {o,c,a,b}(px)"""
    x0, y0, x1, y1 = seg
    if o == 'h': return dict(o='h', c=(y0 + y1) / 2, a=min(x0, x1), b=max(x0, x1))
    return dict(o='v', c=(x0 + x1) / 2, a=min(y0, y1), b=max(y0, y1))


def entries(corr, plan=None):
    """판정된 교차검증 항목(decision ∈ model/rule/later)"""
    out = []
    for e in corr.get('xcheck') or []:
        if e.get('decision') not in DECISIONS: continue
        if plan is not None and int(e.get('plan', -1)) != int(plan): continue
        out.append(e)
    return out


def plan_ops(corr, plan):
    """도면 1장의 판정 → 연산 목록(px). op: cut / add / del_door / add_door / drop_doors / neg / confirm"""
    ops = []
    for e in entries(corr, plan):
        d, k, sub = e['decision'], e.get('kind'), e.get('sub')
        if d == 'later': continue
        if k == 'xwall':
            w = dict(seg_wall(e['seg'], e['o']), t=float(e.get('tpx') or 1.5))
            if sub == 'rule_only':
                if d == 'model': ops += [dict(op='cut', w=w, tol=TOL_RULE_M, id=e['id']), dict(op='neg', cat='wall_rejected', w=w, id=e['id'])]
                else: ops.append(dict(op='confirm', w=w, id=e['id']))
            else:
                if d == 'model':
                    if not e.get('filled'): ops += [dict(op='add', w=w, id=e['id']), dict(op='drop_doors', w=w, id=e['id'])]
                    ops.append(dict(op='confirm', w=w, id=e['id']))
                else:
                    if e.get('filled'): ops.append(dict(op='cut', w=w, tol=TOL_FILL_M, id=e['id']))
                    ops.append(dict(op='neg', cat='wall_rejected', w=w, id=e['id']))
        elif k == 'xdoor':
            px, py = e['loc']
            door = dict(o=e['o'], px=px, py=py, w=float(e.get('wpx') or 0), ext=bool(e.get('ext')))
            if sub == 'added':
                if d == 'rule': ops += [dict(op='del_door', d=door, id=e['id']), dict(op='neg', cat='door_rejected', d=door, id=e['id'])]
            else:
                if d == 'model': ops.append(dict(op='add_door', d=door, id=e['id']))
                else: ops.append(dict(op='neg', cat='door_rejected', d=door, id=e['id']))
    return ops


# ───────── 벽 연산(단위 무관: u = 1 m 당 단위 — m 면 1, px 면 축척 s) ─────────
def cut_walls(walls, w, tol_c, min_left=0.0):
    """같은 선(|c|≤tol_c) 벽에서 [a,b] 구간을 잘라냄(남은 조각이 min_left 미만이면 버림) → (새 목록, 잘린 벽 수)"""
    out, n = [], 0
    for v in walls:
        if v['o'] != w['o'] or abs(v['c'] - w['c']) > tol_c or v['b'] <= w['a'] or v['a'] >= w['b']:
            out.append(v); continue
        n += 1
        out += [dict(v, b=w['a'])] if w['a'] - v['a'] >= max(min_left, 1e-6) else []
        out += [dict(v, a=w['b'])] if v['b'] - w['b'] >= max(min_left, 1e-6) else []
    return out, n


def add_wall(walls, w, u, src='xcheck'):
    """모델 벽 조각을 벽으로 추가(같은 선 기존 벽이 80% 이상 덮으면 생략) → (새 목록, 추가 여부)"""
    same = [v for v in walls if v['o'] == w['o'] and abs(v['c'] - w['c']) < SNAP_M * u]
    cov = sum(max(0.0, min(v['b'], w['b']) - max(v['a'], w['a'])) for v in same)
    if cov >= 0.8 * (w['b'] - w['a']): return walls, False
    c = min(same, key=lambda v: abs(v['c'] - w['c']))['c'] if same else w['c']
    nw = dict(o=w['o'], c=c, a=w['a'] - EXT_M * u, b=w['b'] + EXT_M * u, t=w['t'], review=src)
    return walls + [nw], True


def run_wall_ops(walls, ops, u, conv=None):
    """벽 목록에 cut/add 적용. conv: px 벽 → 목록 단위 변환 함수(apply 에서 px→m). 반환 (새 목록, 통계)"""
    st = dict(cut=0, added=0)
    for op in ops:
        if op['op'] not in ('cut', 'add'): continue
        w = conv(op['w']) if conv else dict(op['w'])
        if op['op'] == 'cut':            # 자르기 경계는 판정 구간 그대로(남은 조각은 MIN_LEFT 이상만)
            walls, n = cut_walls(walls, w, op['tol'] * u, MIN_LEFT_M * u); st['cut'] += n
        else:
            walls, ok = add_wall(walls, w, u); st['added'] += ok
    return walls, st


# ───────── 문 연산(항상 도면 px, s = px/m) ─────────
def run_door_ops(doors, ops, s):
    """검수 문 목록(px)에 del_door / add_door / drop_doors 적용 → (새 목록, 통계)"""
    doors = [dict(d) for d in doors]; st = dict(door_deleted=0, door_added=0, door_dropped=0)
    for op in ops:
        if op['op'] == 'del_door':
            q = op['d']
            if not doors: continue
            k, dist = min(((k, math.hypot(d['px'] - q['px'], d['py'] - q['py'])) for k, d in enumerate(doors)), key=lambda t: t[1])
            if dist <= DOOR_NEAR_M * s: doors.pop(k); st['door_deleted'] += 1
        elif op['op'] == 'add_door':
            q = op['d']
            if any(math.hypot(d['px'] - q['px'], d['py'] - q['py']) < DOOR_NEAR_M * s for d in doors): continue
            doors.append(dict(id=f"x{op['id']}", o=q['o'], px=q['px'], py=q['py'], w=q['w'] or 0.9 * s, ext=q['ext'], src='added', via='xcheck'))
            st['door_added'] += 1
        elif op['op'] == 'drop_doors':
            w = op['w']; keep = []
            for d in doors:
                u_, v_ = (d['px'], d['py']) if w['o'] == 'h' else (d['py'], d['px'])
                on = d['o'] == w['o'] and abs(v_ - w['c']) <= SNAP_M * s and w['a'] - 0.1 * s <= u_ <= w['b'] + 0.1 * s
                if on: st['door_dropped'] += 1
                else: keep.append(d)
            doors = keep
    return doors, st


def summary(corr, plans=None):
    """판정 집계: {sub: {model, rule, later}} + 모델이 맞은 비율(검수 기반 교차검증 성능)"""
    out = {}
    for e in entries(corr):
        if plans is not None and str(e.get('plan')) not in plans: continue
        key = f"{e.get('kind')}:{e.get('sub')}" + (':filled' if e.get('filled') else '') + (':bridge' if e.get('bridge') else '')
        c = out.setdefault(key, dict(model=0, rule=0, later=0)); c[e['decision']] += 1
    m = sum(v['model'] for v in out.values()); r = sum(v['rule'] for v in out.values())
    return dict(by_kind=out, decided=m + r, model_right=m, rule_right=r, model_right_rate=round(m / (m + r), 3) if m + r else None)
