# -*- coding: utf-8 -*-
"""소방시설 분류표(한 곳) — 정보카드 '층별 소방시설 설치현황' 항목명 → 법정 분류 · IFC 형식 · 위치 확실성 방식

근거(2026-09-30 확인):
- 분류: 「소방시설 설치 및 관리에 관한 법률 시행령」 [별표 1] 소방시설(제3조) — 소화설비·경보설비·피난구조설비·소화용수설비·소화활동설비
  (찾기쉬운 생활법령 easylaw.go.kr csmSeq=1574 '소방시설의 이해' 정리본, 경보설비 '화재알림설비'는 [별표 4] 시행 2023.12.1. 에 설치대상 명시)
- 설치대상: 같은 시행령 [별표 4](시행 2023.12.1.) — 연결송수관: 5층 이상 & 연면적 6천㎡ 이상 또는 지하 포함 7층 이상,
  비상콘센트: 11층 이상 건물의 11층 이상 층, 스프링클러: 6층 이상이면 모든 층
- 방수구 위치: 「연결송수관설비의 화재안전성능기준(NFPC 502)」 제6조 — 층마다 설치(아파트 1·2층 등 단서), 계단(아파트·1,000㎡ 미만 층은 1개,
  1,000㎡ 이상 층은 2개)으로부터 5 m 이내, 추가 설치 수평거리 25 m(지하층 합계 3,000㎡ 이상)/50 m, 11층 이상 쌍구형, 호스접결구 0.5~1 m
- 비상콘센트 위치: 「비상콘센트설비의 화재안전기술기준(NFTC 504)」 2.1.5.1(높이 0.8~1.5 m) · 2.1.5.2(1,000㎡ 미만 층은 계단 출입구 5 m 이내,
  1,000㎡ 이상 층은 각 계단 출입구 또는 부속실 출입구 5 m 이내(계단 3개 이상이면 2개), 추가 25 m/50 m)
- 비상용승강기: 「건축법」 제64조 제2항(높이 31 m 초과 건축물) — 소방시설(별표 1)이 아닌 건축설비이므로 분류 '소방활동 관련 건축설비'
위치 확실성(certainty): located = 평면도 범례 기호로 위치 확인 · estimated = 규칙(NFPC/NFTC)으로 추정한 위치(반경 표시) · present = 정보카드 수량만(위치 없음)
"""
import re

CATS = ['소화설비', '경보설비', '피난구조설비', '소화용수설비', '소화활동설비', '소방활동 관련 건축설비', '기타']

# code, 법정 명칭, 대분류, 중분류(별표 1 묶음), 정보카드 별칭(정규화 전), IFC (class, PredefinedType, ObjectType), 단위, 도면 기호 종류, 위치 추정 규칙
_C = [
    # ── 소화설비
    ('EXT', '소화기', '소화설비', '소화기구', ['소화기', '수동식소화기', '분말소화기', '대형소화기', '소형소화기', 'ABC소화기', '이산화탄소소화기', '할론소화기', 'K급소화기'],
     ('IfcFireSuppressionTerminal', 'USERDEFINED', 'FIREEXTINGUISHER'), '개', 'ext', None),
    ('EXT_AUTO', '자동확산소화기', '소화설비', '소화기구', ['자동확산소화기', '자동확산소화용구'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'AUTOMATICEXTINGUISHER'), '개', None, None),
    ('EXT_SIMPLE', '간이소화용구', '소화설비', '소화기구', ['간이소화용구', '투척용소화용구', '소공간용소화용구'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'SIMPLEEXTINGUISHER'), '개', None, None),
    ('AUTO_KITCHEN', '주거용 주방자동소화장치', '소화설비', '자동소화장치', ['주거용주방자동소화장치', '주방자동소화장치', '주거용자동소화장치'],
     ('IfcFireSuppressionTerminal', 'USERDEFINED', 'KITCHENSUPPRESSION'), '개', None, None),
    ('AUTO_ETC', '자동소화장치', '소화설비', '자동소화장치', ['자동소화장치', '상업용주방자동소화장치', '캐비닛형자동소화장치', '가스자동소화장치', '분말자동소화장치', '고체에어로졸자동소화장치'],
     ('IfcFireSuppressionTerminal', 'USERDEFINED', 'AUTOMATICSUPPRESSION'), '개', None, None),
    ('HOSE', '옥내소화전설비', '소화설비', '옥내소화전설비', ['옥내소화전', '옥내소화전설비', '호스릴옥내소화전', '호스릴소화전', '소화전(옥내)', '옥내소화전함'],
     ('IfcFireSuppressionTerminal', 'HOSEREEL', None), '개', 'hose', None),
    ('SSP', '간이스프링클러설비', '소화설비', '스프링클러설비등', ['간이스프링클러', '간이스프링클러설비', '간이스프링클러헤드', '간이S/P', '캐비닛형간이스프링클러'],
     ('IfcFireSuppressionTerminal', 'SPRINKLER', None), '개(헤드)', None, None),
    ('ESFR', '화재조기진압용 스프링클러설비', '소화설비', '스프링클러설비등', ['화재조기진압용스프링클러', '화재조기진압용스프링클러설비', 'ESFR', 'ESFR헤드'],
     ('IfcFireSuppressionTerminal', 'SPRINKLER', None), '개(헤드)', None, None),
    ('SP', '스프링클러설비', '소화설비', '스프링클러설비등', ['스프링클러', '스프링클러설비', '스프링클러헤드', '스프링클러설비(헤드)', 'S/P', 'S/P헤드', 'SP헤드', 'SP', '스프링쿨러', '스프링쿨러헤드'],
     ('IfcFireSuppressionTerminal', 'SPRINKLER', None), '개(헤드)', None, None),
    ('WSPRAY', '물분무소화설비', '소화설비', '물분무등소화설비', ['물분무', '물분무소화설비', '물분무헤드'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'WATERSPRAY'), '개', None, None),
    ('MIST', '미분무소화설비', '소화설비', '물분무등소화설비', ['미분무', '미분무소화설비', '미분무헤드'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'WATERMIST'), '개', None, None),
    ('FOAM', '포소화설비', '소화설비', '물분무등소화설비', ['포소화설비', '포헤드', '포소화설비(헤드)', '포소화전', '폼헤드'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'FOAM'), '개', None, None),
    ('CO2', '이산화탄소소화설비', '소화설비', '물분무등소화설비', ['이산화탄소소화설비', 'CO2소화설비', 'CO2', '이산화탄소'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'CO2'), '개', None, None),
    ('HALON', '할론소화설비', '소화설비', '물분무등소화설비', ['할론소화설비', '할론', '할론1301'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'HALON'), '개', None, None),
    ('CLEAN', '할로겐화합물 및 불활성기체소화설비', '소화설비', '물분무등소화설비',
     ['할로겐화합물및불활성기체소화설비', '청정소화약제소화설비', '청정소화설비', '불활성기체소화설비', '할로겐화합물소화설비', '할로겐화합물및불활성기체'],
     ('IfcFireSuppressionTerminal', 'USERDEFINED', 'CLEANAGENT'), '개', None, None),
    ('POWDER', '분말소화설비', '소화설비', '물분무등소화설비', ['분말소화설비'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'POWDER'), '개', None, None),
    ('ALKALI', '강화액소화설비', '소화설비', '물분무등소화설비', ['강화액소화설비'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'WETCHEMICAL'), '개', None, None),
    ('AEROSOL', '고체에어로졸소화설비', '소화설비', '물분무등소화설비', ['고체에어로졸소화설비', '고체에어로졸'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'AEROSOL'), '개', None, None),
    ('OUTHYD', '옥외소화전설비', '소화설비', '옥외소화전설비', ['옥외소화전', '옥외소화전설비'], ('IfcFireSuppressionTerminal', 'FIREHYDRANT', None), '개', None, None),
    # ── 경보설비
    ('ALONE', '단독경보형 감지기', '경보설비', '단독경보형감지기', ['단독경보형감지기', '단독경보형화재감지기', '주택용화재경보기'], ('IfcSensor', 'SMOKESENSOR', None), '개', None, None),
    ('BELL', '비상경보설비', '경보설비', '비상경보설비', ['비상경보설비', '비상벨', '비상벨설비', '자동식사이렌', '자동식사이렌설비'], ('IfcAlarm', 'BELL', None), '개', None, None),
    ('DET', '자동화재탐지설비(감지기)', '경보설비', '자동화재탐지설비', ['자동화재탐지기(감)', '자동화재탐지설비(감지기)', '감지기', '화재감지기', '자탐(감)', '연기감지기', '열감지기', '차동식감지기', '광전식감지기'],
     ('IfcSensor', 'FIRESENSOR', None), '개', None, None),
    ('ZONE', '자동화재탐지설비(경계구역·회로)', '경보설비', '자동화재탐지설비', ['자동화재탐지기(회)', '자동화재탐지설비(회로)', '경계구역', '회로', '자탐(회)'], (None, None, None), '회로', None, None),
    ('PANEL', '자동화재탐지설비(수신기)', '경보설비', '자동화재탐지설비', ['수신기', '수신반', '화재수신기', 'P형수신기', 'R형수신기', '복합형수신기'],
     ('IfcUnitaryControlElement', 'ALARMPANEL', None), '대', 'panel', None),
    ('VISUAL', '시각경보기', '경보설비', '시각경보기', ['시각경보기', '시각경보장치', '시각경보'], ('IfcAlarm', 'LIGHT', None), '개', None, None),
    ('NOTIFY', '화재알림설비', '경보설비', '화재알림설비', ['화재알림설비', '화재알림형감지기'], ('IfcAlarm', 'USERDEFINED', 'FIRENOTIFICATION'), '개', None, None),
    ('PA', '비상방송설비', '경보설비', '비상방송설비', ['비상방송', '비상방송설비', '비상방송(스피커)', '비상방송스피커', '확성기'], ('IfcAudioVisualAppliance', 'SPEAKER', None), '개(스피커)', None, None),
    ('AUTOCALL', '자동화재속보설비', '경보설비', '자동화재속보설비', ['자동화재속보설비', '자동화재속보기', '속보기', '속보설비'], ('IfcCommunicationsAppliance', 'USERDEFINED', 'FIREAUTODIALER'), '대', None, None),
    ('MONITOR', '통합감시시설', '경보설비', '통합감시시설', ['통합감시시설'], ('IfcUnitaryControlElement', 'USERDEFINED', 'INTEGRATEDMONITOR'), '식', None, None),
    ('LEAK', '누전경보기', '경보설비', '누전경보기', ['누전경보기'], ('IfcAlarm', 'USERDEFINED', 'EARTHLEAKAGE'), '대', None, None),
    ('GASLEAK', '가스누설경보기', '경보설비', '가스누설경보기', ['가스누설경보기', '가스누설탐지기', '가스경보기'], ('IfcSensor', 'GASSENSOR', None), '개', None, None),
    # ── 피난구조설비
    ('ESCAPE', '피난기구', '피난구조설비', '피난기구', ['피난기구', '완강기', '간이완강기', '구조대', '피난사다리', '미끄럼대', '피난교', '피난용트랩', '공기안전매트', '다수인피난장비', '승강식피난기', '하향식피난구용내림식사다리', '하향식피난구'],
     ('IfcDistributionElement', None, 'ESCAPEDEVICE'), '개', None, None),
    ('RESCUE', '인명구조기구', '피난구조설비', '인명구조기구', ['인명구조기구', '공기호흡기', '방열복', '방화복', '인공소생기'], ('IfcDistributionElement', None, 'RESCUEEQUIPMENT'), '개', None, None),
    ('GUIDE', '유도등', '피난구조설비', '유도등', ['유도등', '피난구유도등', '통로유도등', '객석유도등', '유도표지', '피난유도선', '거실통로유도등', '복도통로유도등', '계단통로유도등'],
     ('IfcLightFixture', 'SECURITYLIGHTING', None), '개', None, None),
    ('EMLIGHT', '비상조명등', '피난구조설비', '비상조명등및휴대용비상조명등', ['비상조명등', '휴대용비상조명등', '비상조명'], ('IfcLightFixture', 'SECURITYLIGHTING', None), '개', None, None),
    # ── 소화용수설비
    ('CITYW', '상수도소화용수설비', '소화용수설비', '상수도소화용수설비', ['상수도소화용수설비', '소화용수설비'], ('IfcFireSuppressionTerminal', 'FIREHYDRANT', None), '개', None, None),
    ('TANK', '소화수조', '소화용수설비', '소화수조·저수조', ['소화수조', '저수조', '소화수조·저수조', '소화용수조'], ('IfcTank', 'STORAGE', None), '개', None, None),
    # ── 소화활동설비
    ('SMOKE', '제연설비', '소화활동설비', '제연설비', ['제연설비', '제연', '부속실제연', '제연설비(부속실)', '거실제연', '제연설비(거실)', '제연구역', '급기가압'], ('IfcDistributionElement', None, 'SMOKECONTROL'), '구역', None, None),
    ('STANDPIPE', '연결송수관설비(방수구)', '소화활동설비', '연결송수관설비', ['연결송수관(방수구)', '연결송수관설비(방수구)', '방수구', '연결송수관', '연결송수관설비', '송수관방수구', '연송(방수구)'],
     ('IfcFireSuppressionTerminal', 'USERDEFINED', 'LANDINGVALVE'), '개', 'outlet', 'nfpc502'),
    ('SIAMESE', '연결송수관설비(송수구)', '소화활동설비', '연결송수관설비', ['연결송수관(송수구)', '연결송수관설비(송수구)', '송수구', '쌍구형송수구', '연송(송수구)'],
     ('IfcFireSuppressionTerminal', 'BREECHINGINLET', None), '개', 'siamese', None),
    ('SPCONN', '연결살수설비', '소화활동설비', '연결살수설비', ['연결살수', '연결살수설비', '연결살수헤드', '연결살수(헤드)'], ('IfcFireSuppressionTerminal', 'SPRINKLER', None), '개(헤드)', None, None),
    ('OUTLET', '비상콘센트설비', '소화활동설비', '비상콘센트설비', ['비상콘센트', '비상콘센트설비', '비상전원콘센트'], ('IfcOutlet', 'POWEROUTLET', None), '개', 'socket', 'nftc504'),
    ('RADIO', '무선통신보조설비', '소화활동설비', '무선통신보조설비', ['무선통신보조설비', '무선통신보조', '무선통신보조설비(안테나)', '누설동축케이블', '무선통신보조(접속단자)'],
     ('IfcCommunicationsAppliance', 'ANTENNA', None), '개', None, None),
    ('CABLEPROT', '연소방지설비', '소화활동설비', '연소방지설비', ['연소방지설비', '연소방지'], ('IfcFireSuppressionTerminal', 'USERDEFINED', 'CABLETUNNEL'), '개', None, None),
    # ── 소방활동 관련 건축설비(별표 1 밖)
    ('EMEV', '비상용승강기', '소방활동 관련 건축설비', '비상용승강기(건축법 제64조)', ['비상용승강기', '비상승강기', '승강기(비상)', '비상용엘리베이터', '피난용승강기'],
     ('IfcTransportElement', 'ELEVATOR', None), '대', None, 'elevator'),
]

CATALOG = {c[0]: dict(code=c[0], name=c[1], cat=c[2], group=c[3], aliases=c[4], ifc=c[5], unit=c[6], symbol=c[7], rule=c[8]) for c in _C}
SYMBOL_CODE = {v['symbol']: k for k, v in CATALOG.items() if v['symbol']}        # 도면 기호 종류 → 분류 코드
RULES = {
    'nfpc502': dict(radius=5.0, ref='NFPC 502 제6조(방수구: 층마다, 계단으로부터 5 m 이내, 1,000㎡ 이상 층은 계단 2개, 11층 이상 쌍구형)'),
    'nftc504': dict(radius=5.0, ref='NFTC 504 2.1.5.2(비상콘센트: 계단 출입구 5 m 이내, 1,000㎡ 이상 층은 각 계단(3개 이상이면 2개))'),
    'elevator': dict(radius=None, ref='건축법 제64조 제2항·「건축물의 설비기준 등에 관한 규칙」 제10조(승강장 6㎡ 이상·부속실 겸용·표지·피난층 출구 30 m) — '
                                     '비상용승강기는 도면 승강기 중 비상용 글자·승강장(부속실) 구획·계단 근접·전 층 운행 순으로 추정(대수=정보카드)'),
}


def norm(s):
    s = re.sub(r'\s+', '', s or '').replace('ㆍ', '·').upper()
    return s.replace('（', '(').replace('）', ')').replace('：', ':')


_ALIAS = {}
for _k, _v in CATALOG.items():
    for _a in [_v['name']] + _v['aliases']:
        _ALIAS.setdefault(norm(_a), _k)
_ALIAS_BY_LEN = sorted(_ALIAS.items(), key=lambda kv: -len(kv[0]))


def lookup(raw):
    """정보카드 항목명 → (code|None, match: 'exact'|'partial'|None). 부분 일치는 가장 긴 별칭(예: '간이스프링클러'가 '스프링클러'보다 먼저)"""
    k = norm(raw)
    if k in _ALIAS: return _ALIAS[k], 'exact'
    k2 = re.sub(r'\([^)]*\)$', '', k)                  # 괄호 꼬리 제거 후 재시도 (예: '방수구(쌍구형)')
    if k2 in _ALIAS: return _ALIAS[k2], 'exact'
    for a, code in _ALIAS_BY_LEN:
        if len(a) >= 2 and a in k: return code, 'partial'
    return None, None


ITEM_RE = re.compile(r'([^\s:,;][^:;]*?)\s*:\s*([\d][\d,]*)')


def parse_items(txt):
    """'층별 소방시설 설치현황' 셀 글자 → [{code, name, category, group, count, raw, match}] (모르는 항목은 code '기타', 원문 유지)"""
    out = []
    for m in ITEM_RE.finditer(txt or ''):
        raw = m.group(1).strip(); n = int(m.group(2).replace(',', '') or 0)
        code, how = lookup(raw)
        if code:
            c = CATALOG[code]
            out.append(dict(code=code, name=c['name'], category=c['cat'], group=c['group'], count=n, raw=raw, match=how))
        else:
            out.append(dict(code='기타', name=raw, category='기타', group='기타', count=n, raw=raw, match=None))
    return out


def merge_items(items):
    """같은 코드(기타는 원문) 합치기 — 원문 별칭은 목록으로 보존"""
    acc = {}
    for it in items:
        k = it['code'] if it['code'] != '기타' else '기타:' + it['raw']
        if k in acc: acc[k]['count'] += it['count']; acc[k]['raw'] += ' + ' + it['raw']
        else: acc[k] = dict(it)
    return list(acc.values())
