# -*- coding: utf-8 -*-
"""OpenCV 4.x / 5.x 호환: 5.0 부터 cv2.HoughLinesP 가 (N,4) 로 돌려준다(4.x 는 (N,1,4)).
엔진 코드는 4.x 모양(ls[:, 0] → (N,4))을 가정하므로, 5.x 에서는 결과를 (N,1,4) 로 맞춰 준다."""
try:
    import cv2 as _cv2
    if int(_cv2.__version__.split('.')[0]) >= 5 and not getattr(_cv2.HoughLinesP, '_fb_compat', False):
        _orig = _cv2.HoughLinesP

        def _hough_lines_p(*a, **k):
            r = _orig(*a, **k)
            return r if r is None or r.ndim == 3 else r.reshape(-1, 1, 4)
        _hough_lines_p._fb_compat = True
        _cv2.HoughLinesP = _hough_lines_p
except Exception:          # cv2 가 없거나 버전 문자열이 특이하면 그대로 둔다
    pass
