import cv2 as cv
ref_bgr = cv.imread('org.jpg')
cmp_bgr = cv.imread('pap.jpg')
ref_y = cv.cvtColor(ref_bgr, cv.COLOR_BGR2YCrCb)[:, :, 0]
cmp_y = cv.cvtColor(cmp_bgr, cv.COLOR_BGR2YCrCb)[:, :, 0]

# 2) quality 모듈로 PSNR 계산 (단일 채널 → 결과도 스칼라 1개)
psnr_y = cv.quality.QualityPSNR_compute(ref_y, cmp_y)[0]  # 튜플 첫 칸만 의미 있음
print(psnr_y)
