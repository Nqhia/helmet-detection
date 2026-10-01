"""helmet_pipeline — nhận diện người đi xe máy không đội mũ bảo hiểm (không train, dùng model sẵn có).

Pipeline: COCO detector (person/motorcycle/bicycle) -> helmet detector (helmet/no_helmet head-level
hoặc rider-level) -> association đầu->người->xe -> tracker theo nhóm xe -> bỏ phiếu N/M theo track
-> sự kiện vi phạm + ảnh bằng chứng (+ ALPR tuỳ chọn).
"""
__version__ = "0.1.0"
