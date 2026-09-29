@echo off
REM ── เปิดแดชบอร์ด AI ทำนายการยกเลิกการจองโรงแรม (ทีม PGD 032-034-038) ──
cd /d "%~dp0"
echo กำลังเปิดแอป... เบราว์เซอร์จะเปิดที่ http://localhost:8501
echo (ปิดหน้าต่างนี้เพื่อหยุดแอป)
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
