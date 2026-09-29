# -*- coding: utf-8 -*-
"""Streamlit app — ทีม PGD (PlerngDoiGun) 032-034-038
AI ทำนายการยกเลิกการจองโรงแรม (Hotel Booking Cancellation)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI ทำนายการยกเลิกการจองโรงแรม | ทีม PGD",
                   page_icon="🏨", layout="wide")

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "team_model.joblib")


@st.cache_data
def load_meta():
    with open(BASE_DIR / "team_features.json", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
meta = load_meta()
FEATURES = meta["features"]
FMETA = meta.get("feature_meta", {})
TEAM = [(c, n) for c, n in meta.get("members", [["032", "รมิดา แจ่มจํารัส"],
                                               ["034", "รัตนาภรณ์ ธรรมนู"],
                                               ["038", "วิศรุต อินโต"]])]
TEST_ACC = float(meta.get("test_accuracy", 0.7796))
CV_ACC = float(meta.get("cv_accuracy", 0.7693))
BASE_CV = float(meta.get("baseline_all_features_cv", 0.7047))
N_ROWS = int(meta.get("n_rows", 87371))
CANCEL_RATE = float(meta.get("cancel_rate", 0.2749))
IMP = meta.get("feature_importance", {})

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', 'Leelawadee UI', Tahoma, sans-serif; }

  .hero {
      background: linear-gradient(135deg, #7C2D12 0%, #B45309 55%, #F59E0B 100%);
      border-radius: 22px; padding: 28px 32px 24px; color: #fff; position: relative; overflow: hidden;
      box-shadow: 0 24px 48px -28px rgba(124, 45, 18, .75); margin-bottom: 18px;
  }
  .hero:after { content: ""; position: absolute; right: -70px; top: -70px; width: 240px; height: 240px;
                background: radial-gradient(circle, rgba(255,255,255,.22), transparent 65%); border-radius: 50%; }
  .hero .kicker { font-size: .82rem; letter-spacing: 2.2px; text-transform: uppercase; opacity: .85; }
  .hero h1 { font-size: 2rem; font-weight: 700; margin: 6px 0 8px; line-height: 1.25; }
  .hero p { margin: 0; opacity: .95; font-size: .97rem; line-height: 1.6; }
  .chips { margin-top: 14px; }
  .chip { display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.32);
          padding: 6px 14px; border-radius: 999px; font-size: .84rem; margin: 0 8px 8px 0; }
  .chip.solid { background: #FFFFFF; color: #7C2D12 !important; border-color: #FFFFFF; font-weight: 600; }

  div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px !important;
      border: 1px solid #F3E0C7 !important; background: #FFFDF9 !important; padding: 6px 4px; }
  div.stFormSubmitButton > button, div.stButton > button {
      background: linear-gradient(135deg, #B45309, #F59E0B) !important; color: #fff !important;
      border: none !important; border-radius: 12px !important; font-weight: 600 !important;
      padding: .62rem 1rem !important; }
  div.stFormSubmitButton > button:hover, div.stButton > button:hover { transform: translateY(-1px); }

  [data-testid="stMetric"] { background: linear-gradient(180deg, #FFFDF9, #FEF6E7);
      border: 1px solid #F3E0C7; border-radius: 14px; padding: 14px 18px; }
  [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #7C2D12 !important; }
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #8A6A4F !important; }

  /* พื้นแอปโทนสว่างเสมอ กันธีม Dark ของผู้ชม */
  [data-testid="stAppViewContainer"], .stApp { background: #FDF9F3 !important; color-scheme: light; }
  [data-testid="stHeader"] { background: transparent !important; }
  .stApp { color: #43260F; }
  [data-testid="stMarkdownContainer"] > p, [data-testid="stMarkdownContainer"] li,
  [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { color: #43260F; }
  .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] *,
  .stApp small { color: #8A6A4F !important; }
  /* คืนสีส่วนที่ออกแบบเอง */
  .hero h1, .hero p, .hero .kicker, .hero .chip { color: #FFFFFF !important; }
  .hero .chip.solid { color: #7C2D12 !important; }

  .badge { border-radius: 16px; padding: 18px 20px; margin-bottom: 12px; }
  .badge h3 { margin: 0 0 6px; font-size: 1.15rem; font-weight: 700; color: inherit !important; }
  .badge p { margin: 0; font-size: .94rem; line-height: 1.6; color: inherit !important; }
  .badge .tag { display: inline-block; font-size: .72rem; letter-spacing: 1.3px; text-transform: uppercase;
                font-weight: 700; opacity: .8; margin-bottom: 6px; color: inherit !important; }
  .badge.risk { background: #FDECEC; border: 1px solid #F5C6C0; color: #8F2C22 !important; }
  .badge.safe { background: #E9F7EF; border: 1px solid #BFE7D0; color: #1B6B42 !important; }
  .mini { display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 2px; }
  .mini div { flex: 1 1 120px; background: #FFFDF9; border: 1px solid #F3E0C7; border-radius: 12px;
              padding: 10px 12px; text-align: center; }
  .mini span { display: block; font-size: .74rem; color: #8A6A4F !important; }
  .mini strong { font-size: 1.02rem; color: #7C2D12 !important; }
  .footer { text-align: center; color: #A08A72 !important; font-size: .84rem; margin-top: 26px; line-height: 1.8; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <div class="kicker">โครงงาน AI เพื่อธุรกิจดิจิทัล · ใบงานสัปดาห์ที่ 12</div>
  <h1>🏨 AI ทำนายการยกเลิกการจองโรงแรม</h1>
  <p>ระบบประเมินว่าลูกค้าท่านนี้ <b>มีแนวโน้มยกเลิกการจอง</b> หรือจะมาเข้าพักจริง —
     ใช้โมเดล Random Forest ที่คัดฟีเจอร์ด้วยขั้นตอนวิธีเชิงพันธุกรรม (GA) จากข้อมูลการจองจริง {N_ROWS:,} รายการ<br>
     วิชา 306-23-06 / DT36822N ปัญญาประดิษฐ์เพื่อธุรกิจดิจิทัล</p>
  <div class="chips">
    <div class="chip solid">ทีม PGD (PlerngDoiGun) · 032-034-038</div>
    {''.join(f'<div class="chip">{c} {n}</div>' for c, n in TEAM)}
    <div class="chip">GA: {meta.get("n_features_all", 72)} → {meta.get("n_features_ga", 16)} ฟีเจอร์ · โมเดลใช้ {len(FEATURES)} ฟีเจอร์</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_model, tab_help = st.tabs(["🎯 ประเมินความเสี่ยงยกเลิก", "🧠 ข้อมูลโมเดล", "📘 วิธีใช้ & ข้อจำกัด"])

with tab_predict:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        values = {}
        with st.form("cancel_form"):
            st.markdown("**ข้อมูลการจองและลูกค้า**")
            st.caption("ฟอร์มตรงกับ 22 ฟีเจอร์ของโมเดล = 16 ตัวที่ GA คัด + 6 ตัวที่กรอกได้จริง "
                       "(จัดกลุ่มเป็นช่องกรอกทั้งหมด 16 ช่อง)")
            NUM = [f for f in FEATURES if FMETA.get(f, {}).get("kind") == "number"]
            c1, c2 = st.columns(2)
            for i, f in enumerate(NUM):
                info = FMETA[f]
                target = c1 if i % 2 == 0 else c2
                with target:
                    values[f] = st.number_input(f"• {info.get('label', f)}",
                                                min_value=int(info.get("min", 0)),
                                                max_value=int(info.get("max", 100)),
                                                value=int(info.get("default", 0)),
                                                help=info.get("help") or None)
            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                deposit = st.selectbox("💰 ประเภทเงินมัดจำ (Deposit type)",
                                       ["No Deposit / Refundable", "Non Refund"],
                                       help="Non Refund = มัดจำแบบไม่คืนเงิน")
                cust = st.selectbox("👥 ประเภทลูกค้า (Customer type)",
                                    ["Transient / Transient-Party", "Group"],
                                    help="Group = มาเป็นหมู่คณะ/กรุ๊ปทัวร์")
                seg = st.selectbox("📣 ช่องทางตลาด (Market segment)",
                                   ["Online TA", "Corporate", "Direct / Offline / อื่น ๆ"])
            with c4:
                month = st.selectbox("🌤️ เดือนที่เข้าพัก", ["May", "July", "December", "เดือนอื่น"])
                channel = st.selectbox("🔌 ช่องทางการจอง (Distribution channel)", ["GDS", "TA/TO / Direct"])
            c5, c6 = st.columns(2)
            with c5:
                rroom = st.selectbox("🛏️ ประเภทห้องที่จอง (Reserved)", ["D", "A / B / C / E / อื่น ๆ"])
            with c6:
                aroom = st.selectbox("🔑 ประเภทห้องที่ได้จริง (Assigned)",
                                     ["B", "D", "F", "H", "A / C / E / อื่น ๆ"])
            go = st.form_submit_button("ประเมินความเสี่ยงยกเลิก", type="primary", use_container_width=True)
    with right:
        if go:
            row = {f: 0 for f in FEATURES}
            for k, v in values.items():
                if k in row:
                    row[k] = v
            row["deposit_type_Non Refund"] = 1 if deposit == "Non Refund" else 0
            row["customer_type_Group"] = 1 if cust == "Group" else 0
            row["market_segment_Corporate"] = 1 if seg == "Corporate" else 0
            row["market_segment_Online TA"] = 1 if seg == "Online TA" else 0
            row["arrival_date_month_May"] = 1 if month == "May" else 0
            row["arrival_date_month_July"] = 1 if month == "July" else 0
            row["arrival_date_month_December"] = 1 if month == "December" else 0
            row["distribution_channel_GDS"] = 1 if channel == "GDS" else 0
            row["reserved_room_type_D"] = 1 if rroom.startswith("D") else 0
            for r in ("B", "D", "F", "H"):
                row[f"assigned_room_type_{r}"] = 1 if aroom.startswith(r) else 0

            X = pd.DataFrame([row], columns=FEATURES)
            pred = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            p = float(proba[list(model.classes_).index(1)]) if 1 in list(model.classes_) else float(pred)

            st.metric("โอกาสที่ลูกค้าจะยกเลิกการจอง", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))

            if pred == 1:
                st.markdown("""
                <div class="badge risk">
                  <div class="tag">ผลการประเมิน</div>
                  <h3>⚠️ มีแนวโน้ม “ยกเลิกการจอง”</h3>
                  <p><b>สิ่งที่ควรทำ:</b> ให้เจ้าหน้าที่โทรติดตามยืนยันการจองตั้งแต่เนิ่น ๆ
                  พิจารณาเก็บเงินมัดจำแบบไม่คืน หรือเสนอส่วนลด/สิทธิพิเศษเพื่อล็อกการเข้าพัก
                  และเตรียมขายห้องนี้ซ้ำหากลูกค้ายืนยันยกเลิก</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="badge safe">
                  <div class="tag">ผลการประเมิน</div>
                  <h3>✅ มีแนวโน้ม “มาเข้าพักตามจอง”</h3>
                  <p><b>สิ่งที่ควรทำ:</b> จัดห้องตามที่จองไว้และดูแลตามมาตรฐาน
                  ใช้โอกาสนี้อัปเซลบริการเสริม (อาหารเช้า/สปา/ทัวร์)
                  โดยยังไม่จำเป็นต้องเก็บมัดจำเพิ่ม</p>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="mini">
              <div><span>ระดับความเสี่ยง</span><strong>{'สูง' if p >= 0.5 else 'ต่ำ'}</strong></div>
              <div><span>เกณฑ์ตัดสินใจ</span><strong>0.50</strong></div>
              <div><span>ความแม่นโมเดล</span><strong>{TEST_ACC:.4f}</strong></div>
            </div>""", unsafe_allow_html=True)

            with st.expander("ดูค่าที่ส่งเข้าโมเดล + ฟีเจอร์ที่โมเดลให้ความสำคัญ"):
                st.dataframe(X.T.rename(columns={0: "ค่าที่ส่งเข้าโมเดล"}), use_container_width=True)
                st.markdown("**ฟีเจอร์ที่มีอิทธิพลมากที่สุด (จากโมเดล)**")
                for f, v in list(IMP.items())[:5]:
                    st.write(f"- {f} — **{v * 100:.1f}%**")
        else:
            st.info("กรอกข้อมูลการจองด้านซ้าย แล้วกด **ประเมินความเสี่ยงยกเลิก** เพื่อดูผล")
            st.caption("ระบบจะแสดง % ความเสี่ยง พร้อมข้อเสนอที่ฝ่ายต้อนรับ/สำรองห้องควรทำ")

with tab_model:
    st.markdown("#### สรุปโมเดลสุดท้ายของทีม")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ชนิดโมเดล", "Random Forest")
    m2.metric("จำนวนฟีเจอร์", f"{len(FEATURES)} (GA {meta.get('n_features_ga', 16)} + เพิ่ม 6)")
    m3.metric("ความแม่น (test)", f"{TEST_ACC:.4f}")
    m4.metric("ความแม่น (3-fold CV)", f"{CV_ACC:.4f}")

    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**ฟีเจอร์ที่ GA คัดเลือกไว้** (จาก 72 ฟีเจอร์ที่เตรียมไว้)")
            st.code("\n".join(FEATURES), language="text")
            st.caption(f"CV ของ DT เมื่อใช้ทุก 72 ฟีเจอร์ = {BASE_CV:.4f} · ใช้ชุดของ GA = 76.47% "
                       f"→ GA ช่วยให้ต้นไม้แม่นขึ้นชัดเจน")
    with right:
        with st.container(border=True):
            st.markdown("**ข้อมูลและข้อกำหนด**")
            st.write(f"- ข้อมูลเทรน: **{N_ROWS:,} แถว** (Hotel Booking Demand · Kaggle)")
            st.write(f"- อัตรายกเลิกจริงในข้อมูล: **{CANCEL_RATE:.1%}**")
            st.write("- ผลลัพธ์: `1` = ยกเลิก · `0` = มาเข้าพัก")
            st.write("- เกณฑ์ตัดสินใจ: **0.50**")
            st.write("- ตัดคอลัมน์ที่รั่วคำตอบ (`reservation_status`, `reservation_status_date`) ออกแล้ว")
        with st.container(border=True):
            st.markdown("**ข้อควรระวัง**")
            st.write(f"- recall ของคลาส “ยกเลิก” = 0.445 → โมเดลพลาดเคสที่ยกเลิกจริงบางส่วน")
            st.write("- ใช้เป็นตัวช่วยคัดกรอง/จัดลำดับการติดตาม ไม่ใช่คำตอบสุดท้าย")

with tab_help:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### วิธีใช้")
            st.markdown(
                "1. กรอก **ระยะเวลาจองล่วงหน้า** และ **วันที่เข้าพัก**\n"
                "2. ระบุ **เงินมัดจำ / ประเภทลูกค้า / ช่องทางตลาด / เดือน / ช่องทางการจอง**\n"
                "3. เลือก **ประเภทห้องที่จอง** และ **ห้องที่ได้จริง** (กรณีเปลี่ยนห้อง)\n"
                "4. กด **ประเมินความเสี่ยงยกเลิก** → ดู % ความเสี่ยงและข้อเสนอที่ควรทำ"
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### ข้อจำกัดของโมเดล")
            st.markdown(
                "1. โมเดลถูกคัดฟีเจอร์ด้วย GA เพื่อลดภาระการเก็บข้อมูล → แลกความแม่นบางส่วน "
                "เมื่อเทียบกับโมเดลที่ใช้ทั้ง 72 ฟีเจอร์\n"
                "2. ข้อมูลเป็นชุดปี 2015–2017 ของโรงแรม 2 แห่ง (City/Resort) — โรงแรมอื่นควรทดสอบใหม่\n"
                "3. คลาสไม่สมดุล (ยกเลิก 27.5%) → ควรดู precision/recall ประกอบ ไม่ดูแค่ accuracy\n"
                "4. ควรใช้ร่วมกับดุลยพินิจของทีมสำรองห้องพัก"
            )

st.markdown(f"""
<div class="footer">
  จัดทำโดย <b>ทีม PGD (PlerngDoiGun) · 032-034-038</b> — {' · '.join(f'{c} {n}' for c, n in TEAM)}<br>
  วิชา DT36822N ปัญญาประดิษฐ์เพื่อธุรกิจดิจิทัล · ใบงานสัปดาห์ที่ 12 (Streamlit) · ข้อมูล: Hotel Booking Demand (Kaggle)
</div>
""", unsafe_allow_html=True)
