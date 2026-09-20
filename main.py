import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 페이지 설정 (탭 제목 및 아이콘)
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 2. 제목 화면
st.title("🧠 뇌졸중 예측 실습실")
st.caption("공공 데이터를 활용한 뇌졸중 예측 데이터 분석 및 모델링 실습 공간입니다.")
st.markdown("---")

# 3. 데이터 로드 (캐싱 적용)
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data(DATA_URL)

# 4. 상단 지표 (Metric Cards 4개)
total_count = len(df)
total_cols = len(df.columns)
stroke_count = int(df['stroke'].sum())
stroke_ratio = (stroke_count / total_count) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 사람 수", f"{total_count:,} 명")

with col2:
    st.metric("열 개수", f"{total_cols} 개")

with col3:
    st.metric("뇌졸중 환자 수 (stroke=1)", f"{stroke_count:,} 명")

with col4:
    st.metric("뇌졸중 비율", f"{stroke_ratio:.2f} %")

st.markdown("---")

# 5. 열 정보 안내 표 (우리말 뜻 직접 작성 가능한 에디터)
st.subheader("📋 데이터 열(Column) 정보")
st.write("교재를 참고하여 **'우리말 뜻'** 칸을 직접 채워보세요.")

# 변수 정보 DataFrame 구성
null_counts = df.isnull().sum()
dtypes = df.dtypes

meta_data = {
    "열 이름": df.columns,
    "우리말 뜻": [""] * len(df.columns),  # 빈 값으로 비워둠
    "값의 종류 (데이터 타입)": [str(dtypes[col]) for col in df.columns],
    "빈 값 개수 (결측치)": [int(null_counts[col]) for col in df.columns]
}

meta_df = pd.DataFrame(meta_data)

# 사용자가 입력할 수 있는 Data Editor 제공
edited_meta_df = st.data_editor(
    meta_df,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed"
)

st.markdown("---")

# 6. 데이터 상위 5개 출력
st.subheader("🔍 데이터 미리보기 (처음 5행)")
st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# 7. 데이터 출처 작성란
st.subheader("📌 데이터 출처")
source_text = st.text_area(
    "교재에 적힌 데이터 출처 내용을 아래에 입력하세요:",
    placeholder="예: Kaggle Stroke Prediction Dataset / 출처 문구를 여기에 적으세요.",
    height=100
)

if source_text:
    st.info(f"**입력한 출처:** {source_text}")
