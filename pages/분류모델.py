import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree

# 1. 페이지 설정
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 머신러닝 분류 모델")
st.caption("여러 속성을 활용해 뇌졸중 유무(stroke=1: 뇌졸중, stroke=0: 아님)를 예측하는 모델을 학습하고 평가합니다.")
st.markdown("---")

# 2. 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    return df

df_raw = load_data(DATA_URL)

# 3. 속성 매핑 정보 (열 이름 <-> 우리말 이름)
FEATURE_MAP = {
    'age': '나이',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량지수',
    'hypertension': '고혈압',
    'heart_disease': '심장병'
}
REVERSE_FEATURE_MAP = {v: k for k, v in FEATURE_MAP.items()}

# 4. 입력 속성 선택 (화면 지정)
st.subheader("1. 입력 속성 선택")
selected_korean_features = st.multiselect(
    "모델 학습 및 예측에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=list(FEATURE_MAP.values()),
    default=['나이', '평균 혈당', '고혈압', '심장병'] # bmi를 제외한 4개가 기본값
)

# 속성이 2개 미만일 경우 안내문 표시 후 멈춤
if len(selected_korean_features) < 2:
    st.warning("⚠️ 최소 2개 이상의 속성을 고르셔야 모델 학습 및 산점도 시각화가 가능합니다.")
    st.stop()

selected_cols = [REVERSE_FEATURE_MAP[name] for name in selected_korean_features]

# 5. 데이터 분할 및 결측치 처리 규칙 적용
# 규칙: 번호 순으로 정렬 -> 10명씩 묶고 앞 3명을 테스트용으로 고정 (1,533명)
df = df_raw.copy()
if 'id' in df.columns:
    df = df.sort_values(by='id').reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

test_indices = []
train_indices = []

for i in range(len(df)):
    if (i % 10) < 3:
        test_indices.append(i)
    else:
        train_indices.append(i)

train_df = df.iloc[train_indices].copy()
test_df = df.iloc[test_indices].copy()

# bmi 결측치 처리 (선택된 경우에만 훈련 데이터의 중앙값으로 채움)
if 'bmi' in selected_cols:
    bmi_median = train_df['bmi'].median()
    train_df['bmi'] = train_df['bmi'].fillna(bmi_median)
    test_df['bmi'] = test_df['bmi'].fillna(bmi_median)

X_train = train_df[selected_cols]
y_train = train_df['stroke']
X_test = test_df[selected_cols]
y_test = test_df['stroke']

# 크기 맞추기 (StandardScaler) - 훈련용으로만 fit
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 학습
# A. 다수결(최빈값) 기준선 모델
majority_class = y_train.mode()[0]
baseline_train_acc = (y_train == majority_class).mean()
baseline_test_acc = (y_test == majority_class).mean()

# B. 로지스틱 회귀 모델
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_train_acc = log_reg.score(X_train_scaled, y_train)
log_test_acc = log_reg.score(X_test_scaled, y_test)

# C. 의사결정트리 모델 (질문 3번 제한, min_samples_leaf=5, 난수 고정)
dt_cls = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_cls.fit(X_train, y_train)  # 트리 해석 및 시각화를 위해 원본 단위 학습
dt_train_acc = dt_cls.score(X_train, y_train)
dt_test_acc = dt_cls.score(X_test, y_test)

st.markdown("---")

# 7. 모델 성능 평가 카드 (3개)
st.subheader("2. 모델 성능 비교 (정확도)")

card_col1, card_col2, card_col3 = st.columns(3)

with card_col1:
    st.markdown("### 기준선 모델")
    st.caption("입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델")
    st.metric("테스트 정확도", f"{baseline_test_acc * 100:.2f}%")
    st.caption(f"훈련 데이터: {baseline_train_acc * 100:.2f}% | 테스트 데이터: {baseline_test_acc * 100:.2f}%")

with card_col2:
    st.markdown("### 로지스틱 회귀")
    st.caption("확률로 답하는 모델")
    st.metric("테스트 정확도", f"{log_test_acc * 100:.2f}%")
    st.caption(f"훈련 데이터: {log_train_acc * 100:.2f}% | 테스트 데이터: {log_test_acc * 100:.2f}%")

with card_col3:
    st.markdown("### 의사결정트리")
    st.caption("질문으로 답하는 모델")
    st.metric("테스트 정확도", f"{dt_test_acc * 100:.2f}%")
    st.caption(f"훈련 데이터: {dt_train_acc * 100:.2f}% | 테스트 데이터: {dt_test_acc * 100:.2f}%")

st.markdown("---")

# 8. 2차원 시각화 (산점도, 결정 경계, 트리 분할 영역)
st.subheader("3. 2차원 시각화 및 결정 경계")

col_x_ui, col_y_ui = st.columns(2)
with col_x_ui:
    x_korean = st.selectbox("가로축(X축) 선택:", selected_korean_features, index=0)
with col_y_ui:
    # 두 번째 선택지는 기본적으로 다른 속성을 선택
    default_y_idx = 1 if len(selected_korean_features) > 1 else 0
    y_korean = st.selectbox("세로축(Y축) 선택:", selected_korean_features, index=default_y_idx)

if x_korean == y_korean:
    st.info("💡 가로축과 세로축에 서로 다른 속성을 선택하셔야 분할 영역 시각화가 올바르게 작동합니다.")

x_col = REVERSE_FEATURE_MAP[x_korean]
y_col = REVERSE_FEATURE_MAP[y_korean]

# 테스트 데이터 기준축 설정 및 나머지 속성 중앙값 가공
other_cols = [c for c in selected_cols if c not in [x_col, y_col]]
fixed_values_text = []

for c in other_cols:
    med_val = test_df[c].median()
    fixed_values_text.append(f"**{FEATURE_MAP[c]}**: 테스트 데이터 중앙값({med_val:.2f})")

if fixed_values_text:
    st.write("📌 **고정된 속성 값:** " + ", ".join(fixed_values_text))

# 격자 생성
x_min, x_max = test_df[x_col].min() - 1, test_df[x_col].max() + 1
y_min, y_max = test_df[y_col].min() - 1, test_df[y_col].max() + 1

# 스케일 조정된 그리드용
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

# 그리드 포인트 데이터프레임 생성
grid_data = pd.DataFrame()
for c in selected_cols:
    if c == x_col:
        grid_data[c] = xx.ravel()
    elif c == y_col:
        grid_data[c] = yy.ravel()
    else:
        grid_data[c] = test_df[c].median()

# 의사결정트리 예측 영역 계산
dt_grid_pred = dt_cls.predict(grid_data[selected_cols]).reshape(xx.shape)

# Plotly 바탕 그림 생성
fig = go.Figure()

# A. 의사결정트리 경계 옅은 배경 색칠 (Contour)
fig.add_trace(go.Contour(
    x=np.linspace(x_min, x_max, 200),
    y=np.linspace(y_min, y_max, 200),
    z=dt_grid_pred,
    showscale=False,
    opacity=0.25,
    colorscale=[[0, '#1f77b4'], [1, '#ff7f0e']], # 정상: 푸른색계열, 뇌졸중: 주황색계열
    hoverinfo='skip'
))

# B. 로지스틱 회귀 0.5 경계선 (Logistic Regression Decision Boundary)
# w1*x1_scaled + w2*x2_scaled + sum(w_i * x_i_scaled) + b = 0
grid_scaled = scaler.transform(grid_data[selected_cols])
log_probs = log_reg.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

# 경계선 추가 (확률 0.5 위치)
fig.add_trace(go.Contour(
    x=np.linspace(x_min, x_max, 200),
    y=np.linspace(y_min, y_max, 200),
    z=log_probs,
    contours_start=0.5,
    contours_end=0.5,
    contours_coloring='lines',
    line_width=3,
    line_color='black',
    name='로지스틱 회귀 경계선 (p=0.5)',
    showscale=False
))

# C. 테스트 데이터 산점도 점 찍기
test_df_plot = test_df.copy()
test_df_plot['stroke_label'] = test_df_plot['stroke'].map({0: '정상 (0)', 1: '뇌졸중 (1)'})

for stroke_val, color, name in [(0, '#1f77b4', '실제 정상 (0)'), (1, '#d62728', '실제 뇌졸중 (1)')]:
    sub_df = test_df_plot[test_df_plot['stroke'] == stroke_val]
    fig.add_trace(go.Scatter(
        x=sub_df[x_col],
        y=sub_df[y_col],
        mode='markers',
        name=name,
        marker=dict(color=color, size=7, opacity=0.8)
    ))

fig.update_layout(
    title=f"{x_korean} vs {y_korean} (결정 경계 및 테스트 데이터 시각화)",
    xaxis_title=x_korean,
    yaxis_title=y_korean,
    xaxis=dict(range=[x_min, x_max]),
    yaxis=dict(range=[y_min, y_max])
)

st.plotly_chart(fig, use_container_width=True)

# 경계선이 그려졌는지 확인하는 문구 출력
if (log_probs.min() > 0.5) or (log_probs.max() < 0.5):
    st.info("📢 로지스틱 회귀의 분류 경계선(0.5)이 해당 속성 범위 밖에 위치하여 그림에 선이 나타나지 않습니다.")
else:
    st.caption("검은색 실선은 로지스틱 회귀 모델이 뇌졸중 확률을 0.5로 가르는 경계선입니다.")

st.markdown("---")

# 9. 의사결정트리 구조 Graphviz 시각화 및 해석
st.subheader("4. 의사결정트리 질문 가지 그림")

def export_dot_custom(model, feature_names):
    tree = model.tree_
    
    dot = ['digraph Tree {']
    dot.append('node [shape=box, style="filled, rounded", color="black", fontname="NanumGothic, Malgun Gothic, sans-serif"] ;')
    dot.append('edge [fontname="NanumGothic, Malgun Gothic, sans-serif"] ;')
    
    leaf_count = 0
    neg_leaf_count = 0
    used_features = set()

    def recurse(node, depth):
        nonlocal leaf_count, neg_leaf_count
        
        # 훈련 데이터 인원 수 및 클래스별 수치
        n_samples = tree.n_node_samples[node]
        val = tree.value[node][0]
        n_stroke = int(val[1]) if len(val) > 1 else 0
        ratio = (n_stroke / n_samples) * 100 if n_samples > 0 else 0.0

        if tree.feature[node] != _tree.TREE_UNDEFINED:
            feat_idx = tree.feature[node]
            feat_name = feature_names[feat_idx]
            used_features.add(feat_name)
            threshold = tree.threshold[node]
            
            label_text = f"{feat_name} <= {threshold:.2f}\\n사람 수: {n_samples}명\\n뇌졸중: {n_stroke}명 ({ratio:.1f}%)"
            dot.append(f'{node} [label="{label_text}", fillcolor="#ffffff"] ;')
            
            left_child = tree.children_left[node]
            right_child = tree.children_right[node]
            
            dot.append(f'{node} -> {left_child} [label="예"] ;')
            dot.append(f'{node} -> {right_child} [label="아니요"] ;')
            
            recurse(left_child, depth + 1)
            recurse(right_child, depth + 1)
        else:
            leaf_count += 1
            pred_class = np.argmax(val)
            if pred_class == 0:
                neg_leaf_count += 1
                fillcolor = "#e8f4f8"  # 정상(아님) - 연푸른색
                pred_label = "답: 정상(아님)"
            else:
                fillcolor = "#ffe6e6"  # 뇌졸중 - 연분홍색
                pred_label = "답: 뇌졸중"
                
            label_text = f"{pred_label}\\n사람 수: {n_samples}명\\n뇌졸중: {n_stroke}명 ({ratio:.1f}%)"
            dot.append(f'{node} [label="{label_text}", fillcolor="{fillcolor}"] ;')

    recurse(0, 0)
    dot.append('}')
    return "\n".join(dot), leaf_count, neg_leaf_count, list(used_features)

korean_feature_names = [FEATURE_MAP[c] for c in selected_cols]
dot_string, leaf_count, neg_leaf_count, used_features_list = export_dot_custom(dt_cls, korean_feature_names)

st.graphviz_chart(dot_string)

# 요약 설명 한 줄씩 작성
st.markdown(f"- 답을 내는 마디(최종 잎)는 모두 **{leaf_count}칸**이고, 그중 **{neg_leaf_count}칸**이 **'아님(정상)'**이라고 답합니다.")
st.markdown(f"- 고른 속성 가운데 이 나무가 실제로 물은 것은 **{', '.join(used_features_list)}** 입니다.")
