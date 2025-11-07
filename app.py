import streamlit as st
from google import genai
from google.generativeai.errors import ResourceExhaustedError, APIError
import time
import uuid

# --- 1. 환경 설정 및 상수 정의 ---
# Streamlit 앱 설정
st.set_page_config(
    page_title="🧠 멘탈 헬스 코치: 편안함",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 모델 옵션 정의
MODEL_OPTIONS = [
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-pro-preview-09-2025",
]
DEFAULT_MODEL = "gemini-2.5-flash-preview-09-2025"

# API 호출 재시도 설정 (429 Rate Limit 대비)
MAX_RETRIES = 5
INITIAL_DELAY = 1  # 초 (지수 백오프 시작 값)

# 고유 세션 ID 생성
SESSION_ID = str(uuid.uuid4())[:8]

# --- 2. 시스템 프롬프트 (요청 스펙 반영) ---
SYSTEM_PROMPT = """
당신은 '편안함(Pyeonan-Ham)'이라는 이름의 전문적인 멘탈 헬스 코치입니다.
당신의 주된 임무는 사용자의 감정을 경청하고 스트레스 관리를 돕는 것입니다. 당신은 심리학 전문가가 아니며, 의료적 진단이나 약물 조언을 제공하지 않습니다.

[상담 스타일 및 대화 원칙]
1.  **역할:** '편안함(Pyeonan-Ham)' 코치로서, 사용자의 감정을 경청하고 스트레스 관리를 돕는 역할을 수행할 것. **절대 전문 의료 진단이나 약물 조언은 제공하지 않아야 함.**
2.  **톤 앤 매너:** 항상 **차분하고 따뜻하며 희망을 주는 지지적인 톤**을 유지하며, 사용자 감정에 대해 절대 판단하지 않고 공감할 것. (예: "그렇게 느끼시는 것이 당연합니다", "힘든 시간을 보내고 계시는군요.")
3.  **상담 기법:** 인지 행동 치료(CBT) 기본 원칙에 따라, 사용자가 부정적 사고를 표현하면, 그 생각의 **논리적 근거를 스스로 질문하도록 유도**하는 방식으로 대화해야 함. (예: "그 생각이 사실이라는 증거는 무엇인가요?", "다른 관점에서 볼 여지는 없을까요?")
4.  **활동 제안:** 사용자의 기분 개선을 위해 실천 가능한 **스트레스 해소 활동(심호흡, 5분 명상, 산책 등)**을 제안해야 함.
5.  **면책 조항:** 대화 중 필요하다고 판단될 때, 아래 면책 문구를 반드시 포함하여 전문 의료인의 필요성을 안내해야 함.
    "저는 전문 의료인이 아닙니다. 위급한 상황이거나 지속적인 정신 건강의 어려움을 겪는다면 반드시 전문가(정신과 의사, 임상 심리사 등)를 찾아주세요."

[챗봇 시작 문구]
"안녕하세요. 저는 당신의 이야기를 들어줄 준비가 된 코치, 편안함입니다. 오늘은 어떤 감정을 느끼고 계신가요? 천천히 말씀해 주세요."
"""

# --- 3. 세션 상태 초기화 함수 ---
def initialize_session_state():
    """Streamlit 세션 상태를 초기화합니다."""
    if 'client' not in st.session_state:
        st.session_state.client = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'model_name' not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL
    if 'chat_initialized' not in st.session_state:
        st.session_state.chat_initialized = False

# --- 4. API 키 처리 및 클라이언트 설정 ---
def get_api_key():
    """API 키를 secret 또는 사용자 입력으로부터 가져옵니다."""
    try:
        # st.secrets에서 키를 가져오려고 시도
        return st.secrets['GEMINI_API_KEY']
    except (AttributeError, KeyError):
        # secrets이 없으면 사이드바에서 사용자 입력 UI 표시
        with st.sidebar:
            st.warning("⚠️ Streamlit `secrets.toml`에 API 키가 없습니다.")
            return st.text_input("Gemini API 키를 임시로 입력하세요.", type="password")

def setup_client(api_key, model_name):
    """API 클라이언트를 초기화하고 챗 세션을 시작합니다."""
    # 이미 클라이언트가 설정되어 있고 모델이 동일하면 재설정하지 않음
    if st.session_state.client and st.session_state.model_name == model_name and st.session_state.chat_initialized:
        return

    try:
        st.session_state.client = genai.Client(api_key=api_key)
        st.session_state.model_name = model_name

        # 새로운 채팅 세션 시작 (System Prompt 적용)
        st.session_state.chat = st.session_state.client.chats.create(
            model=model_name,
            system_instruction=SYSTEM_PROMPT
        )
        
        # history 초기화 및 시작 메시지 추가
        st.session_state.history = []
        initial_message = "안녕하세요. 저는 당신의 이야기를 들어줄 준비가 된 코치, 편안함입니다. 오늘은 어떤 감정을 느끼고 계신가요? 천천히 말씀해 주세요."
        st.session_state.history.append({"role": "model", "parts": [{"text": initial_message}]})
        st.session_state.chat_initialized = True

    except Exception as e:
        st.error(f"API 클라이언트 설정 오류가 발생했습니다: {e}")
        st.session_state.chat_initialized = False
        st.stop()


# --- 5. 대화 기록 관리 및 모델 호출 로직 (재시도 포함) ---
def generate_response(prompt):
    """프롬프트를 API로 보내고 429 에러 발생 시 재시도 로직을 적용합니다."""
    if not st.session_state.client or not st.session_state.chat_initialized:
        return "죄송합니다. 챗봇 초기화가 완료되지 않았습니다. API 키를 확인하거나 초기화 버튼을 눌러주세요."

    # API 호출 및 재시도 로직
    for i in range(MAX_RETRIES):
        try:
            # 429 재시도 시, Gemini Chat History API의 제약 상, 
            # 최근 6턴(User 3, Model 3)의 대화 기록만 유지합니다.
            
            # 현재 history를 genai.types.Content 객체 리스트로 변환
            context_history = []
            
            # history는 {role, parts: [{text}]} 형태입니다. 이를 Content 형태로 변환합니다.
            # 초기 메시지(1턴)를 제외하고 최근 6턴(3쌍)만 유지합니다.
            recent_history = st.session_state.history[1:]
            
            # 최대 6턴만 사용 (3쌍: 3 user, 3 model)
            if len(recent_history) > 6:
                recent_history = recent_history[-6:] 

            for message in recent_history:
                context_history.append(genai.types.Content(
                    role=message["role"], 
                    parts=[genai.types.Part.from_text(message["parts"][0]["text"])]
                ))
            
            # 현재 사용자 프롬프트를 추가
            context_history.append(genai.types.Content(
                role="user",
                parts=[genai.types.Part.from_text(prompt)]
            ))
            
            # API 호출 (send_message 대신 generate_content 사용)
            response = st.session_state.client.models.generate_content(
                model=st.session_state.model_name,
                contents=context_history,
                system_instruction=SYSTEM_PROMPT
            )
            
            return response.text

        except ResourceExhaustedError:
            delay = INITIAL_DELAY * (2 ** i) # 지수 백오프
            st.warning(f"⚠️ API 요청 제한(429)으로 인해 {delay:.1f}초 후 재시도합니다. (시도: {i + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            
            if i == MAX_RETRIES - 1:
                st.error("죄송합니다. 현재 API 요청 제한을 초과했습니다. 잠시 후 다시 시도해 주세요.")
                return None
            
        except APIError as e:
            st.error(f"API 오류 발생: {e}")
            return "죄송합니다. API 처리 중 오류가 발생했습니다."
        
        except Exception as e:
            st.error(f"예상치 못한 오류 발생: {e}")
            return "죄송합니다. 처리 중 오류가 발생했습니다."

    return None

# --- 6. 유틸리티 함수 ---
def reset_chat():
    """대화 기록을 초기화하고 클라이언트 재설정을 위해 상태를 리셋합니다."""
    st.session_state.history = []
    st.session_state.client = None
    st.session_state.chat_initialized = False
    st.experimental_rerun()

# --- 7. Streamlit UI 구성 ---
st.title("🧠 멘탈 헬스 코치: 편안함")
st.markdown("### 당신의 이야기를 판단 없이 들어주는 코치")

# API 키 및 클라이언트 설정
api_key = get_api_key()

with st.sidebar:
    st.header("⚙️ 설정 및 상태")
    
    selected_model = st.selectbox(
        "사용할 Gemini 모델 선택",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(DEFAULT_MODEL)
    )

    if api_key:
        setup_client(api_key, selected_model)
        if st.session_state.chat_initialized:
             st.success("API 키 확인 및 챗봇 준비 완료")
        
        st.info(f"**현재 모델:** `{selected_model.split('/')[0]}`\n\n**세션 ID:** `{SESSION_ID}`")

        # 대화 초기화 버튼
        if st.button("🔄 대화 초기화 (새 대화 시작)", use_container_width=True):
            reset_chat()
    else:
        st.warning("API 키를 입력하거나 설정해 주세요.")

initialize_session_state()

# 대화 기록 출력
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["parts"][0]["text"])

# 사용자 입력 처리
if st.session_state.chat_initialized and st.session_state.client:
    if prompt := st.chat_input("당신의 감정을 이야기해주세요..."):
        # 사용자 메시지 화면 출력 및 히스토리 기록
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 모델 응답 생성
        with st.spinner("코치가 당신의 이야기에 귀 기울이고 있습니다..."):
            ai_response = generate_response(prompt)

        # 모델 응답 화면 출력
        if ai_response:
            with st.chat_message("model"):
                st.markdown(ai_response)
            # Streamlit 출력을 위한 history 업데이트
            st.session_state.history.append({"role": "user", "parts": [{"text": prompt}]})
            st.session_state.history.append({"role": "model", "parts": [{"text": ai_response}]})