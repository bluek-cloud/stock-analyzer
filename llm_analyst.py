import os
import json
import requests
from rag_retriever import retrieve_relevant_knowledge, format_knowledge_for_prompt
from backtest_engine import match_current_setup, format_stats_for_llm

# Google Gemini 공식 무료 모델
DEFAULT_MODEL = "gemini-1.5-flash"

def get_gemini_api_key(passed_key=None):
    """우선순위에 따라 Gemini API Key를 탐색"""
    if passed_key and passed_key.strip():
        return passed_key.strip()
    
    # 1. 환경 변수
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 2. 로컬 .env 파일 체크
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('GEMINI_API_KEY='):
                        return line.strip().split('=', 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass

    # 3. Streamlit secrets 체크
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        pass

    return None


_WORKING_MODEL = None

EXCLUDED_KEYWORDS = ["tts", "audio", "embedding", "imagen", "realtime", "aqa"]

def get_available_gemini_models(api_key):
    """Google API로부터 현재 키에서 generateContent를 지원하는 순수 텍스트 생성 모델 목록을 동적 조회"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "").lower()
                # 오디오, TTS, 임베딩 전용 모델 제외
                if any(ex in name for ex in EXCLUDED_KEYWORDS):
                    continue
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    orig_name = m.get("name", "")
                    if orig_name:
                        models.append(orig_name)
            return models
    except Exception:
        pass
    return []


def call_gemini_api(prompt, api_key):
    """
    Google Gemini REST API 스마트 호출:
    1. 계정에서 지원하는 순수 텍스트 생성 모델 목록을 동적으로 탐색
    2. 404/400(모달리티 불일치 등) 발생 시 대체 텍스트 모델 후보군으로 자동 재시도
    """
    global _WORKING_MODEL

    candidate_models = []
    if _WORKING_MODEL:
        candidate_models.append(_WORKING_MODEL)

    # 계정 지원 모델 목록 동적 조회 (TTS 등 제외된 텍스트 모델만)
    discovered = get_available_gemini_models(api_key)
    if discovered:
        # 일반 flash 모델 최우선 정렬
        flash_models = [m for m in discovered if "flash" in m.lower() and not any(ex in m.lower() for ex in EXCLUDED_KEYWORDS)]
        pro_models = [m for m in discovered if "pro" in m.lower() and not any(ex in m.lower() for ex in EXCLUDED_KEYWORDS)]
        other_models = [m for m in discovered if m not in flash_models and m not in pro_models]
        for m in flash_models + pro_models + other_models:
            if m not in candidate_models:
                candidate_models.append(m)

    # 기본 하드코딩 폴백 후보군 (오디오/TTS가 아닌 순수 텍스트 모델)
    default_fallbacks = [
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-pro",
        "models/gemini-pro"
    ]
    for m in default_fallbacks:
        if m not in candidate_models:
            candidate_models.append(m)

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "topP": 0.8,
            "maxOutputTokens": 2048
        }
    }

    last_error = ""
    for model_path in candidate_models:
        clean_model = model_path if model_path.startswith("models/") else f"models/{model_path}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{clean_model}:generateContent?key={api_key}"

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=25)
            if response.status_code == 200:
                _WORKING_MODEL = clean_model
                res_json = response.json()
                try:
                    return res_json["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    return "⚠️ AI 응답 파싱 중 문제가 발생했습니다."
            elif response.status_code == 404:
                last_error = f"404 ({clean_model} 미지원)"
                continue  # 다음 모델로 자동 전환
            elif response.status_code == 400:
                err_text = response.text.lower()
                # 모달리티 불일치(TTS 등) 또는 모델 미지원 오류인 경우 다음 모델로 자동 재시도
                if "modalities" in err_text or "not supported" in err_text or "not found" in err_text:
                    last_error = f"400 ({clean_model} 텍스트 미지원)"
                    continue
                return f"⚠️ API 요청 오류 (잘못된 API Key 또는 파라미터): {response.text}"
            elif response.status_code == 429:
                return "⚠️ 무료 API 호출 분당 한도(15 RPM)를 초과했습니다. 잠시 후(약 1분 뒤) 다시 시도해 주세요."
            else:
                last_error = f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            last_error = str(e)

    return f"⚠️ Gemini API 호출 실패: 사용 가능한 텍스트 모델을 찾을 수 없습니다. (원인: {last_error})"




def generate_rag_analyst_report(stock_info, market_context, api_key=None):
    """
    RAG 지식 검색과 Google Gemini를 결합하여
    전문가 수준의 주식 심층 진단 리포트를 생성
    """
    actual_key = get_gemini_api_key(api_key)
    if not actual_key:
        return (
            "🔑 **Gemini 무료 API Key가 설정되지 않았습니다.**\n\n"
            "사이드바의 [⚙️ Gemini API 설정] 입력창에 무료 API Key를 입력하시면, "
            "월가 수석 애널리스트 수준의 **RAG 기반 심층 진단 리포트**를 즉시 열람하실 수 있습니다.\n\n"
            "👉 [Google AI Studio에서 10초 만에 무료 API Key 발급받기](https://aistudio.google.com/app/apikey)"
        )

    # 1. RAG: 현재 종목 상황에 가장 부합하는 전문 트레이딩 지식 추출
    relevant_chunks = retrieve_relevant_knowledge(market_context, top_k=3)
    knowledge_text = format_knowledge_for_prompt(relevant_chunks)

    # 2. 역사적 백테스트 통계 추출
    _, matched_stats = match_current_setup(market_context)
    backtest_text = format_stats_for_llm(matched_stats)

    # 3. 프롬프트 구성
    prompt = f"""
당신은 20년 경력의 월스트리트 헤지펀드 수석 기술적 분석가(Chief Technical Analyst)입니다.
아래 제공된 **[종목 정량 지표 데이터]**, 검증된 **[전문 트레이딩 원칙 및 지식 베이스]**, 그리고 **[역사적 백테스트 통계 데이터]**를 결합하여, 실전 투자자를 위한 심층 트레이딩 진단 리포트를 한국어로 명확하고 전문성 있게 작성하세요.

---
### 1. 종목 정량 분석 데이터:
- 종목명/코드: {stock_info.get('name', '미상')} ({stock_info.get('code', '-')})
- 현재가: {stock_info.get('current_price', '-')} {stock_info.get('currency', '원')}
- 퀀트 종합 스코어: {stock_info.get('quant_score', 0)} / 100점
- 확정된 시장 국면(Regime): {market_context.get('regime', '횡보')}
- 1차 지지선: {stock_info.get('support', 0):,} {stock_info.get('currency', '원')}
- 1차 저항선: {('신고가(저항 없음)' if stock_info.get('resistance', 0) == 0 else f"{stock_info.get('resistance', 0):,} {stock_info.get('currency', '원')}")}
- 포착된 캔들 패턴: {', '.join(market_context.get('patterns', [])) if market_context.get('patterns') else '특이 패턴 없음'}
- 주요 보조지표 상태:
  • RSI(14): {stock_info.get('rsi', 0):.1f}
  • MACD 오실레이터: {stock_info.get('macd_diff', 0):.2f}
  • 상대 거래량(Vol Ratio): {stock_info.get('vol_ratio', 100):.0f}%
  • ATR(예상 변동폭): {stock_info.get('atr', 0):,.0f} {stock_info.get('currency', '원')}
  • 상승 다이버전스 감지: {'예 (포착됨)' if market_context.get('bullish_div') else '아니오'}
  • 룰 엔진 1차 제안 포지션: {stock_info.get('rule_position', '-')}

---
### 2. 검색 증강(RAG) 주입된 전문 트레이딩 지식 베이스:
{knowledge_text}

---
### 3. 역사적 백테스트 통계 및 기대 승률 (Statistical Edge):
{backtest_text}

---
### 4. 작성 가이드라인 (반드시 준수):
1. **뜬구름 잡는 일반론이나 뻔한 교과서 설명은 절대 금지합니다.** (예: "주식은 변동성이 있으니 주의하세요" 같은 상투적 조언 배제)
2. 반드시 상기 **[전문 트레이딩 지식 베이스]**에 명시된 원칙(캔들 꼬리 역학, 볼린저 스퀴즈/페이크, 다이버전스 신뢰도, 손익비 1:2 원칙 등)을 바탕으로 현재 주가 위치와 거래량을 직접 대조하여 해석하십시오.
3. 상기 **[역사적 백테스트 통계]**(승률, 손익비, 평균 기대수익률)를 리포트에 직접 인용하여, 투자자에게 통계적 우위(Statistical Edge)를 근거로 진입 타당성을 제시하십시오.
4. 다음 4개 항목을 갖춘 구조적 마크다운 리포트로 작성하십시오:
   - **🏛️ [거시 수급 및 시장 국면 심층 판정]**: 현재 국면과 세력 수급의 질적 분석
   - **🔬 [RAG 전문 지식 기반 기술적 분석]**: 캔들/패턴/지표와 전문 지식 원칙의 매칭 검증
   - **🎯 [실전 트레이딩 시나리오 및 가격 레벨]**:
     • 권장 진입 타점
     • 1차 / 2차 목표가
     • **분석 무효화 손절선(Invalidation Level)**: 정확한 수치 명시
     • **기대 손익비(Risk/Reward Ratio) 및 역사적 승률** 평가
   - **💡 [수석 애널리스트 최종 결론]**: 1~2문장의 명료한 핵심 요약
"""

    # 4. Gemini API 호출
    return call_gemini_api(prompt, actual_key)

