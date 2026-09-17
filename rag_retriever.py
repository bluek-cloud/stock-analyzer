import os
import re

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge')
_CACHED_CHUNKS = None

def load_knowledge_chunks(force_reload=False):
    """knowledge/ 폴더의 마크다운 파일들을 섹션 단위로 청킹하여 메모리에 로드"""
    global _CACHED_CHUNKS
    if _CACHED_CHUNKS is not None and not force_reload:
        return _CACHED_CHUNKS

    chunks = []
    if not os.path.exists(KNOWLEDGE_DIR):
        _CACHED_CHUNKS = []
        return _CACHED_CHUNKS

    for fname in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not fname.endswith('.md'):
            continue
        fpath = os.path.join(KNOWLEDGE_DIR, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        # H1 제목 추출
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        doc_title = h1_match.group(1).strip() if h1_match else fname

        # H2 섹션 단위로 분할
        sections = re.split(r'(?m)^##\s+', content)
        for idx, sec in enumerate(sections):
            sec = sec.strip()
            if not sec:
                continue
            lines = sec.split('\n', 1)
            sec_title = lines[0].strip() if len(lines) > 0 else f"섹션 {idx}"
            body = lines[1].strip() if len(lines) > 1 else ""

            # H1 서두 부분인 경우
            if sec_title.startswith('#'):
                continue

            full_text = f"### {sec_title}\n{body}"
            chunks.append({
                'id': f"{fname}#{sec_title}",
                'doc_title': doc_title,
                'section_title': sec_title,
                'content': full_text,
                'search_text': f"{doc_title} {sec_title} {body}".lower()
            })

    _CACHED_CHUNKS = chunks
    return _CACHED_CHUNKS


def retrieve_relevant_knowledge(market_context, top_k=3):
    """
    현재 종목의 기술적 분석 상태(market_context)를 분석하여
    가장 관련성이 높은 전문 트레이딩 지식 청크들을 선별(RAG)
    """
    chunks = load_knowledge_chunks()
    if not chunks:
        return []

    # 1. 시장 상태 기반 가중 검색 키워드 추출
    query_weights = {}

    regime = market_context.get('regime', '')
    if "스퀴즈" in regime or "응축" in regime:
        query_weights['스퀴즈'] = 5.0
        query_weights['볼린저'] = 4.0
        query_weights['수축'] = 4.0
        query_weights['헤드페이크'] = 3.0
    elif "박스" in regime or "횡보" in regime:
        query_weights['박스'] = 5.0
        query_weights['지지'] = 4.0
        query_weights['저항'] = 4.0
        query_weights['스프링'] = 3.0
    elif "강세" in regime:
        query_weights['정배열'] = 5.0
        query_weights['추세'] = 4.0
        query_weights['밴드 워킹'] = 4.0
        query_weights['눌림목'] = 4.0
    elif "약세" in regime:
        query_weights['역배열'] = 5.0
        query_weights['하락'] = 4.0
        query_weights['손절'] = 3.0
    elif "폭발" in regime:
        query_weights['거래량'] = 5.0
        query_weights['돌파'] = 4.0
        query_weights['변동성'] = 4.0

    if market_context.get('bullish_div'):
        query_weights['다이버전스'] = 6.0
        query_weights['상승 다이버전스'] = 5.0
        query_weights['모멘텀'] = 4.0

    patterns = market_context.get('patterns', [])
    for p in patterns:
        if "망치" in p:
            query_weights['망치형'] = 5.0
            query_weights['아랫꼬리'] = 4.0
        elif "장악" in p:
            query_weights['장악형'] = 5.0
        elif "교수형" in p:
            query_weights['교수형'] = 5.0
            query_weights['윗꼬리'] = 4.0

    if market_context.get('is_falling_knife'):
        query_weights['손절'] = 6.0
        query_weights['무효화'] = 5.0
        query_weights['패닉셀'] = 5.0

    # 기본 리스크 관리 및 200일선 키워드 상시 부여
    query_weights['손익비'] = 2.0
    query_weights['200일'] = 2.0

    # 2. 각 청크별 관련도 점수 산출
    scored_chunks = []
    for chunk in chunks:
        score = 0.0
        text = chunk['search_text']
        for kw, weight in query_weights.items():
            if kw.lower() in text:
                score += weight
        scored_chunks.append((score, chunk))

    # 3. 상위 점수 청크 정렬
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    results = [item[1] for item in scored_chunks[:top_k]]
    return results


def format_knowledge_for_prompt(knowledge_chunks):
    """검색된 지식 청크들을 LLM 프롬프트에 주입하기 알맞은 텍스트로 변환"""
    if not knowledge_chunks:
        return "참조할 특화 전문 지식이 없습니다."

    formatted = []
    for idx, chunk in enumerate(knowledge_chunks, 1):
        formatted.append(f"[전문 지식 #{idx}: {chunk['doc_title']} - {chunk['section_title']}]\n{chunk['content']}\n")
    return "\n".join(formatted)
