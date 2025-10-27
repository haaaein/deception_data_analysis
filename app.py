from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
from collections import defaultdict, Counter
import re
import itertools
import numpy as np
import glob

app = Flask(__name__)

# 파일 경로들
EXCEL_FILE = 'integrated_deceptive_taxonomy.xlsx'

# 모델별 JSON 파일 매핑
MODEL_FILES = {
    'gemini-2.5-pro': {
        'v1': 'gemini-2.5-pro/v1_deceptive_taxonomy.json',
        'v2': 'gemini-2.5-pro/v2_deceptive_taxonomy.json.json',
        'v3': 'gemini-2.5-pro/v3_deceptive_taxonomy.json.json',
        'v4': 'gemini-2.5-pro/v4_deceptive_taxonomy.json.json',
        'v5': 'gemini-2.5-pro/v5_deceptive_taxonomy.json.json'
    },
    'o4-mini': {
        'v1': 'o4-mini/final_deceptive_taxonomy_20250707_185258.json',
        'v2': 'o4-mini/final_deceptive_taxonomy_20250707_185859.json',
        'v3': 'o4-mini/final_deceptive_taxonomy_20250707_190326.json',
        'v4': 'o4-mini/final_deceptive_taxonomy_20250707_191026.json',
        'v5': 'o4-mini/final_deceptive_taxonomy_20250707_225857.json'
    },
    'deepseek-r1': {
        'v1': 'deepseek-r1/final_deceptive_taxonomy_20250707_234546.json',
        'v2': 'deepseek-r1/final_deceptive_taxonomy_20250707_235534.json',
        'v3': 'deepseek-r1/final_deceptive_taxonomy_20250708_001325.json',
        'v4': 'deepseek-r1/final_deceptive_taxonomy_20250708_002554.json',
        'v5': 'deepseek-r1/final_deceptive_taxonomy_deepseek-reasoner_20250708_072728.json'
    }
}

def normalize_strategy_name(strategy_name):
    """전략 이름을 정규화하여 대소문자/구두점 차이를 통일"""
    if not strategy_name:
        return ""
    
    # 소문자로 변환하고 특수문자 정리
    normalized = strategy_name.lower()
    
    # 여러 단어를 하나로 통합하는 매핑
    normalization_map = {
        # Appeal to X 계열
        'appeal to fear': 'appeal_to_fear',
        'fear appeal': 'appeal_to_fear',
        'appeal to emotion': 'appeal_to_emotion',
        'emotional appeal': 'appeal_to_emotion',
        'appeal to authority': 'appeal_to_authority',
        'appeal to pity': 'appeal_to_pity',
        'appeal to popularity': 'appeal_to_popularity',
        'appeal to tradition': 'appeal_to_tradition',
        'appeal to nature': 'appeal_to_nature',
        'appeal to consequences': 'appeal_to_consequences',
        'appeal to novelty': 'appeal_to_novelty',
        'appeal to ignorance': 'appeal_to_ignorance',
        'argument from ignorance': 'appeal_to_ignorance',
        
        # Cherry Picking 계열
        'cherry picking': 'cherry_picking',
        'cherry-picking': 'cherry_picking',
        'cherry‐picking': 'cherry_picking',
        
        # Slippery Slope
        'slippery slope': 'slippery_slope',
        
        # False Dilemma/Dichotomy
        'false dilemma': 'false_dilemma',
        'false dichotomy': 'false_dilemma',
        'either/or fallacy': 'false_dilemma',
        'either-or fallacy': 'false_dilemma',
        
        # Straw Man
        'straw man': 'straw_man',
        'strawman': 'straw_man',
        'straw man fallacy': 'straw_man',
        'strawman fallacy': 'straw_man',
        
        # Red Herring
        'red herring': 'red_herring',
        
        # Loaded Language
        'loaded language': 'loaded_language',
        
        # Hasty Generalization
        'hasty generalization': 'hasty_generalization',
        'overgeneralization': 'hasty_generalization',
        
        # False Cause
        'false cause': 'false_cause',
        
        # False Analogy
        'false analogy': 'false_analogy',
        'faulty analogy': 'false_analogy',
        
        # Bandwagon
        'bandwagon': 'bandwagon',
        'bandwagon appeal': 'bandwagon',
        'bandwagon effect': 'bandwagon',
        'bandwagon fallacy': 'bandwagon',
        
        # Card Stacking
        'card stacking': 'card_stacking',
        
        # Downplaying/Minimization
        'downplaying': 'downplaying',
        'minimization': 'downplaying',
        'minimization of risks': 'downplaying',
        'downplaying risks': 'downplaying',
        
        # Glittering Generalities
        'glittering generalities': 'glittering_generalities',
        'glittering generality': 'glittering_generalities',
        
        # Misleading Statistics
        'misleading statistics': 'misleading_statistics',
        'misleading statistic': 'misleading_statistics',
        'misleading use of statistics': 'misleading_statistics',
        
        # False Equivalence
        'false equivalence': 'false_equivalence',
        
        # Anecdotal Evidence
        'anecdotal evidence': 'anecdotal_evidence',
        
        # Begging the Question
        'begging the question': 'begging_the_question',
        
        # Weasel Words
        'weasel words': 'weasel_words',
        
        # Fear Mongering
        'fear mongering': 'fear_mongering',
        'fearmongering': 'fear_mongering',
        'fear-mongering': 'fear_mongering',
        
        # Vague Authority
        'vague authority': 'vague_authority',
        
        # Moving the Goalposts
        'moving the goalposts': 'moving_the_goalposts',
        
        # Poisoning the Well
        'poisoning the well': 'poisoning_the_well',
        
        # Thought-Terminating Cliché
        'thought-terminating cliché': 'thought_terminating_cliche',
        'thought-terminating cliche': 'thought_terminating_cliche',
        
        # Appeal to Common Sense
        'appeal to common sense': 'appeal_to_common_sense',
        
        # False Urgency
        'false urgency': 'false_urgency',
        
        # One-Sided Argument
        'one-sided argument': 'one_sided_argument',
        'one-sidedness': 'one_sided_argument',
        
        # Selective Evidence
        'selective evidence': 'selective_evidence',
    }
    
    # 정규화 매핑 적용
    if normalized in normalization_map:
        return normalization_map[normalized]
    
    # 매핑에 없으면 기본 정규화
    # 특수문자 제거하고 공백을 언더스코어로
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', '_', normalized.strip())
    
    return normalized

def load_excel_data():
    """Excel 파일을 로드합니다."""
    try:
        # Excel 파일 읽기
        df = pd.read_excel(EXCEL_FILE)
        return df
    except Exception as e:
        return None, str(e)

def load_json_data(model, version):
    """특정 모델의 특정 버전 JSON 파일을 로드합니다."""
    try:
        if model not in MODEL_FILES:
            return None, f"지원하지 않는 모델입니다: {model}"
        
        if version not in MODEL_FILES[model]:
            return None, f"지원하지 않는 버전입니다: {version}"
        
        file_path = MODEL_FILES[model][version]
        if not os.path.exists(file_path):
            return None, f"파일을 찾을 수 없습니다: {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data, None
    except Exception as e:
        return None, str(e)

def process_json_data(data, strategy_filter=None):
    """JSON 데이터를 처리하여 테이블 형태로 변환합니다."""
    try:
        processed_data = []
        
        for item in data:
            strategy_name = item.get('strategy_name', '')
            description = item.get('description', '')
            examples = item.get('examples', [])
            sub_strategies = item.get('sub_strategies', [])
            
            # "No Deceptive Strategy Detected" 전략 제거
            if strategy_name == 'No Deceptive Strategy Detected':
                continue
            
            # 전략 필터링
            if strategy_filter and strategy_name != strategy_filter:
                continue
            
            # examples를 번호와 함께 HTML 형태로 변환
            examples_html = ''
            if examples:
                examples_list = [f"{i+1}. {example}" for i, example in enumerate(examples)]
                examples_html = '<br>'.join(examples_list)
            
            # sub_strategies를 문자열로 변환
            sub_strategies_str = ', '.join(sub_strategies) if sub_strategies else ''
            
            processed_data.append({
                'strategy_name': strategy_name,
                'description': description,
                'examples': examples_html,
                'sub_strategies': sub_strategies_str,
                'sub_strategies_count': len(sub_strategies)
            })
        
        return processed_data, None
    except Exception as e:
        return None, str(e)

def get_strategy_counts(df, model_filter=None):
    """각 전략별 파일 개수를 반환합니다."""
    if 'strategy_name' not in df.columns:
        return {}
    
    # "No Deceptive Strategy Detected" 전략 제거
    df = df[df['strategy_name'] != 'No Deceptive Strategy Detected'].copy()
    
    # 모델 필터링
    if model_filter and 'model' in df.columns:
        df = df[df['model'] == model_filter].copy()
    
    # 각 전략별로 고유한 모델-버전 조합 개수를 계산
    strategy_counts = df.groupby('strategy_name')[['model', 'version']].apply(
        lambda x: x.drop_duplicates().shape[0], include_groups=False
    ).to_dict()
    
    return strategy_counts

def get_json_strategy_counts(data):
    """JSON 데이터에서 각 전략별 개수를 반환합니다."""
    strategy_counts = {}
    for item in data:
        strategy_name = item.get('strategy_name', '')
        if strategy_name and strategy_name != 'No Deceptive Strategy Detected':
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
    return strategy_counts

def process_excel_data(df, strategy_filter=None, model_filter=None):
    """통합 Excel 데이터를 처리합니다."""
    try:
        # "No Deceptive Strategy Detected" 전략 제거
        if 'strategy_name' in df.columns:
            df = df[df['strategy_name'] != 'No Deceptive Strategy Detected'].copy()
        
        # 전략 필터링
        if strategy_filter and 'strategy_name' in df.columns:
            df = df[df['strategy_name'] == strategy_filter].copy()
        
        # 모델 필터링
        if model_filter and 'model' in df.columns:
            df = df[df['model'] == model_filter].copy()
        
        # 필요한 컬럼 선택 및 정리
        required_columns = ['model', 'version', 'strategy_name', 'description', 'example']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if not available_columns:
            return None, "필요한 컬럼을 찾을 수 없습니다."
        
        # 예시들을 전략별로 그룹화
        grouped_data = []
        for strategy_name, group in df.groupby('strategy_name'):
            # 각 전략에 대한 모든 예시들을 수집
            examples = []
            models_versions = []
            sub_strategies_list = []
            
            for _, row in group.iterrows():
                if pd.notna(row.get('example', '')) and str(row.get('example', '')).strip():
                    examples.append(str(row['example']))
                    models_versions.append(f"{row.get('model', '')}-{row.get('version', '')}")
                
                # sub_strategies 수집
                if pd.notna(row.get('sub_strategies', '')) and str(row.get('sub_strategies', '')).strip():
                    sub_strategies_list.append(str(row['sub_strategies']))
            
            # examples를 번호와 함께 HTML 형태로 변환
            examples_html = ''
            if examples:
                examples_list = [f"{i+1}. {example}" for i, example in enumerate(examples)]
                examples_html = '<br>'.join(examples_list)
            
            # 고유한 모델-버전 조합
            unique_sources = list(set(models_versions))
            
            # sub_strategies 합치기 (중복 제거)
            all_sub_strategies = set()
            for sub_str in sub_strategies_list:
                # 콤마로 분리하여 개별 sub_strategy들 추출
                individual_subs = [s.strip() for s in sub_str.split(',')]
                all_sub_strategies.update(individual_subs)
            
            # sub_strategies를 정렬된 리스트로 변환
            sorted_sub_strategies = sorted(list(all_sub_strategies))
            sub_strategies_display = ', '.join(sorted_sub_strategies)
            
            grouped_data.append({
                'strategy_name': strategy_name,
                'description': group.iloc[0].get('description', ''),
                'examples': examples_html,
                'sources': ', '.join(sorted(unique_sources)),
                'sub_strategies': sub_strategies_display,
                'sub_strategies_count': len(sorted_sub_strategies)
            })
        
        return grouped_data, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """메인 페이지 - 통합 결과 및 모델 선택"""
    return render_template('index.html', models=list(MODEL_FILES.keys()))

@app.route('/integrated')
def integrated_viewer():
    """통합 Excel 데이터 뷰어 페이지"""
    return render_template('data_viewer.html', data_source='integrated', version='통합 데이터')

@app.route('/model/<model>/<version>')
def model_version_viewer(model, version):
    """특정 모델의 특정 버전 데이터 뷰어 페이지"""
    if model not in MODEL_FILES:
        return f"지원하지 않는 모델입니다: {model}", 404
    if version not in MODEL_FILES[model]:
        return f"지원하지 않는 버전입니다: {version}", 404
    
    return render_template('data_viewer.html', 
                         data_source='model_version', 
                         model=model, 
                         version=version.upper(),
                         display_name=f"{model} {version.upper()}")

@app.route('/taxonomy')
def final_taxonomy():
    """최종 taxonomy 뷰어"""
    return render_template('taxonomy_viewer.html')

@app.route('/network_analysis')
def network_analysis():
    """네트워크 분석 및 상관관계 분석 페이지"""
    return render_template('network_analysis.html')

def get_final_taxonomy_mapping():
    """완전한 92개 전략을 8개 카테고리에 매핑한 원래 분류 체계"""
    return {
        '감정 조작': {
            'description': '논리적 근거를 제시하는 대신 공포, 동정, 분노, 희망과 같은 감정을 직접적으로 자극하는 전략. 이를 통해 사용자의 비판적 사고 능력을 마비시키고, 감정적 반응에 기반하여 특정 결론을 즉각적으로 수용하도록 유도하는 것을 목표로 함',
            'strategies': [
                'Appeal to Emotion',
                'Emotional Appeal', 
                'Emotional Appeals and Loaded Language',
                'Emotional Manipulation',
                'Emotional Manipulation and Loaded Language',
                'Emotional Manipulation and Moral Framing',
                'Emotional and Loaded Language',
                'Emotional and Moral Appeals',
                'Exploiting Emotions and Biased Language',
                'Exaggerating Threats and Consequences',
                'Risk Amplification'
            ]
        },
        '권위 오용': {
            'description': '존재하지 않는 전문가나 기관을 인용하거나, 해당 주제와 관련 없는 전문가의 의견을 근거로 제시하는 전략. 이를 통해 사용자가 내용의 타당성을 직접 검증하는 노력을 생략하고, 출처의 권위에 의존하여 주장을 무비판적으로 수용하도록 만드는 것을 목표로 함',
            'strategies': [
                'Appeal to Authority',
                'Appeal to Flawed Authority or Belief',
                'Appeal to Unwarranted Authority',
                'Appealing to False Authority',
                'Appeals to Authority and Social Pressure',
                'Misleading Appeals to Authority',
                'Misuse of Authority and Sourcing'
            ]
        },
        '증거 왜곡·선택적 제시': {
            'description': '통계 수치를 왜곡하거나(e.g., 상관관계를 인과관계처럼 제시), 전체 맥락을 무시하고 자신에게 유리한 특정 사례나 데이터만 선별하여(Cherry-picking) 제시하는 전략. 이를 통해 증거가 충분하고 객관적인 것처럼 보이게 만들어, 주장의 신뢰도를 거짓으로 포장하는 것을 목표로 함',
            'strategies': [
                'Data Manipulation Tactics',
                'Evidence Distortion',
                'Evidence Misrepresentation',
                'Manipulating Data and Evidence',
                'Misleading Use of Data',
                'Misleading Use of Evidence',
                'Misleading Use of Evidence and Authority',
                'Misrepresentation of Evidence',
                'Misrepresenting Data and Evidence',
                'Misrepresenting Evidence',
                'Misrepresenting Evidence or Authority',
                'Misuse of Evidence and Authority',
                'Presenting a One-Sided Case'
            ]
        },
        '논점 전환·왜곡': {
            'description': '상대방의 주장을 왜곡하여 공격하거나(허수아비), 주장을 하는 사람 자체를 공격하거나(인신공격), 완전히 다른 주제로 화제를 돌려(논점 흐리기) 핵심 쟁점에 대한 정당한 논증을 회피하는 것을 목표로 하는 전략',
            'strategies': [
                'Ad Hominem and Social Pressure',
                'Ad Hominem and Source Attacks',
                'Argument Dismissal and Diversion',
                'Argument Diversion and Misdirection',
                'Argument Misrepresentation',
                'Distorting the Argument',
                'Diversion and Evasion',
                'Misrepresentation',
                'Misrepresentation and Oversimplification',
                'Misrepresentation of the Argument',
                'Misrepresenting Opposing Arguments',
                'Misrepresenting the Argument',
                'Misrepresenting the Opponent\'s Argument'
            ]
        },
        '추론 오류': {
            'description': '전제들이 결론을 논리적으로 전혀 뒷받침하지 못하는, 즉 논증의 내부 구조 자체에 명백한 결함이 있는 주장을 제시하는 전략. 다른 기만 전략과 달리, 내용이나 출처가 아닌 순수한 논리적 형식의 오류를 통해 사용자를 혼란스럽게 만드는 것을 목표로 함',
            'strategies': [
                'Circular Reasoning',
                'Circular Reasoning and Assertion',
                'False Choices and Exaggerated Outcomes',
                'False Dilemma and Exaggerated Outcomes',
                'False Equivalence',
                'Faulty Logic and Argument Manipulation',
                'Faulty Logic and Causality',
                'Faulty Logic and Framing',
                'Flawed Causal Reasoning',
                'Flawed Causal and Analogical Reasoning',
                'Flawed Logic and Causation',
                'Flawed Logic and False Connections',
                'Flawed Logical Structure',
                'Flawed Reasoning and False Logic',
                'Logical Fallacies',
                'Logical Fallacies and Faulty Reasoning',
                'Presenting a False Dilemma',
                'Slippery Slope Argument',
                'Using Faulty Reasoning',
                'Using Flawed Comparisons'
            ]
        },
        '프레이밍': {
            'description': '동일한 정보나 상황에 대해, 의도적으로 선택된 단어(e.g., \'세금\' vs \'사회적 투자\'), 긍정/부정적 비유, 강조점을 사용하여 특정 \'틀(Frame)\'을 만드는 전략. 이를 통해 사용자가 정보를 받아들이는 인식 자체를, 논증 없이도 자신에게 유리한 방향으로 유도하는 것을 목표로 함',
            'strategies': [
                'Deceptive Framing and Language',
                'Deceptive Framing and Minimization',
                'Deceptive Language',
                'Distortion and Misleading Framing',
                'Distortion of Context and Reality',
                'False Choice Framing',
                'False Compromise',
                'False Neutrality',
                'False Solution',
                'Manipulative Framing and Language',
                'Manipulative Language',
                'Manipulative Language and Framing',
                'Oversimplification and Minimization',
                'Rhetorical Devices',
                'Strategic Framing and Simplification',
                'Technological Solutionism'
            ]
        },
        '사회적 통념에 호소': {
            'description': '"다수가 그렇게 믿는다(군중심리)", "원래부터 그래왔다(전통)"와 같이, 사회적으로 널리 받아들여지는 통념이나 관습을 근거로 제시하는 전략. 이를 통해 자신의 주장에 대한 논리적 입증 책임을 회피하고, 사회적 압력을 통해 주장을 수용하게 만드는 것을 목표로 함',
            'strategies': [
                'Appeal to Popularity',
                'Appealing to Common Practice & Belief',
                'Appealing to Group Beliefs',
                'Appeals to Common Belief and Tradition',
                'Appeals to Common Beliefs and Social Pressure',
                'Exploitation of Social Biases',
                'Exploiting Social Biases and Norms',
                'Misapplied Appeals to Principle',
                'Systemic Bias Appeals'
            ]
        },
        '불확실성 관련 전략': {
            'description': '완벽 주의의 오류를 지적하거나 "내 주장이 틀렸다는 것을 증명해봐라"는 식으로 입증 책임을 상대에게 전가하는 전략. 이를 통해 자신의 주장을 반박하거나 검증하는 것 자체를 어렵게 만들어, 논쟁에서 빠져나가거나 의도적인 혼란을 조성하는 것을 목표로 함',
            'strategies': [
                'Appeal to Ignorance',
                'Appeal to Impracticality',
                'Exploiting Uncertainty and Impossible Standards'
            ]
        }
    }

def get_strategy_descriptions():
    """각 전략에 대한 상세 설명을 반환하는 함수 - 실제 92개 전략만"""
    return {
        # 감정 조작 전략들 (11개)
        'Appeal to Emotion': '논리적 근거 대신 감정적 반응을 유도하여 판단력을 흐리게 만드는 전략. 공포, 동정, 분노, 희망 등의 감정을 직접적으로 자극하여 비판적 사고를 마비시키고 즉각적인 동의를 얻으려 함.',
        'Emotional Appeal': '감정에 호소하여 논리적 판단을 우회하는 전략. 이성적 분석보다는 감정적 반응을 우선시하도록 유도하여 특정 결론을 수용하게 만듦.',
        'Emotional Appeals and Loaded Language': '감정적 호소와 편향된 언어를 결합하여 더 강력한 감정적 반응을 이끌어내는 전략. 중립적 표현을 피하고 감정적으로 치우친 단어를 사용함.',
        'Emotional Manipulation': '감정을 의도적으로 조작하여 논리적 사고를 방해하는 전략. 다양한 감정적 기법을 사용해 비판적 판단력을 무력화시킴.',
        'Emotional Manipulation and Loaded Language': '감정 조작과 편향된 언어를 동시에 사용하는 복합적 전략. 감정적 반응을 극대화하면서 특정 관점을 강요함.',
        'Emotional Manipulation and Moral Framing': '감정 조작과 도덕적 프레이밍을 결합하는 전략. 도덕적 우월감이나 죄책감을 유발하여 감정적 동의를 강요함.',
        'Emotional and Loaded Language': '감정적 언어와 편향된 표현을 사용하여 객관적 판단을 흐리는 전략. 중립적 사실보다는 감정적 인상을 우선시함.',
        'Emotional and Moral Appeals': '감정적 호소와 도덕적 호소를 결합하여 더 강력한 설득력을 만들어내는 전략. 옳고 그름의 판단을 감정에 의존하게 만듦.',
        'Exploiting Emotions and Biased Language': '감정을 악용하고 편향된 언어를 사용하여 불공정한 이익을 취하는 전략. 감정적 취약점을 이용하여 판단력을 흐림.',
        'Exaggerating Threats and Consequences': '위협과 결과를 과장하여 공포감을 조성하는 전략. 실제 위험성보다 훨씬 크게 부풀려 극단적인 반응을 유도함.',
        'Risk Amplification': '실제 위험성을 과장하여 제시하는 전략. 통계적 확률이나 과학적 근거를 무시하고 위험성만을 극대화하여 공포감을 증폭시킴.',
        
        # 권위 오용 전략들 (7개)
        'Appeal to Authority': '전문가나 권위자의 의견을 무비판적으로 받아들이도록 유도하는 전략. 해당 분야의 전문가가 아니거나 편향된 권위를 인용하여 주장의 신뢰성을 거짓으로 높임.',
        'Appeal to Flawed Authority or Belief': '결함이 있는 권위나 믿음에 호소하는 전략. 편향되거나 신뢰할 수 없는 권위자의 의견을 근거로 제시함.',
        'Appeal to Unwarranted Authority': '부적절한 권위나 자격이 없는 전문가의 의견을 근거로 사용하는 전략. 권위의 적절성을 검증하지 않고 무조건적인 신뢰를 요구함.',
        'Appealing to False Authority': '해당 분야의 전문가가 아닌 사람의 의견을 전문가 의견인 것처럼 제시하는 전략. 유명인이나 다른 분야의 전문가를 해당 주제의 권위자로 둔갑시킴.',
        'Appeals to Authority and Social Pressure': '권위에 대한 호소와 사회적 압력을 결합하는 전략. 권위자의 의견과 다수의 압력을 동시에 사용함.',
        'Misleading Appeals to Authority': '권위자의 의견을 왜곡하거나 맥락을 벗어나게 인용하여 자신의 주장을 뒷받침하는 전략. 전문가의 실제 견해와 다르게 해석하여 제시함.',
        'Misuse of Authority and Sourcing': '권위와 출처를 잘못 사용하는 전략. 부적절한 권위를 인용하거나 출처를 왜곡하여 신뢰성을 가장함.',
        
        # 증거 왜곡·선택적 제시 전략들 (13개)
        'Data Manipulation Tactics': '데이터를 조작하는 전술. 원래 데이터의 의미를 왜곡하거나 편향된 방식으로 해석하여 원하는 결론을 도출함.',
        'Evidence Distortion': '증거를 왜곡하는 전략. 사실을 비틀거나 과장하여 실제와 다른 인상을 주도록 조작함.',
        'Evidence Misrepresentation': '증거를 잘못 제시하는 전략. 증거의 원래 의미나 맥락을 무시하고 자신에게 유리하게 해석함.',
        'Manipulating Data and Evidence': '데이터와 증거를 조작하는 전략. 선택적으로 데이터를 제시하거나 증거의 맥락을 왜곡함.',
        'Misleading Use of Data': '데이터를 오도하는 방식으로 사용하는 전략. 데이터의 본래 의미를 왜곡하거나 부적절하게 해석함.',
        'Misleading Use of Evidence': '증거를 오도하는 방식으로 사용하는 전략. 증거를 선택적으로 제시하거나 맥락을 무시함.',
        'Misleading Use of Evidence and Authority': '증거와 권위를 동시에 오도하는 방식으로 사용하는 전략. 왜곡된 증거와 부적절한 권위를 결합함.',
        'Misrepresentation of Evidence': '증거를 잘못 표현하는 전략. 증거의 실제 내용과 다르게 해석하여 제시함.',
        'Misrepresenting Data and Evidence': '데이터와 증거를 잘못 표현하는 전략. 실제 내용을 왜곡하여 다른 의미로 해석함.',
        'Misrepresenting Evidence': '증거를 잘못 표현하는 전략. 원래 증거의 의미를 바꿔서 제시함.',
        'Misrepresenting Evidence or Authority': '증거나 권위를 잘못 표현하는 전략. 증거나 권위자의 실제 입장을 왜곡함.',
        'Misuse of Evidence and Authority': '증거와 권위를 잘못 사용하는 전략. 부적절한 증거나 권위를 근거로 사용함.',
        'Presenting a One-Sided Case': '일방적인 사례만 제시하는 전략. 반대 증거나 다른 관점은 완전히 배제하고 한쪽 면만 강조함.',
        
        # 논점 전환·왜곡 전략들 (13개)
        'Ad Hominem and Social Pressure': '인신공격과 사회적 압력을 결합하는 전략. 개인을 공격하면서 동시에 사회적 압력을 가함.',
        'Ad Hominem and Source Attacks': '인신공격과 출처 공격을 결합하는 전략. 개인과 그 출처를 동시에 공격하여 신뢰성을 훼손함.',
        'Argument Dismissal and Diversion': '논증 무시와 논점 전환을 결합하는 전략. 상대방 논증을 무시하면서 다른 주제로 돌림.',
        'Argument Diversion and Misdirection': '논증 전환과 잘못된 방향 유도를 결합하는 전략. 논점을 다른 곳으로 돌려 혼란을 야기함.',
        'Argument Misrepresentation': '논증을 잘못 표현하는 전략. 상대방의 실제 주장을 왜곡하여 다른 의미로 해석함.',
        'Distorting the Argument': '논증을 왜곡하는 전략. 상대방의 주장을 비틀어서 원래 의미와 다르게 만듦.',
        'Diversion and Evasion': '논점 전환과 회피를 결합하는 전략. 핵심 문제에서 벗어나 다른 주제로 도피함.',
        'Misrepresentation': '잘못된 표현 전략. 사실이나 주장을 원래 의미와 다르게 제시함.',
        'Misrepresentation and Oversimplification': '잘못된 표현과 과도한 단순화를 결합하는 전략. 복잡한 문제를 왜곡하여 단순화함.',
        'Misrepresentation of the Argument': '논증에 대한 잘못된 표현 전략. 상대방의 논증을 실제와 다르게 해석하여 제시함.',
        'Misrepresenting Opposing Arguments': '반대 논증을 잘못 표현하는 전략. 상대방의 주장을 왜곡하여 약하게 만듦.',
        'Misrepresenting the Argument': '논증을 잘못 표현하는 전략. 실제 논증의 내용을 왜곡하여 다르게 제시함.',
        'Misrepresenting the Opponent\'s Argument': '상대방 논증을 잘못 표현하는 전략. 상대방의 실제 주장을 왜곡하여 공격하기 쉽게 만듦.',
        
        # 추론 오류 전략들 (20개)
        'Circular Reasoning': '순환논리 전략. 결론을 전제로 사용하여 증명되지 않은 가정으로 그 가정을 증명하려는 논리적 오류.',
        'Circular Reasoning and Assertion': '순환논리와 단언을 결합하는 전략. 증명 없는 주장을 반복하여 마치 증명된 것처럼 제시함.',
        'False Choices and Exaggerated Outcomes': '거짓 선택지와 과장된 결과를 결합하는 전략. 제한적 선택지를 제시하면서 극단적 결과를 강조함.',
        'False Dilemma and Exaggerated Outcomes': '거짓 딜레마와 과장된 결과를 결합하는 전략. 두 가지 극단적 선택만 제시하면서 과장된 결과를 경고함.',
        'False Equivalence': '거짓 동등성 전략. 본질적으로 다른 두 가지를 동등하게 취급하여 혼란을 야기함.',
        'Faulty Logic and Argument Manipulation': '잘못된 논리와 논증 조작을 결합하는 전략. 논리적 오류와 논증 왜곡을 동시에 사용함.',
        'Faulty Logic and Causality': '잘못된 논리와 인과관계를 결합하는 전략. 논리적 오류와 잘못된 인과관계를 동시에 사용함.',
        'Faulty Logic and Framing': '잘못된 논리와 프레이밍을 결합하는 전략. 논리적 오류와 편향된 프레이밍을 동시에 사용함.',
        'Flawed Causal Reasoning': '결함 있는 인과 추론 전략. 원인과 결과의 관계를 잘못 설정하여 논리적 오류를 범함.',
        'Flawed Causal and Analogical Reasoning': '결함 있는 인과 추론과 유추 추론을 결합하는 전략. 잘못된 인과관계와 부적절한 유추를 동시에 사용함.',
        'Flawed Logic and Causation': '결함 있는 논리와 인과관계를 결합하는 전략. 논리적 오류와 잘못된 인과관계를 동시에 범함.',
        'Flawed Logic and False Connections': '결함 있는 논리와 거짓 연결을 결합하는 전략. 논리적 오류와 존재하지 않는 연관성을 동시에 제시함.',
        'Flawed Logical Structure': '결함 있는 논리적 구조 전략. 논증의 기본 구조 자체에 오류가 있어 결론이 전제를 따르지 않음.',
        'Flawed Reasoning and False Logic': '결함 있는 추론과 거짓 논리를 결합하는 전략. 잘못된 추론 과정과 논리적 오류를 동시에 범함.',
        'Logical Fallacies': '논리적 오류 전략. 다양한 논리적 오류를 사용하여 잘못된 결론을 도출함.',
        'Logical Fallacies and Faulty Reasoning': '논리적 오류와 잘못된 추론을 결합하는 전략. 여러 논리적 오류와 추론 오류를 동시에 범함.',
        'Presenting a False Dilemma': '거짓 딜레마 제시 전략. 실제로는 더 많은 선택지가 있음에도 두 가지 극단적 선택만 제시함.',
        'Slippery Slope Argument': '미끄러운 비탈길 논증 전략. 작은 변화가 연쇄적으로 극단적 결과를 가져올 것이라고 주장함.',
        'Using Faulty Reasoning': '잘못된 추론 사용 전략. 논리적 결함이 있는 추론 과정을 사용하여 결론을 도출함.',
        'Using Flawed Comparisons': '결함 있는 비교 사용 전략. 적절하지 않은 비교를 통해 잘못된 결론을 도출함.',
        
        # 프레이밍 전략들 (16개)
        'Deceptive Framing and Language': '기만적 프레이밍과 언어를 결합하는 전략. 왜곡된 틀과 편향된 언어를 동시에 사용함.',
        'Deceptive Framing and Minimization': '기만적 프레이밍과 최소화를 결합하는 전략. 왜곡된 틀로 문제를 축소하여 제시함.',
        'Deceptive Language': '기만적 언어 전략. 진실을 왜곡하거나 숨기는 언어를 사용하여 잘못된 인상을 줌.',
        'Distortion and Misleading Framing': '왜곡과 오도하는 프레이밍을 결합하는 전략. 사실을 비틀고 잘못된 틀로 제시함.',
        'Distortion of Context and Reality': '맥락과 현실을 왜곡하는 전략. 실제 상황을 비틀어서 다른 의미로 해석하게 만듦.',
        'False Choice Framing': '거짓 선택 프레이밍 전략. 실제보다 제한적인 선택지만 있는 것처럼 틀을 만듦.',
        'False Compromise': '거짓 타협 전략. 실제로는 타협이 아닌 것을 타협인 것처럼 제시하여 수용을 유도함.',
        'False Neutrality': '거짓 중립성을 주장하는 전략. 편향된 입장을 중립적인 것처럼 포장하여 객관성을 가장함.',
        'False Solution': '거짓 해결책 전략. 실제로는 문제를 해결하지 못하는 방안을 해결책인 것처럼 제시함.',
        'Manipulative Framing and Language': '조작적 프레이밍과 언어를 결합하는 전략. 조작된 틀과 편향된 언어를 동시에 사용함.',
        'Manipulative Language': '조작적 언어 전략. 상대방을 조작하려는 의도가 담긴 언어를 사용함.',
        'Manipulative Language and Framing': '조작적 언어와 프레이밍을 결합하는 전략. 조작된 언어와 편향된 틀을 동시에 사용함.',
        'Oversimplification and Minimization': '과도한 단순화와 최소화를 결합하는 전략. 복잡한 문제를 지나치게 단순화하면서 중요성을 축소함.',
        'Rhetorical Devices': '수사적 기법 전략. 논리적 근거보다는 수사적 효과를 통해 설득하려는 기법들.',
        'Strategic Framing and Simplification': '전략적 프레이밍과 단순화를 결합하는 전략. 의도적으로 특정 틀을 만들고 문제를 단순화함.',
        'Technological Solutionism': '기술만능주의 전략. 복잡한 사회적 문제를 기술로만 해결할 수 있다고 주장함.',
        
        # 사회적 통념에 호소 전략들 (9개)
        'Appeal to Popularity': '인기에 대한 호소 전략. 많은 사람이 지지한다는 이유만으로 그것이 옳다고 주장함.',
        'Appealing to Common Practice & Belief': '일반적 관행과 믿음에 호소하는 전략. 널리 행해지는 것이 옳다는 잘못된 가정을 이용함.',
        'Appealing to Group Beliefs': '집단 믿음에 호소하는 전략. 특정 집단의 공통된 믿음을 근거로 주장의 타당성을 주장함.',
        'Appeals to Common Belief and Tradition': '일반적 믿음과 전통에 호소하는 전략. 오래된 믿음이나 전통이 옳다는 가정을 이용함.',
        'Appeals to Common Beliefs and Social Pressure': '일반적 믿음과 사회적 압력에 호소하는 전략. 다수의 믿음과 사회적 압력을 동시에 활용함.',
        'Exploitation of Social Biases': '사회적 편견 악용 전략. 사회에 널리 퍼진 편견을 이용하여 특정 결론을 유도함.',
        'Exploiting Social Biases and Norms': '사회적 편견과 규범을 악용하는 전략. 사회적 편견과 규범을 동시에 이용함.',
        'Misapplied Appeals to Principle': '원칙에 대한 잘못된 호소 전략. 원칙을 부적절하게 적용하여 자신의 주장을 정당화함.',
        'Systemic Bias Appeals': '체계적 편향에 대한 호소 전략. 시스템에 내재된 편향을 이용하여 특정 관점을 강화함.',
        
        # 불확실성 관련 전략들 (3개)
        'Appeal to Ignorance': '무지에 대한 호소 전략. 어떤 것이 거짓으로 증명되지 않았다는 이유로 그것이 참이라고 주장함.',
        'Appeal to Impracticality': '비실용성에 대한 호소 전략. 실행하기 어렵다는 이유로 이론적 타당성을 부정함.',
        'Exploiting Uncertainty and Impossible Standards': '불확실성과 불가능한 기준을 악용하는 전략. 완벽한 확실성을 요구하여 합리적 결론을 방해함.'
    }

@app.route('/api/get_strategy_description/<strategy_name>')
def api_get_strategy_description(strategy_name):
    """특정 전략의 상세 설명을 반환하는 API"""
    descriptions = get_strategy_descriptions()
    
    # URL에서 받은 전략명을 디코딩
    from urllib.parse import unquote
    decoded_strategy_name = unquote(strategy_name)
    
    if decoded_strategy_name in descriptions:
        return jsonify({
            'strategy_name': decoded_strategy_name,
            'description': descriptions[decoded_strategy_name]
        })
    else:
        # 전략명이 딕셔너리에 없는 경우 기본 설명 반환
        return jsonify({
            'strategy_name': decoded_strategy_name,
            'description': f'"{decoded_strategy_name}"는 기만적 설득 기법 중 하나입니다. 이 전략에 대한 구체적인 설명은 현재 준비 중입니다.'
        })

@app.route('/api/get_strategies')
def api_get_strategies():
    """전략 이름 목록과 개수를 반환하는 API (통합 Excel용)"""
    model_filter = request.args.get('model', None)
    
    df = load_excel_data()
    if df is None:
        return jsonify({'error': '파일을 로드할 수 없습니다.'}), 400
    
    if 'strategy_name' not in df.columns:
        return jsonify({'error': 'strategy_name 컬럼을 찾을 수 없습니다.'}), 400
    
    # 각 전략별 개수
    strategy_counts = get_strategy_counts(df, model_filter)
    
    # 개수가 많은 순으로 정렬된 전략 목록
    sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
    strategies = [strategy for strategy, count in sorted_strategies]
    
    # 사용 가능한 모델 목록
    available_models = ['all'] + list(df['model'].unique()) if 'model' in df.columns else ['all']
    
    return jsonify({
        'strategies': strategies,
        'strategy_counts': strategy_counts,
        'available_models': available_models
    })

@app.route('/api/get_strategies/<model>/<version>')
def api_get_strategies_model_version(model, version):
    """특정 모델/버전의 전략 이름 목록과 개수를 반환하는 API"""
    data, error = load_json_data(model, version)
    if error:
        return jsonify({'error': error}), 400
    
    # 각 전략별 개수
    strategy_counts = get_json_strategy_counts(data)
    
    # 개수가 많은 순으로 정렬된 전략 목록
    sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
    strategies = [strategy for strategy, count in sorted_strategies]
    
    return jsonify({
        'strategies': strategies,
        'strategy_counts': strategy_counts
    })

@app.route('/api/load_data')
def api_load_data():
    """데이터 로드 API (통합 Excel용)"""
    strategy_filter = request.args.get('strategy', None)
    model_filter = request.args.get('model', None)
    
    df = load_excel_data()
    if df is None:
        return jsonify({'error': '파일을 로드할 수 없습니다.'}), 400
    
    # 데이터 처리
    processed_data, error = process_excel_data(df, strategy_filter, model_filter)
    if error:
        return jsonify({'error': error}), 400
    
    # 컬럼 이름들
    columns = ['strategy_name', 'description', 'examples', 'sources', 'sub_strategies', 'sub_strategies_count']
    
    return jsonify({
        'data': processed_data,
        'columns': columns,
        'total_rows': len(processed_data),
        'total_columns': len(columns)
    })

@app.route('/api/load_data/<model>/<version>')
def api_load_data_model_version(model, version):
    """데이터 로드 API (특정 모델/버전용)"""
    strategy_filter = request.args.get('strategy', None)
    
    data, error = load_json_data(model, version)
    if error:
        return jsonify({'error': error}), 400
    
    # 데이터 처리
    processed_data, error = process_json_data(data, strategy_filter)
    if error:
        return jsonify({'error': error}), 400
    
    # 컬럼 이름들
    columns = ['strategy_name', 'description', 'examples', 'sub_strategies', 'sub_strategies_count']
    
    return jsonify({
        'data': processed_data,
        'columns': columns,
        'total_rows': len(processed_data),
        'total_columns': len(columns)
    })

@app.route('/api/get_taxonomy_data')
def api_get_taxonomy_data():
    """최종 taxonomy 데이터와 각 카테고리별 전략 매핑을 반환하는 API"""
    try:
        taxonomy_mapping = get_final_taxonomy_mapping()
        result = {}
        
        for category_name, category_data in taxonomy_mapping.items():
            result[category_name] = {
                'description': category_data['description'],
                'strategies': category_data['strategies']
            }
        
        return jsonify({
            'taxonomy': result,
            'total_categories': len(result)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check_strategy_coverage')
def api_check_strategy_coverage():
    """전략 커버리지 확인 API"""
    # 실제 데이터에서 전략들 추출
    df = load_excel_data()
    if df is None:
        return jsonify({'error': '파일을 로드할 수 없습니다.'}), 400
    
    actual_strategies = set(df['strategy_name'].unique())
    
    # 매핑된 전략들
    taxonomy_mapping = get_final_taxonomy_mapping()
    mapped_strategies = set()
    for category_data in taxonomy_mapping.values():
        mapped_strategies.update(category_data['strategies'])
    
    # 차집합
    not_mapped = actual_strategies - mapped_strategies
    not_in_data = mapped_strategies - actual_strategies
    
    return jsonify({
        'actual_strategies_count': len(actual_strategies),
        'mapped_strategies_count': len(mapped_strategies),
        'not_mapped': list(not_mapped),
        'not_in_data': list(not_in_data),
        'coverage_percentage': (len(actual_strategies - not_mapped) / len(actual_strategies) * 100) if len(actual_strategies) > 0 else 0
    })

@app.route('/api/gemini/v4/analysis')
def gemini_v4_analysis():
    """
    Performs a full pipeline analysis for Gemini v4 data.
    1. Builds mappings from raw strategies -> final strategies.
    2. Validates the mapping for ambiguity.
    3. Processes all v4 open coding data.
    4. Answers key analysis questions (complexity, co-occurrence).
    """
    try:
        # --- Phase 1: Build and Validate Mappings ---
        v4_taxonomy_path = 'gemini-2.5-pro/v4_deceptive_taxonomy.json.json'
        if not os.path.exists(v4_taxonomy_path):
            return jsonify({'error': f'Taxonomy file not found: {v4_taxonomy_path}'}), 404

        with open(v4_taxonomy_path, 'r', encoding='utf-8') as f:
            v4_taxonomy_data = json.load(f)

        raw_to_final_map = {}
        raw_strategy_conflicts = defaultdict(set)
        all_final_strategies_from_mapping = set()

        for final_strategy_group in v4_taxonomy_data:
            final_strategy_name = final_strategy_group.get('strategy_name')
            if not final_strategy_name:
                continue
            all_final_strategies_from_mapping.add(final_strategy_name)
            for raw_strategy in final_strategy_group.get('sub_strategies', []):
                if raw_strategy in raw_to_final_map and raw_to_final_map[raw_strategy] != final_strategy_name:
                    raw_strategy_conflicts[raw_strategy].add(raw_to_final_map[raw_strategy])
                    raw_strategy_conflicts[raw_strategy].add(final_strategy_name)
                raw_to_final_map[raw_strategy] = final_strategy_name

        # --- Phase 2: Aggregate and Analyze Gemini v4 Open Coding Data ---
        open_coding_path = 'open_coding/gemini-2.5-pro/v4/'
        all_files = glob.glob(os.path.join(open_coding_path, '*.json'))
        
        argument_data = defaultdict(lambda: {'raw_strategies': set()})
        all_raw_strategies_from_data = set()

        for f_path in all_files:
            with open(f_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for item in data:
                        raw_strat = item.get('strategy_name')
                        if raw_strat:
                            argument_data[item['argument']]['raw_strategies'].add(raw_strat)
                            all_raw_strategies_from_data.add(raw_strat)
                except json.JSONDecodeError:
                    continue

        argument_final_strategies = {}
        for arg, data in argument_data.items():
            final_strategies = {raw_to_final_map.get(rs) for rs in data['raw_strategies'] if raw_to_final_map.get(rs)}
            if final_strategies:
                argument_final_strategies[arg] = final_strategies

        # --- Phase 3: Answer Key Questions ---

        # Question 1: Argument Complexity
        complexities = [len(strats) for strats in argument_final_strategies.values()]
        complexity_distribution = Counter(complexities)
        
        # Convert numpy types to native Python types for JSON serialization
        complexity_analysis = {
            'average_strategies_per_argument': float(np.mean(complexities)) if complexities else 0,
            'distribution': {str(k): int(v) for k, v in sorted(complexity_distribution.items())}
        }

        # Question 2: Co-occurrence of Final Strategies
        co_occurrence_counter = Counter()
        for strats in argument_final_strategies.values():
            if len(strats) > 1:
                for pair in itertools.combinations(sorted(list(strats)), 2):
                    co_occurrence_counter[pair] += 1
        
        co_occurrence_analysis = {
            'top_pairs': [
                {'pair': list(pair), 'count': int(count)}
                for pair, count in co_occurrence_counter.most_common(10)
            ]
        }

        # --- Final Report Assembly ---
        
        # Reliability Report
        reliability_report = {
            'mapping_conflicts': [{
                'raw_strategy': k,
                'maps_to': list(v)
            } for k, v in raw_strategy_conflicts.items()],
            'total_arguments': int(len(argument_data)),
            'total_raw_strategies': int(len(all_raw_strategies_from_data)),
            'total_final_strategies': int(len(all_final_strategies_from_mapping)),
        }

        return jsonify({
            'reliability_report': reliability_report,
            'analysis_results': {
                'argument_complexity': complexity_analysis,
                'strategy_cooccurrence': co_occurrence_analysis
            }
        })

    except Exception as e:
        app.logger.error(f"Error in /api/gemini/v4/analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/model_diversity')
def api_model_diversity():
    """모델별 전략 다양성 데이터를 반환하는 API"""
    try:
        # 각 모델별로 고유 전략 개수 계산
        models = ['gemini-2.5-pro', 'o4-mini', 'deepseek-r1']
        strategy_counts = []
        top_strategies = {}
        
        for model in models:
            # 각 모델의 모든 버전에서 전략 수집
            all_strategies = set()
            for version in ['v1', 'v2', 'v3', 'v4', 'v5']:
                if model in MODEL_FILES and version in MODEL_FILES[model]:
                    file_path = MODEL_FILES[model][version]
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                for item in data:
                                    if 'strategy_name' in item:
                                        all_strategies.add(item['strategy_name'])
                        except Exception as e:
                            app.logger.warning(f"Error reading {file_path}: {e}")
                            continue
            
            strategy_counts.append(len(all_strategies))
            
            # 상위 20개 전략 (빈도순)
            strategy_freq = Counter()
            for version in ['v1', 'v2', 'v3', 'v4', 'v5']:
                if model in MODEL_FILES and version in MODEL_FILES[model]:
                    file_path = MODEL_FILES[model][version]
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                for item in data:
                                    if 'strategy_name' in item:
                                        strategy_freq[item['strategy_name']] += 1
                        except Exception as e:
                            continue
            
            top_strategies[model] = {
                'strategies': [strategy for strategy, count in strategy_freq.most_common(20)]
            }
        
        return jsonify({
            'models': models,
            'strategy_counts': strategy_counts,
            'top_strategies': top_strategies
        })
        
    except Exception as e:
        app.logger.error(f"Error in model diversity API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/category_distribution_correct')
def api_category_distribution():
    """카테고리별 분포 데이터를 반환하는 API"""
    try:
        # 최종 taxonomy 매핑 가져오기
        taxonomy_mapping = get_final_taxonomy_mapping()
        categories = list(taxonomy_mapping.keys())
        
        # 전체 카테고리별 전략 개수
        total_counts = [len(taxonomy_mapping[cat]['strategies']) for cat in categories]
        
        # 모델별 분포 계산
        model_distributions = {}
        models = ['gemini-2.5-pro', 'o4-mini', 'deepseek-r1']
        
        for model in models:
            model_counts = [0] * len(categories)
            
            # 각 모델의 모든 버전에서 전략 수집
            for version in ['v1', 'v2', 'v3', 'v4', 'v5']:
                if model in MODEL_FILES and version in MODEL_FILES[model]:
                    file_path = MODEL_FILES[model][version]
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                for item in data:
                                    if 'strategy_name' in item:
                                        strategy = item['strategy_name']
                                        # 어떤 카테고리에 속하는지 확인
                                        for i, cat in enumerate(categories):
                                            if strategy in taxonomy_mapping[cat]['strategies']:
                                                model_counts[i] += 1
                                                break
                        except Exception as e:
                            app.logger.warning(f"Error reading {file_path}: {e}")
                            continue
            
            model_distributions[model] = {
                'counts': model_counts
            }
        
        return jsonify({
            'categories': categories,
            'total_counts': total_counts,
            'model_distributions': model_distributions
        })
        
    except Exception as e:
        app.logger.error(f"Error in category distribution API: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 