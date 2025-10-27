import json
import os
from collections import defaultdict
from datetime import datetime
from datasets import load_dataset

def load_taxonomy_data(file_path):
    """final_deceptive_taxonomy 파일에서 데이터를 로드하고 sub_strategies를 추출"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    taxonomy_mapping = {}
    for item in data:
        strategy_name = item['strategy_name']
        for sub_strategy in item['sub_strategies']:
            taxonomy_mapping[sub_strategy] = strategy_name
    
    return taxonomy_mapping

def load_open_coding_data(directory):
    """open_coding v1~v10 파일들에서 데이터를 로드"""
    open_coding_data = defaultdict(list)
    
    # v1~v10 파일들을 찾아서 로드
    for filename in os.listdir(directory):
        if filename.startswith('open_coding_v') and filename.endswith('.json'):
            version = filename.split('_')[2]  # v1, v2, ... v10 추출
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                strategy_name = item['strategy_name']
                item['version'] = version
                open_coding_data[strategy_name].append(item)
    
    return open_coding_data

def load_huggingface_dataset():
    """허깅페이스 Anthropic/persuasion 데이터셋을 로드"""
    try:
        print("🔄 허깅페이스 데이터셋 로딩 중...")
        dataset = load_dataset("Anthropic/persuasion", split="train")
        print(f"📊 총 {len(dataset)} 개의 데이터 로드 완료")
        
        # 데이터를 딕셔너리 형태로 변환
        hf_data = []
        for item in dataset:
            hf_data.append({
                'worker_id': item.get('worker_id', ''),
                'claim': item.get('claim', ''),
                'argument': item.get('argument', ''),
                'source': item.get('source', ''),
                'rating_initial': item.get('rating_initial', ''),
                'rating_final': item.get('rating_final', ''),
                'persuasiveness_metric': item.get('persuasiveness_metric', ''),
                'prompt_type': item.get('prompt_type', ''),
                'original_index': len(hf_data)  # 원본 인덱스 저장
            })
        
        return hf_data
        
    except Exception as e:
        print(f"❌ 허깅페이스 데이터셋 로딩 실패: {e}")
        return []

def create_matched_dataset(taxonomy_mapping, open_coding_data):
    """taxonomy의 sub_strategies와 open_coding의 strategy_name을 매칭"""
    matched_dataset = []
    
    for sub_strategy, taxonomy_strategy in taxonomy_mapping.items():
        if sub_strategy in open_coding_data:
            # 각 entry에 taxonomy_strategy 정보 추가하여 플랫한 배열로 만듦
            for entry in open_coding_data[sub_strategy]:
                # 기존 entry를 복사하고 taxonomy_strategy 정보 추가
                enhanced_entry = entry.copy()
                enhanced_entry['taxonomy_strategy'] = taxonomy_strategy
                matched_dataset.append(enhanced_entry)
    
    return matched_dataset

def connect_worker_ids(matched_dataset, hf_data):
    """매칭된 데이터셋과 허깅페이스 데이터셋을 worker_id로 연결"""
    print("🔗 Worker ID 연결 수행 중...")
    
    # 허깅페이스 데이터를 worker_id로 인덱싱
    hf_by_worker_id = defaultdict(list)
    for item in hf_data:
        if item['worker_id']:
            hf_by_worker_id[item['worker_id']].append(item)
    
    print(f"📊 허깅페이스 데이터에서 {len(hf_by_worker_id)}개의 고유한 worker_id 발견")
    
    # 매칭된 데이터에 허깅페이스 데이터 연결
    enhanced_dataset = []
    
    for item in matched_dataset:
        enhanced_item = item.copy()
        
        # worker_id가 있는지 확인 (기존 데이터에 worker_id가 있을 수 있음)
        worker_id = item.get('worker_id', '')
        
        if worker_id and worker_id in hf_by_worker_id:
            # 해당 worker_id의 허깅페이스 데이터 중 하나를 선택 (첫 번째 또는 랜덤)
            hf_item = hf_by_worker_id[worker_id][0]  # 첫 번째 선택
            
            # 허깅페이스 데이터의 필드들을 추가 (hf_claim, hf_argument 제외하고 hf_ 접두사 제거)
            enhanced_item.update({
                'source': hf_item['source'],
                'rating_initial': hf_item['rating_initial'],
                'rating_final': hf_item['rating_final'],
                'persuasiveness_metric': hf_item['persuasiveness_metric'],
                'prompt_type': hf_item['prompt_type'],
                'original_index': hf_item['original_index']
            })
        
        enhanced_dataset.append(enhanced_item)
    
    return enhanced_dataset

def save_datasets(enhanced_dataset):
    """결과 데이터셋을 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 통합 데이터셋 저장
    output_file = f"deception_matched_dataset_o4_mini_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
    
    return output_file

def main():
    # 파일 경로 설정
    taxonomy_file = 'o4_mini/final_deceptive_taxonomy_20250707_185859.json'
    open_coding_dir = 'o4_mini'
    
    print("🔍 데이터 로딩 중...")
    
    # 기존 데이터 로드
    taxonomy_mapping = load_taxonomy_data(taxonomy_file)
    open_coding_data = load_open_coding_data(open_coding_dir)
    
    print(f"📊 Taxonomy에서 {len(taxonomy_mapping)}개의 sub_strategies 추출")
    print(f"📊 Open coding에서 {len(open_coding_data)}개의 고유한 strategy_name 추출")
    
    # 허깅페이스 데이터셋 로드
    hf_data = load_huggingface_dataset()
    if not hf_data:
        print("❌ 허깅페이스 데이터 로드 실패. 기존 방식으로 진행합니다.")
        # 기존 방식으로 진행
        matched_dataset = create_matched_dataset(taxonomy_mapping, open_coding_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"deception_matched_dataset_gpt4omini_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(matched_dataset, f, indent=2, ensure_ascii=False)
        print(f"📁 기본 매칭 데이터셋 저장: {output_file}")
        return
    
    # 기존 매칭 수행
    print("\n🔗 기존 매칭 수행 중...")
    matched_dataset = create_matched_dataset(taxonomy_mapping, open_coding_data)
    
    # Worker ID 연결
    enhanced_dataset = connect_worker_ids(matched_dataset, hf_data)
    
    # 결과 저장
    output_file = save_datasets(enhanced_dataset)
    
    # 결과 출력
    print(f"\n✅ 매칭 완료!")
    print(f"📁 통합 데이터셋 저장: {output_file}")
    print(f"📁 총 {len(enhanced_dataset)}개 항목이 통합 데이터셋에 저장됨")

if __name__ == "__main__":
    main() 