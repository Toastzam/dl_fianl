"""
실험 추적 및 결과 분석 스크립트
Loss < 1.0 목표를 위한 하이퍼파라미터 실험 관리
"""

import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ExperimentTracker:
    def __init__(self, base_dir="experiment_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def log_experiment(self, config, results, experiment_name=None):
        """실험 설정과 결과를 로그로 저장"""
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        experiment_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "config": config,
            "results": results
        }
        
        # JSON 파일로 저장
        log_file = self.base_dir / f"experiment_{experiment_name}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)
        
        print(f"실험 로그 저장: {log_file}")
        return log_file
    
    def compare_experiments(self):
        """모든 실험 결과를 비교 분석"""
        experiment_files = list(self.base_dir.glob("experiment_*.json"))
        
        if not experiment_files:
            print("저장된 실험 결과가 없습니다.")
            return
        
        experiments = []
        for file in experiment_files:
            with open(file, 'r', encoding='utf-8') as f:
                experiments.append(json.load(f))
        
        # 실험 결과 정리
        results_df = []
        for exp in experiments:
            config = exp['config']
            results = exp['results']
            
            row = {
                'experiment': exp['experiment_name'],
                'batch_size': config.get('batch_size', 'N/A'),
                'lr': config.get('lr', 'N/A'),
                'temperature': config.get('temperature', 'N/A'),
                'weight_decay': config.get('weight_decay', 'N/A'),
                'best_loss': results.get('best_loss', 'N/A'),
                'final_epoch': results.get('final_epoch', 'N/A'),
                'target_achieved': results.get('best_loss', float('inf')) < 1.0 if results.get('best_loss') != 'N/A' else False
            }
            results_df.append(row)
        
        # 결과 출력
        print("\n=== 실험 결과 비교 ===")
        print(f"{'실험명':<20} {'배치':<6} {'학습률':<8} {'온도':<6} {'가중치감쇠':<10} {'최고Loss':<8} {'목표달성':<8}")
        print("-" * 80)
        
        for row in results_df:
            target_str = "✅" if row['target_achieved'] else "❌"
            print(f"{row['experiment']:<20} {row['batch_size']:<6} {row['lr']:<8} {row['temperature']:<6} {row['weight_decay']:<10} {row['best_loss']:<8.4f} {target_str:<8}")
        
        # 시각화
        self.plot_experiment_comparison(results_df)
        
        return results_df
    
    def plot_experiment_comparison(self, results_df):
        """실험 결과 시각화"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 배치 크기 vs Loss
        batch_sizes = [row['batch_size'] for row in results_df if row['best_loss'] != 'N/A']
        losses = [row['best_loss'] for row in results_df if row['best_loss'] != 'N/A']
        
        if batch_sizes and losses:
            ax1.scatter(batch_sizes, losses, alpha=0.7, s=100)
            ax1.axhline(y=1.0, color='r', linestyle='--', label='목표 (Loss < 1.0)')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Best Loss')
            ax1.set_title('Batch Size vs Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 학습률 vs Loss
        lrs = [row['lr'] for row in results_df if row['best_loss'] != 'N/A']
        if lrs and losses:
            ax2.scatter(lrs, losses, alpha=0.7, s=100)
            ax2.axhline(y=1.0, color='r', linestyle='--', label='목표 (Loss < 1.0)')
            ax2.set_xlabel('Learning Rate')
            ax2.set_ylabel('Best Loss')
            ax2.set_title('Learning Rate vs Loss')
            ax2.set_xscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 온도 vs Loss
        temps = [row['temperature'] for row in results_df if row['best_loss'] != 'N/A']
        if temps and losses:
            ax3.scatter(temps, losses, alpha=0.7, s=100)
            ax3.axhline(y=1.0, color='r', linestyle='--', label='목표 (Loss < 1.0)')
            ax3.set_xlabel('Temperature')
            ax3.set_ylabel('Best Loss')
            ax3.set_title('Temperature vs Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 목표 달성 현황
        achieved = sum(1 for row in results_df if row['target_achieved'])
        total = len(results_df)
        
        ax4.pie([achieved, total - achieved], 
                labels=[f'목표 달성 ({achieved})', f'목표 미달성 ({total - achieved})'],
                colors=['lightgreen', 'lightcoral'],
                autopct='%1.1f%%',
                startangle=90)
        ax4.set_title('목표 달성 현황 (Loss < 1.0)')
        
        plt.tight_layout()
        plt.savefig(self.base_dir / 'experiment_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"비교 그래프 저장: {self.base_dir / 'experiment_comparison.png'}")

# 현재 실험 설정 정의
CURRENT_EXPERIMENT = {
    "batch_size": 64,
    "epochs": 80,
    "lr": 5e-5,
    "temperature": 0.1,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "optimizer": "AdamW",
    "scheduler": "WarmupCosineAnnealing",
    "augmentation": "enhanced",
    "gradient_clipping": 1.0,
    "early_stopping": 10
}

if __name__ == "__main__":
    tracker = ExperimentTracker()
    
    print("=== 현재 실험 설정 ===")
    for key, value in CURRENT_EXPERIMENT.items():
        print(f"{key}: {value}")
    
    print("\n실험 진행 중... 결과는 학습 완료 후 수동으로 추가하세요.")
    
    # 예시: 실험 결과 로그 (실제로는 학습 완료 후 수동 입력)
    # example_results = {
    #     "best_loss": 0.95,
    #     "final_epoch": 25,
    #     "training_time": "45분",
    #     "early_stopped": True
    # }
    # tracker.log_experiment(CURRENT_EXPERIMENT, example_results, "enhanced_config_v1")
    
    # 이전 실험들과 비교
    tracker.compare_experiments()
