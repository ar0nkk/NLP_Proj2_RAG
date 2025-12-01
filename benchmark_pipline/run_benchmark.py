#!/usr/bin/env python3
"""
完整的benchmark测试脚本
- 检查本地QA_data目录中的数据集
- 根据config.yaml配置运行benchmark测试
- 生成带时间戳的独立结果文件

注意：运行前请先使用 download_data.py 下载QA数据集
"""

import sys
from pathlib import Path
import csv
import yaml
import pandas as pd
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_agent import RAGAgent
from step4_rag_answer import process_single_question
from step5_judge_evaluation import evaluate_single_question
from visualize import visualize_results


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    MAGENTA = '\033[35m'
    RESET = '\033[0m'


# 硬编码 Hugging Face 仓库地址
REPO_ID = "HEHUA2005/rag-benchmark-qa-dataset"

# 所有可用的splits
ALL_SPLITS = [
    "Mao_Zedong_Thought",
    "Principles_of_Marxism",
    "Outline_of_Modern_and_Contemporary_Chinese_History",
    "Ideological_Morality_and_Legal_System",
    "An_Introduction_to_Xi_Jinping_Thought_on_Socialism_with_Chinese_Characteristics_for_a_New_Era"
]


def load_config(config_path: Path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_datasets(qa_data_dir: Path, splits: list):
    """检查QA数据集是否存在（支持 parquet 和 csv 格式）"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Checking QA Datasets{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    if not qa_data_dir.exists():
        print(f"{Colors.RED}QA data directory not found: {qa_data_dir}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Please run download_data.py first:{Colors.RESET}")
        print(f"{Colors.CYAN}  python download_data.py --download qa{Colors.RESET}\n")
        return {}

    available_files = {}
    missing_splits = []

    # 检查 data 子目录是否存在
    data_subdir = qa_data_dir / "data"
    search_dirs = [qa_data_dir, data_subdir] if data_subdir.exists() else [qa_data_dir]

    for split_name in splits:
        found = False
        
        for search_dir in search_dirs:
            # 优先检查 parquet 文件（支持 HuggingFace 格式）
            parquet_file = search_dir / f"{split_name}-00000-of-00001.parquet"
            csv_file = search_dir / f"{split_name}.csv"
            
            if parquet_file.exists():
                try:
                    df = pd.read_parquet(parquet_file)
                    row_count = len(df)
                    if row_count > 0:
                        available_files[split_name] = parquet_file
                        print(f"{Colors.GREEN}✓ Found: {parquet_file} ({row_count} questions){Colors.RESET}")
                        found = True
                        break
                except Exception as e:
                    print(f"{Colors.RED}✗ Invalid file: {parquet_file} ({e}){Colors.RESET}")
            
            elif csv_file.exists():
                try:
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        row_count = sum(1 for _ in reader)
                    if row_count > 0:
                        available_files[split_name] = csv_file
                        print(f"{Colors.GREEN}✓ Found: {csv_file} ({row_count} questions){Colors.RESET}")
                        found = True
                        break
                except Exception as e:
                    print(f"{Colors.RED}✗ Invalid file: {csv_file} ({e}){Colors.RESET}")
        
        if not found:
            print(f"{Colors.RED}✗ Not found: {split_name}{Colors.RESET}")
            missing_splits.append(split_name)

    if missing_splits:
        print(f"\n{Colors.YELLOW}Missing or invalid datasets: {', '.join(missing_splits)}{Colors.RESET}")
        print(f"{Colors.YELLOW}Please download them first:{Colors.RESET}")
        print(f"{Colors.CYAN}  python download_data.py --download qa --splits {' '.join(missing_splits)}{Colors.RESET}\n")

    return available_files


def load_questions_from_file(file_path: Path, max_questions: int = None):
    """从文件加载问题（支持 parquet 和 csv 格式）"""
    questions = []
    
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.parquet':
        df = pd.read_parquet(file_path)
        if max_questions:
            df = df.head(max_questions)
        questions = df.to_dict('records')
    else:
        # CSV 格式
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_questions and i >= max_questions:
                    break
                questions.append(row)

    return questions


def run_rag_answers(questions: list, agent: RAGAgent, workers: int = 4):
    """批量生成RAG回答"""
    results = []
    lock = threading.Lock()

    print(f"\n{Colors.YELLOW}Generating RAG answers...{Colors.RESET}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_question = {
            executor.submit(process_single_question, agent, q, lock): q
            for q in questions
        }

        with tqdm(total=len(questions), desc="RAG Answering") as pbar:
            for future in as_completed(future_to_question):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)

    return results


def run_evaluations(rag_results: list, config: dict, workers: int = 4):
    """批量评估"""
    evaluations = []

    print(f"\n{Colors.YELLOW}Evaluating answers...{Colors.RESET}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_result = {
            executor.submit(evaluate_single_question, r, config): r
            for r in rag_results
        }

        with tqdm(total=len(rag_results), desc="Evaluating") as pbar:
            for future in as_completed(future_to_result):
                evaluation = future.result()
                if evaluation:
                    evaluations.append(evaluation)
                pbar.update(1)

    return evaluations


def save_results(evaluations: list, output_dir: Path, split_name: str, timestamp: str):
    """保存结果到带时间戳的文件"""
    # 在output_dir下创建带时间戳的子文件夹
    timestamped_dir = output_dir / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)

    # 保存文件名只用split_name
    output_file = timestamped_dir / f"{split_name}.csv"

    if not evaluations:
        print(f"{Colors.YELLOW}No results to save{Colors.RESET}")
        return None

    # 保存为CSV
    fieldnames = list(evaluations[0].keys())

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluations)

    print(f"{Colors.GREEN}✓ Results saved to {output_file}{Colors.RESET}")
    return output_file


def calculate_statistics(evaluations: list):
    """计算统计信息"""
    if not evaluations:
        return {}

    total = len(evaluations)

    # 计算各维度平均分
    avg_source = sum(e['source_accuracy_score'] for e in evaluations) / total
    avg_content = sum(e['content_accuracy_score'] for e in evaluations) / total
    avg_completeness = sum(e['completeness_score'] for e in evaluations) / total
    avg_relevance = sum(e['relevance_score'] for e in evaluations) / total
    avg_final = sum(e['final_score'] for e in evaluations) / total

    # 计算通过率（final_score >= 6.0）
    pass_count = sum(1 for e in evaluations if e['final_score'] >= 6.0)
    pass_rate = pass_count / total * 100

    return {
        'total': total,
        'avg_source_accuracy': avg_source,
        'avg_content_accuracy': avg_content,
        'avg_completeness': avg_completeness,
        'avg_relevance': avg_relevance,
        'avg_final_score': avg_final,
        'pass_count': pass_count,
        'pass_rate': pass_rate
    }


def print_statistics(split_name: str, stats: dict):
    """打印统计信息"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}Statistics for {split_name}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"Total Questions: {stats['total']}")
    print(f"Average Scores:")
    print(f"  Source Accuracy: {stats['avg_source_accuracy']:.2f}/10")
    print(f"  Content Accuracy: {stats['avg_content_accuracy']:.2f}/10")
    print(f"  Completeness: {stats['avg_completeness']:.2f}/10")
    print(f"  Relevance: {stats['avg_relevance']:.2f}/10")
    print(f"  {Colors.GREEN}Final Score: {stats['avg_final_score']:.2f}/10{Colors.RESET}")
    print(f"Pass Rate (≥6.0): {Colors.GREEN}{stats['pass_rate']:.1f}%{Colors.RESET} ({stats['pass_count']}/{stats['total']})")


def process_single_split(
    split_name: str,
    data_file: Path,
    agent: RAGAgent,
    config: dict,
    max_questions: int = None,
    workers: int = 4,
    output_dir: Path = None,
    enable_visualization: bool = True,
    timestamp: str = None
):
    """处理单个split的完整流程"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Processing Split: {split_name}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    try:
        # 1. 加载问题
        print(f"\n{Colors.YELLOW}[1/5] Loading questions...{Colors.RESET}")
        questions = load_questions_from_file(data_file, max_questions)
        print(f"{Colors.GREEN}✓ Loaded {len(questions)} questions{Colors.RESET}")

        if max_questions:
            print(f"{Colors.CYAN}  (Limited to first {max_questions} questions as configured){Colors.RESET}")

        # 2. 生成RAG回答
        print(f"\n{Colors.YELLOW}[2/5] Generating RAG answers...{Colors.RESET}")
        rag_results = run_rag_answers(questions, agent, workers)
        print(f"{Colors.GREEN}✓ Generated {len(rag_results)} answers{Colors.RESET}")

        # 3. 评估
        print(f"\n{Colors.YELLOW}[3/5] Evaluating answers...{Colors.RESET}")
        evaluations = run_evaluations(rag_results, config, workers)
        print(f"{Colors.GREEN}✓ Completed {len(evaluations)} evaluations{Colors.RESET}")

        # 4. 保存结果
        print(f"\n{Colors.YELLOW}[4/5] Saving results...{Colors.RESET}")
        output_file = save_results(evaluations, output_dir, split_name, timestamp)

        # 计算并显示统计信息
        stats = calculate_statistics(evaluations)
        print_statistics(split_name, stats)

        # 5. 生成可视化图表
        if enable_visualization and output_file:
            print(f"\n{Colors.YELLOW}[5/5] Generating visualizations...{Colors.RESET}")
            # 可视化结果保存在与CSV同目录下的 visualizations/{split_name}/ 下
            timestamped_dir = output_dir / timestamp
            viz_output_dir = timestamped_dir / "visualizations" / split_name
            visualize_results(output_file, viz_output_dir, split_name)

        return {
            'split_name': split_name,
            'success': True,
            'stats': stats,
            'output_file': output_file
        }

    except Exception as e:
        print(f"{Colors.RED}✗ Error processing {split_name}: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return {
            'split_name': split_name,
            'success': False,
            'error': str(e)
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run complete benchmark evaluation")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"{Colors.RED}Config file not found: {config_path}{Colors.RESET}")
        return

    config = load_config(config_path)

    # 获取benchmark配置
    benchmark_config = config.get('benchmark', {})
    splits_config = benchmark_config.get('splits', 'all')
    max_questions = benchmark_config.get('max_questions_per_split', None)
    enable_visualization = benchmark_config.get('enable_visualization', True)

    # 处理max_questions: null 或 -1 表示无限制
    if max_questions == -1:
        max_questions = None

    # 确定要运行的splits
    if splits_config == 'all':
        splits_to_run = ALL_SPLITS
    else:
        if splits_config not in ALL_SPLITS:
            print(f"{Colors.RED}Invalid split name: {splits_config}{Colors.RESET}")
            print(f"Available splits: {', '.join(ALL_SPLITS)}")
            return
        splits_to_run = [splits_config]

    # 获取其他配置
    workers = config.get('judge_evaluation', {}).get('workers', 4)
    output_dir = Path(config.get('judge_evaluation', {}).get('output', {}).get('directory', 'evaluation_results'))

    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Benchmark Evaluation{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"Repository: {REPO_ID}")
    print(f"Config: {config_path}")
    print(f"Splits to run: {', '.join(splits_to_run)}")
    if max_questions:
        print(f"Max questions per split: {max_questions}")
    else:
        print(f"Max questions per split: ALL")
    print(f"Workers: {workers}")
    print(f"Output directory: {output_dir}")
    print(f"Visualization: {'Enabled' if enable_visualization else 'Disabled'}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    # 1. 检查数据集是否存在
    script_dir = Path(__file__).parent
    qa_data_dir = script_dir / "QA_data"

    available_files = check_datasets(qa_data_dir, splits_to_run)

    if not available_files:
        print(f"\n{Colors.RED}No datasets available. Exiting.{Colors.RESET}")
        return

    # 2. 初始化RAG Agent
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.YELLOW}Initializing RAG Agent...{Colors.RESET}")
    try:
        agent = RAGAgent()
        print(f"{Colors.GREEN}✓ RAG Agent initialized{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to initialize RAG Agent: {e}{Colors.RESET}")
        return

    # 3. 生成本次运行的时间戳
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{Colors.CYAN}Run timestamp: {run_timestamp}{Colors.RESET}")

    # 4. 处理每个split
    all_results = []

    for split_name, data_file in available_files.items():
        result = process_single_split(
            split_name=split_name,
            data_file=data_file,
            agent=agent,
            config=config,
            max_questions=max_questions,
            workers=workers,
            output_dir=output_dir,
            enable_visualization=enable_visualization,
            timestamp=run_timestamp
        )
        all_results.append(result)

    # 5. 总结
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Overall Summary{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    success_count = sum(1 for r in all_results if r['success'])
    total_count = len(all_results)

    for result in all_results:
        if result['success']:
            stats = result['stats']
            status = f"{Colors.GREEN}✓ SUCCESS{Colors.RESET}"
            print(f"{status}  {result['split_name']}: Avg={stats['avg_final_score']:.2f}, Pass={stats['pass_rate']:.1f}%")
        else:
            status = f"{Colors.RED}✗ FAILED{Colors.RESET}"
            print(f"{status}  {result['split_name']}: {result.get('error', 'Unknown error')}")

    print(f"\n{Colors.CYAN}Total: {success_count}/{total_count} splits completed successfully{Colors.RESET}")

    if success_count == total_count:
        print(f"\n{Colors.GREEN}{'='*60}{Colors.RESET}")
        print(f"{Colors.GREEN}All benchmarks completed successfully!{Colors.RESET}")
        print(f"{Colors.GREEN}{'='*60}{Colors.RESET}\n")
    else:
        print(f"\n{Colors.YELLOW}{'='*60}{Colors.RESET}")
        print(f"{Colors.YELLOW}Some benchmarks failed. Please check the errors above.{Colors.RESET}")
        print(f"{Colors.YELLOW}{'='*60}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
