#!/usr/bin/env python3
"""
测试脚本：从每个数据集中挑选一个问题进行测试
测试完整的pipeline：下载数据 -> RAG回答 -> 评分
"""

import sys
from pathlib import Path
from datasets import load_dataset
import csv
import yaml

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_agent import RAGAgent
from step4_rag_answer import process_single_question
from step5_judge_evaluation import (
    evaluate_single_question,
    build_judge_prompt,
    call_llm,
    parse_judge_response
)
import threading


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RESET = '\033[0m'


def load_config(config_path: Path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_single_split(repo_id: str, split_name: str, agent: RAGAgent, config: dict):
    """测试单个split的第一个问题"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Testing split: {split_name}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    try:
        # # 1. 下载数据集
        # print(f"{Colors.YELLOW}[1/3] Downloading dataset...{Colors.RESET}")
        # dataset = load_dataset(repo_id, split=split_name)
        # print(f"{Colors.GREEN}✓ Downloaded {len(dataset)} questions{Colors.RESET}")
        
        # 1. 直接读取本地文件
        dataset = load_dataset(
            "parquet",
            data_files=f"./QA_data/data/{split_name}-00000-of-00001.parquet"
        )["train"]
        # 读取完毕

        # 2. 获取第一个问题
        first_question = dataset[0]
        question_dict = {
            'query': first_question['query'],
            'standard_answer': first_question['standard_answer'],
            'course': first_question['course'],
            'material': first_question['material'],
            'page_range': first_question['page_range'],
            'question_type': first_question['question_type']
        }

        print(f"\n{Colors.CYAN}Question:{Colors.RESET}")
        print(f"  {question_dict['query'][:100]}...")

        # 3. 生成RAG回答
        print(f"\n{Colors.YELLOW}[2/3] Generating RAG answer...{Colors.RESET}")
        lock = threading.Lock()
        result = process_single_question(agent, question_dict, lock)

        if result:
            agent_answer = result['agent_answer']
            print(f"{Colors.GREEN}✓ RAG Answer generated{Colors.RESET}")
            print(f"\n{Colors.CYAN}RAG Answer:{Colors.RESET}")
            print(f"  {agent_answer[:200]}...")

            # 4. 评分
            print(f"\n{Colors.YELLOW}[3/3] Evaluating answer...{Colors.RESET}")
            evaluation = evaluate_single_question(result, config)

            print(f"{Colors.GREEN}✓ Evaluation completed{Colors.RESET}")
            print(f"\n{Colors.CYAN}Scores:{Colors.RESET}")
            print(f"  Source Accuracy: {evaluation['source_accuracy_score']:.2f}/10")
            print(f"  Content Accuracy: {evaluation['content_accuracy_score']:.2f}/10")
            print(f"  Completeness: {evaluation['completeness_score']:.2f}/10")
            print(f"  Relevance: {evaluation['relevance_score']:.2f}/10")
            print(f"  {Colors.GREEN}Final Score: {evaluation['final_score']:.2f}/10{Colors.RESET}")

            return True
        else:
            print(f"{Colors.RED}✗ Failed to generate RAG answer{Colors.RESET}")
            return False

    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.RESET}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test pipeline with one question from each split")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Test only this split (optional, default: test all splits)"
    )

    args = parser.parse_args()

    # 硬编码 Hugging Face 仓库地址
    repo_id = "HEHUA2005/rag-benchmark-qa-dataset"

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"{Colors.RED}Config file not found: {config_path}{Colors.RESET}")
        return

    config = load_config(config_path)

    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Pipeline Test{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"Repository: {repo_id}")
    print(f"Config: {config_path}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    # 初始化RAG Agent
    print(f"\n{Colors.YELLOW}Initializing RAG Agent...{Colors.RESET}")
    try:
        agent = RAGAgent()
        print(f"{Colors.GREEN}✓ RAG Agent initialized{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to initialize RAG Agent: {e}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Hint: Make sure you have implemented RAGAgent in rag_agent.py{Colors.RESET}")
        return

    # 所有splits
    all_splits = [
        "Mao_Zedong_Thought",
        "Principles_of_Marxism",
        "Outline_of_Modern_and_Contemporary_Chinese_History",
        "Ideological_Morality_and_Legal_System",
        "An_Introduction_to_Xi_Jinping_Thought_on_Socialism_with_Chinese_Characteristics_for_a_New_Era"
    ]

    # 如果指定了split，只测试那个
    if args.split:
        splits_to_test = [args.split]
    else:
        splits_to_test = all_splits

    # 测试每个split
    results = {}
    for split_name in splits_to_test:
        success = test_single_split(repo_id, split_name, agent, config)
        results[split_name] = success

    # 总结
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Test Summary{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for split_name, success in results.items():
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if success else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        print(f"{status}  {split_name}")

    print(f"\n{Colors.CYAN}Total: {success_count}/{total_count} passed{Colors.RESET}")

    if success_count == total_count:
        print(f"\n{Colors.GREEN}{'='*60}{Colors.RESET}")
        print(f"{Colors.GREEN}All tests passed! Pipeline is working correctly.{Colors.RESET}")
        print(f"{Colors.GREEN}{'='*60}{Colors.RESET}\n")
    else:
        print(f"\n{Colors.YELLOW}{'='*60}{Colors.RESET}")
        print(f"{Colors.YELLOW}Some tests failed. Please check the errors above.{Colors.RESET}")
        print(f"{Colors.YELLOW}{'='*60}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
