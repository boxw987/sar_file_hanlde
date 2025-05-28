import subprocess
import sys
import os
from datetime import datetime
import argparse

def setup_encoding():
    """设置系统编码"""
    if sys.platform.startswith('win'):
        os.system('chcp 65001')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

def run_training(command, task_num, total_tasks, log_dir, root_dir):
    """执行训练命令并返回执行状态"""
    try:
        # 创建日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_log_task{task_num}_{timestamp}.txt")
        
        # 打印分隔线和任务信息
        task_info = f"""
{'='*80}
开始执行任务 {task_num}/{total_tasks}
命令: {command}
时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}
"""
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 写入日志并打印到控制台
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(task_info)
        print(task_info)

        # 执行命令并实时获取输出
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            cwd=root_dir  # 设置工作目录为项目根目录
        )

        # 实时读取并同时写入日志文件和控制台
        with open(log_file, 'a', encoding='utf-8') as f:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    f.write(output)
                    f.flush()

        return_code = process.poll()
        if return_code == 0:
            success_msg = f"\n任务 {task_num}/{total_tasks} 完成成功!\n"
            print(success_msg)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(success_msg)
            return True
        else:
            error_msg = f"\n任务 {task_num}/{total_tasks} 执行失败! 返回码: {return_code}\n"
            print(error_msg)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg)
            return False

    except Exception as e:
        error_msg = f"\n任务执行出错: {str(e)}\n"
        print(error_msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(error_msg)
        return False

def main():
    # 设置编码
    setup_encoding()
    
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="自动训练脚本")
    parser.add_argument('--log_name', type=str, default=None, help='日志文件夹的自定义名称')
    args = parser.parse_args()
    
    # 定义训练序列
    commands = [
        "python train.py",
        "python val.py --data sar_data/HRSID/yolo_style/ships.yaml --save-conf --task test"
    ]

    total_tasks = len(commands)

    # 创建日志目录，以第一个命令执行的时间命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_name = args.log_name if args.log_name else f"training_logs_{timestamp}"
    log_dir = os.path.join(root_dir, "log", log_dir_name)
    os.makedirs(log_dir, exist_ok=True)

    # 顺序执行训练任务
    for i, cmd in enumerate(commands, 1):
        success = run_training(cmd, i, total_tasks, log_dir, root_dir)
        
        if not success:
            print(f"\n训练序列在第{i}个任务时失败，终止后续训练")
            sys.exit(1)

    print("\n所有训练任务已完成!")

if __name__ == "__main__":
    main()
