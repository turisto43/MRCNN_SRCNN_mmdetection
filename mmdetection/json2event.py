import json
from tensorboardX import SummaryWriter

# 读取 jsonl 文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# 写入 TensorBoard
def write_to_tensorboard(data, log_dir):
    # 创建一个 summary writer
    writer = SummaryWriter(log_dir)
    
    for entry in data:
        epoch = entry['epoch']
        mAP = entry['loss']
        # 记录到 TensorBoard
        writer.add_scalar('loss', mAP, epoch)
    
    writer.close()

# 主函数
def main(jsonl_file_path, log_dir):
    # 读取 jsonl 数据
    data = read_jsonl(jsonl_file_path)
    # 写入 TensorBoard
    write_to_tensorboard(data, log_dir)
    print(f"Data has been written to TensorBoard at {log_dir}")

# 示例用法
jsonl_file_path = '/root/model_saved/neuralnetwork_hw2/mmdetection/mmdetection/vali_loss_wyq_0529_frcnn.jsonl'
log_dir = 'valid_loss_logs_frcnn_wyq'
main(jsonl_file_path, log_dir)
