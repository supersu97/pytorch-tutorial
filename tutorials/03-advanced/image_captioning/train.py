import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # 创建模型目录
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # 图像预处理，对预训练的 resnet 进行归一化
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # 加载词汇表包装器
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # 构建数据加载器
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # 构建模型
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # 训练模型
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # 设置 mini-batch 数据集
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # 前向、反向传播和优化
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印日志信息
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # 保存模型检查点
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='保存训练模型的路径')
    parser.add_argument('--crop_size', type=int, default=224 , help='随机裁剪图像的大小')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='词汇表包装器的路径')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='调整大小后的图像目录')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='训练标注 json 文件的路径')
    parser.add_argument('--log_step', type=int , default=10, help='打印日志信息的步长')
    parser.add_argument('--save_step', type=int , default=1000, help='保存训练模型的步长')
    
    # 模型参数
    parser.add_argument('--embed_size', type=int , default=256, help='词嵌入向量的维度')
    parser.add_argument('--hidden_size', type=int , default=512, help='LSTM 隐藏状态的维度')
    parser.add_argument('--num_layers', type=int , default=1, help='LSTM 的层数')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
