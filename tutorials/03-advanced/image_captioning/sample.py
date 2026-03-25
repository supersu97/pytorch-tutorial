import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # 加载词汇表包装器
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # 构建模型
    encoder = EncoderCNN(args.embed_size).eval()  # 评估模式（batchnorm 使用移动平均/方差）
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 加载训练好的模型参数
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # 准备图像
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # 从图像生成字幕
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # 将词 ID 转换为词
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # 打印图像和生成的字幕
    print (sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='用于生成字幕的输入图像')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='训练好的编码器路径')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='训练好的解码器路径')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='词汇表包装器的路径')
    
    # 模型参数（应与 train.py 中的参数相同）
    parser.add_argument('--embed_size', type=int , default=256, help='词嵌入向量的维度')
    parser.add_argument('--hidden_size', type=int , default=512, help='LSTM 隐藏状态的维度')
    parser.add_argument('--num_layers', type=int , default=1, help='LSTM 的层数')
    args = parser.parse_args()
    main(args)
