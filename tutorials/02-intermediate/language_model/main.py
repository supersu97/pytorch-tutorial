# 部分代码参考自以下链接。
# https://github.com/pytorch/examples/tree/master/word_language_model 
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000     # 采样词的数量
batch_size = 20
seq_length = 30
learning_rate = 0.002

# 加载 "Penn Treebank" 数据集
corpus = Corpus()
ids = corpus.get_data('data/train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length


# 基于 RNN 的语言模型
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h):
        # 将词 ID 嵌入到向量中
        x = self.embed(x)
        
        # 前向传播 LSTM
        out, (h, c) = self.lstm(x, h)
        
        # 将输出重塑为 (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # 解码所有时间步的隐藏状态
        out = self.linear(out)
        return out, (h, c)

model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 截断反向传播
def detach(states):
    return [state.detach() for state in states] 

# 训练模型
for epoch in range(num_epochs):
    # 设置初始隐藏状态和单元状态
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    
    for i in range(0, ids.size(1) - seq_length, seq_length):
        # 获取 mini-batch 输入和目标
        inputs = ids[:, i:i+seq_length].to(device)
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)
        
        # 前向传播
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // seq_length
        if step % 100 == 0:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                   .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

# 测试模型
with torch.no_grad():
    with open('sample.txt', 'w') as f:
        # 设置初始隐藏状态和单元状态
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # 随机选择一个词 ID
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # 前向传播 RNN 
            output, state = model(input, state)

            # 采样一个词 ID
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # 用采样的词 ID 填充输入，用于下一个时间步
            input.fill_(word_id)

            # 写入文件
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))

# 保存模型检查点
torch.save(model.state_dict(), 'model.ckpt')
