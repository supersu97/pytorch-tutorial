import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO 自定义数据集，与 torch.utils.data.DataLoader 兼容。"""
    def __init__(self, root, json, vocab, transform=None):
        """设置图像、字幕和词汇表包装器的路径。
        
        参数:
            root: 图像目录。
            json: coco 标注文件路径。
            vocab: 词汇表包装器。
            transform: 图像转换器。
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """返回一个数据对（图像和字幕）。"""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # 将字幕（字符串）转换为词 ID。
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """从元组列表 (image, caption) 创建 mini-batch 张量。
    
    我们应该构建自定义的 collate_fn，而不是使用默认的 collate_fn，
    因为默认的 collate_fn 不支持合并字幕（包括填充）。

    参数:
        data: 元组列表 (image, caption)。
            - image: 形状为 (3, 256, 256) 的 torch 张量。
            - caption: 形状为 (?) 的 torch 张量；可变长度。

    返回:
        images: 形状为 (batch_size, 3, 256, 256) 的 torch 张量。
        targets: 形状为 (batch_size, padded_length) 的 torch 张量。
        lengths: 列表；每个填充字幕的有效长度。
    """
    # 按字幕长度（降序）对数据列表进行排序。
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # 合并图像（从 3D 张量元组到 4D 张量）。
    images = torch.stack(images, 0)

    # 合并字幕（从 1D 张量元组到 2D 张量）。
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """返回用于自定义 coco 数据集的 torch.utils.data.DataLoader。"""
    # COCO 字幕数据集
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # COCO 数据集的数据加载器
    # 每次迭代将返回 (images, captions, lengths)。
    # images: 形状为 (batch_size, 3, 224, 224) 的张量。
    # captions: 形状为 (batch_size, padded_length) 的张量。
    # lengths: 一个列表，表示每个字幕的有效长度。长度为 (batch_size)。
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
