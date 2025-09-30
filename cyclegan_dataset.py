"""
CycleGAN專用數據載入器
支持兩個獨立的數據集（domain A和domain B）
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random

class SingleDomainDataset(Dataset):
    """單一領域數據集，用於CycleGAN"""
    
    def __init__(self, image_dir, transform=None, max_images=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # 支持的圖像格式
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # 收集所有圖像文件（包括子資料夾）
        self.images = []
        if os.path.exists(image_dir):
            # 遞歸搜索所有子資料夾
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if file.lower().endswith(self.image_extensions):
                        self.images.append(os.path.join(root, file))
        
        # 限制圖像數量
        if max_images is not None and len(self.images) > max_images:
            self.images = self.images[:max_images]
            
        print(f"載入 {len(self.images)} 張圖像從 {image_dir}")
        
        assert len(self.images) > 0, f"在 {image_dir} 中沒有找到圖像"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx % len(self.images)]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"無法載入圖像 {img_path}: {e}")
            # 返回隨機圖像
            idx = random.randint(0, len(self.images) - 1)
            image = Image.open(self.images[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

def get_cyclegan_transforms(image_size=256, is_train=True):
    """獲得CycleGAN訓練用的變換"""
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),  # 稍大一些
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    return transform

def create_cyclegan_dataloaders(domain_A_dir, domain_B_dir, batch_size=4, 
                               image_size=256, num_workers=4, is_train=True,
                               max_images_A=None, max_images_B=None):
    """創建CycleGAN數據載入器"""
    
    # 獲得變換
    transform = get_cyclegan_transforms(image_size, is_train)
    
    # 創建數據集
    dataset_A = SingleDomainDataset(domain_A_dir, transform, max_images_A)
    dataset_B = SingleDomainDataset(domain_B_dir, transform, max_images_B)
    
    # 創建數據載入器
    dataloader_A = DataLoader(
        dataset_A,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 確保每個batch大小一致
    )
    
    dataloader_B = DataLoader(
        dataset_B,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader_A, dataloader_B

def create_infinite_dataloader(dataloader):
    """創建無限循環的數據載入器"""
    while True:
        for batch in dataloader:
            yield batch

class CycleGANDataset(Dataset):
    """CycleGAN配對數據集 - 同步返回兩個域的圖像"""
    
    def __init__(self, root_A, root_B, transform=None, max_images=None, max_images_A=None, max_images_B=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform
        
        # 支持的圖像格式
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # 收集所有圖像文件
        self.images_A = self._collect_images(root_A)
        self.images_B = self._collect_images(root_B)
        
        # 限制圖像數量 - 支持分別限制和統一限制
        if max_images_A is not None:
            if len(self.images_A) > max_images_A:
                self.images_A = self.images_A[:max_images_A]
        elif max_images is not None:
            if len(self.images_A) > max_images:
                self.images_A = self.images_A[:max_images]
                
        if max_images_B is not None:
            if len(self.images_B) > max_images_B:
                self.images_B = self.images_B[:max_images_B]
        elif max_images is not None:
            if len(self.images_B) > max_images:
                self.images_B = self.images_B[:max_images]
        
        print(f"載入 Domain A: {len(self.images_A)} 張圖像從 {root_A}")
        print(f"載入 Domain B: {len(self.images_B)} 張圖像從 {root_B}")
        
        assert len(self.images_A) > 0, f"在 {root_A} 中沒有找到圖像"
        assert len(self.images_B) > 0, f"在 {root_B} 中沒有找到圖像"
    
    def _collect_images(self, image_dir):
        """收集目錄中的所有圖像"""
        images = []
        if os.path.exists(image_dir):
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if file.lower().endswith(self.image_extensions):
                        images.append(os.path.join(root, file))
        return sorted(images)  # 排序以確保一致性
    
    def __len__(self):
        return max(len(self.images_A), len(self.images_B))
    
    def __getitem__(self, idx):
        # 使用模運算確保索引有效
        idx_A = idx % len(self.images_A)
        idx_B = idx % len(self.images_B)
        
        try:
            image_A = Image.open(self.images_A[idx_A]).convert('RGB')
            image_B = Image.open(self.images_B[idx_B]).convert('RGB')
        except Exception as e:
            print(f"無法載入圖像: {e}")
            # 使用隨機索引重試
            idx_A = random.randint(0, len(self.images_A) - 1)
            idx_B = random.randint(0, len(self.images_B) - 1)
            image_A = Image.open(self.images_A[idx_A]).convert('RGB')
            image_B = Image.open(self.images_B[idx_B]).convert('RGB')
        
        if self.transform:
            # 使用相同的隨機種子確保一致的隨機變換
            seed = random.randint(0, 2**32 - 1)
            
            random.seed(seed)
            torch.manual_seed(seed)
            image_A = self.transform(image_A)
            
            random.seed(seed)
            torch.manual_seed(seed)
            image_B = self.transform(image_B)
            
        return image_A, image_B