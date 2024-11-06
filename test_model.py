import  torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms

from model import Identify
#准备数据
models=torch.load('./identify_model9',map_location=torch.device('cpu'))
models.eval()
img_path='./test_data/img_4.png'
img=Image.open(img_path)
img = img.convert('L')
#查看图片和模型结构
print(img)
print(models)
#改变图片格式后输入模型进行判断
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((28,28)),#一定要调整尺寸
                                          torchvision.transforms.ToTensor()])
img=transform(img)
img=torch.reshape(img,(1,1,28,28))
print(img.shape)
#print(img.shape)


with torch.no_grad():
    output=models(img).argmax(1)
print(output)
