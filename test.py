# import utils
from utils import *


# argparser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGG16',
                        help='model name (default: VGG16)')
parser.add_argument('--batch-size', type=int, default=256,
                        help='test input batch size (default: 256)')
parser.add_argument('--dataset', type=str, default='./dataset',
                        help='dataset root path (default: ./dataset)')
parser.add_argument('--mode', type=str, default='res',
                        help='res or img, res is for test accuracy while img is for \
                        test a single img (default: res)')
parser.add_argument('--weights-path', type=str, default='./weights/VGG16_cifar10.pt',
                        help='saved model path (default: ./weights/VGG16_cifar10.pt)')
parser.add_argument('--img', type=str, default="./test.jpeg",
                        help="test img path (default: ./test.jpeg)")
args = parser.parse_args()


#configuration
modelName = args.model
batchsize = args.batch_size
dataset = args.dataset
imgPath = args.img
if args.mode == 'res':
    mode = False
elif args.mode == 'img':
    mode = True
else:
    mode = None
    raise Exception('false mode input')
weightsPath = args.weights_path
Device = torch.device("cuda")


# model select
if modelName == 'AlexNet':
    from models.AlexNet import *
    model = AlexNet()
elif modelName == "SqueezeNet":
    from models.SqueezeNet import *
    model = SqueezeNet()
elif modelName == "VGG16":
    from models.VGG16 import *
    model = VGG16()
elif modelName == "GoogLeNet":
    from models.GoogLeNet import *
    model = GoogLeNet()
elif modelName == "ResNet18":
    from models.ResNet18 import *
    model = ResNet18()
elif modelName == "DenseNet121":
    from models.DenseNet import *
    model = DenseNet121()
elif modelName == "MobileNet":
    from models.MobileNet import *
    model = MobileNet()
else:
    model = None

if model is not None:
    model = model.to(Device)


# test
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if not mode:
    # test on testset
    train_loader, test_loader = getDataloader(root=dataset, 
                                batchsize=batchsize, worker=2)
    load_model = torch.load(weightsPath)
    load_model = load_model.to(Device)
    load_model.eval()
    total_acc = 0.0
    steps = len(test_loader)
    bar = ProgressBar(maxStep=steps)
    for i, (image, label) in enumerate(test_loader):
        image, label = image.to(Device), label.to(Device)
        output = load_model(image)
        _, pred = torch.max(output.data, 1)
        correct = (pred == label).sum()
        total_acc += correct.cpu().numpy()
        bar.updateBar(i+1, headData={'testset':''}, endData={'testing':'...'}, keep=True)
    total_acc /= 10000
    bar.updateBar(i+1, headData={'testset':''}, endData={'accuracy':round(total_acc, 3)})
    print("test accuracy: ", total_acc)
else:
    # test on single image
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    image = transform_test(img).float().unsqueeze(0)
    load_model = torch.load(weightsPath)
    load_model = load_model.to(Device)
    image = image.to(Device)
    load_model.eval()
    out = load_model(image)
    _, pred = torch.max(out.data, 1)
    print("the prediction of " + imgPath + " is " + classes[pred.cpu()[0]])
    