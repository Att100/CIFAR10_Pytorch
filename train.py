# import utils
from utils import *


# argparser
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size (default: 256)')
parser.add_argument("--epoch", type=int, default=150,
                        help='train epoch (default: 150)')
parser.add_argument("--lr", type=float, default=0.1,
                        help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='SGD weight decay (default: 5e-4)')
parser.add_argument('--dataset', type=str, default='./dataset',
                        help='dataset root path (default: ./dataset)')
parser.add_argument('--logdir', type=str, default='./log',
                        help='tensorboard log dir (default: ./log)')
parser.add_argument('--model', type=str, default='VGG16',
                        help='Model name (default: VGG16)')
parser.add_argument('--save-path', type=str, default='./weights',
                        help='model save path (default: ./weights)')
args = parser.parse_args()


# configuration (you can change the default value by given paramaters)
batchsize = args.batch_size
epochs = args.epoch
learning_rate = args.lr  # start lr
momentum = args.momentum
weightDecay = args.weight_decay
dataset = args.dataset
modelName = args.model
savePath = args.save_path


# extra configuration
worker_num = 2
enableWriter = True
Device = torch.device("cuda")
lr_decay = [80, 100, 120, 140]
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == "__main__":
    # writer
    # If you choose to use the writer to record the training, you can retreive 
    # the data by download the csv file from tensorboard afterwards and use 
    # the function 'retrieveCurve' and 'graphCurve'(in utils.py) to show the 
    # result.
    if enableWriter:
        writer = SummaryWriter(log_dir='./log')
        def updateWriter(writer, loss, acc, val_loss, val_acc, n_iter):
            writer.add_scalar('train/loss', loss, n_iter)
            writer.add_scalar('train/acc', acc, n_iter)
            writer.add_scalar('val/loss', val_loss, n_iter)
            writer.add_scalar('val/acc', val_acc, n_iter)


    # dataloader
    train_loader, test_loader = getDataloader(root=dataset, batchsize=batchsize, worker=worker_num)


    # model select
    # If you want to change the details of the model, please go to the models folder 
    # to edit the model.
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
    else:
        raise Exception("model not found")


    # optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                    momentum=momentum, weight_decay=weightDecay)
    loss_func = torch.nn.CrossEntropyLoss()


    # use testset to evaluate
    def evaluateModel():
        total_acc = 0.0
        total_loss = 0.0
        step = 0
        with torch.no_grad():
            for i, (image, label) in enumerate(test_loader):
                image, label = image.to(Device), label.to(Device)
                if modelName != 'GoogLeNet':
                    # Others
                    output = model(image)
                else:
                    # GoogLeNet
                    output, _1, _2 = model(image)
                _, pred = torch.max(output.data, 1)
                correct = (pred == label).sum()
                loss = loss_func(output, label.long())
                total_loss += loss.item()
                total_acc += correct.cpu().numpy()
                step += 1
            total_loss /= step
            total_acc /= 10000
            return total_acc, total_loss


    # train model
    model.train()
    steps = len(train_loader)
    bar = ProgressBar(maxStep=steps)   # initialize progress bar
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for i, (image, label) in enumerate(train_loader):
            image, label = image.to(Device), label.to(Device)  # step 1.
            optimizer.zero_grad()  # step 2.
            if modelName != 'GoogLeNet':
                # Others
                output = model(image)  # step 3.
                loss = loss_func(output, label.long())  # step 4.
            else:
                # GoogLeNet
                output, auxOut_1, auxOut_2 = model(image)
                out_loss = loss_func(output, label.long())
                aux_1_loss = loss_func(auxOut_1, label.long())
                aux_2_loss = loss_func(auxOut_2, label.long())
                loss = out_loss + aux_1_loss*0.3 + aux_2_loss*0.3
            _, pred = torch.max(output.data, 1)
            correct = (pred == label).sum()
            loss.backward()  # step 5.
            optimizer.step()  # step 6.
            epoch_loss += loss.item()
            epoch_acc += correct.cpu().numpy()
            bar.updateBar(step=i+1, headData={'epoch':epoch+1}, 
                            endData={'loss':round(loss.item(), 4)}, keep=True)
        bar.updateBar(step=i+1, headData={'epoch':epoch+1}, endData={'testing':'...'}, keep=True)
        epoch_loss /= steps
        epoch_acc /= 50000
        val_acc, val_loss = evaluateModel()
        if enableWriter:
            updateWriter(writer, epoch_loss, epoch_acc, val_loss, val_acc, epoch + 1)
        # apply learning rate decay
        if epoch in lr_decay:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1
        bar.updateBar(step=i+1, headData={'epoch':epoch+1}, endData={'loss':round(epoch_loss, 4),
                        'accuracy':round(epoch_acc, 2), 'val_loss':round(val_loss, 4), 
                        'val_acc':round(val_acc, 4)})
        bar.resetBar()
    torch.save(model, savePath + "/" + modelName + "_cifar10.pt")



