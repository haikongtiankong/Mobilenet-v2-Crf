import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD
from image_producer import GridImageDataset  # noqa
from mobilenet_V2_crf_model import MobileNetV2
from tensorboardX import SummaryWriter



data_path_train = r'F:\大四\医学图像处理\data_set\NCRF数据\tile_重制\train'
data_path_valid = r'F:\大四\医学图像处理\data_set\NCRF数据\tile_重制\valid'
json_path_train = r'F:\大四\医学图像处理\data_set\NCRF数据\json2\train'
json_path_valid = r'F:\大四\医学图像处理\data_set\NCRF数据\json2\valid'
image_size = 768
patch_size = 256
crop_size = 224
batch_size_train = 5
batch_size_valid = batch_size_train
num_workers = 0
grid_size = 9
learning_rate = 1e-4
momentum = 0.9
epoch = 10
total_train_step=0
total_valid_step=0
device = torch.device("cuda:0")

#Preparing dataset
dataset_train = GridImageDataset(data_path_train,
                                 json_path_train,
                                 image_size,
                                 patch_size,
                                 crop_size=crop_size)
dataset_valid = GridImageDataset(data_path_valid,
                                 json_path_valid,
                                 image_size,
                                 patch_size,
                                 crop_size=crop_size)

#loading dataset
dataloader_train = DataLoader(dataset_train,
                              batch_size=batch_size_train,
                              num_workers=num_workers,
                              shuffle=True,
                              drop_last=True)
dataloader_valid = DataLoader(dataset_valid,
                              batch_size=batch_size_valid,
                              num_workers=num_workers,
                              drop_last=True)

model = MobileNetV2(num_classes=1, use_crf=True, num_nodes=grid_size)
#model = resnet18(num_nodes=grid_size, use_crf=False)
model.load_state_dict(torch.load(r"D:\CKPT\mobilev2crf\mobilev2crf_20.pth", map_location='cuda:0'))
#model.load_state_dict(torch.load("D:/CKPT/resnet18_base.ckpt")['state_dict'])
if torch.cuda.is_available():
    model = model.to(device)

loss_fn = BCEWithLogitsLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.to(device)

optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)

writer = SummaryWriter("logs-BUCHONG")
train_length = (batch_size_train * len(dataloader_train)) * grid_size
test_length = (batch_size_valid * len(dataloader_valid)) * grid_size


for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+21))


    total_train_correct = 0
    model.train()
    train_data_positive = 0  # the number of positive samples correctly classified
    train_data_negative = 0
    train_positive = 0  # the number of positive samples
    train_negative = 0
    for data in dataloader_train:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.to(device)
            targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = outputs.sigmoid()
        predicts = (probs >= 0.5).type(torch.FloatTensor).to(device)
        acc_data = (predicts == targets).sum().item()
        total_train_correct = total_train_correct + acc_data

        for x in range(batch_size_train):
            for y in range(grid_size):
                if targets[x][y] == torch.tensor(0):
                    train_negative = train_negative + 1
                    if targets[x][y] == predicts[x][y]:
                        train_data_negative = train_data_negative + 1
                elif targets[x][y] == torch.tensor(1):
                    train_positive = train_positive + 1
                    if targets[x][y] == predicts[x][y]:
                        train_data_positive = train_data_positive + 1

        total_train_step = total_train_step + 1
        if total_train_step % 20 == 1:
            print('训练次数:{}, Loss:{}'.format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    print("The accuracy of positive samples:{}".format(train_data_positive/train_positive))
    print("The accuracy of negative samples:{}".format(train_data_negative/train_negative))
    print("The overall accuracy:{}".format(total_train_correct/train_length))
    writer.add_scalar("train_accuracy", total_train_correct/train_length, i)

    #测试步骤开始
    model.eval()
    total_test_correct = 0
    total_test_loss = 0
    acc_data_positive = 0
    acc_data_negative = 0
    positive_number = 0
    negative_number = 0
    test_img_num = test_length / grid_size
    with torch.no_grad():
        for data in dataloader_valid:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.to(device)
                targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()

            probs = outputs.sigmoid()
            predicts = (probs >= 0.5).type(torch.FloatTensor).to(device)
            acc_data = (predicts == targets).sum().item()
            total_test_correct = total_test_correct + acc_data

            for x in range(batch_size_train):
                for y in range(grid_size):
                    if targets[x][y] == torch.tensor(0):
                        negative_number = negative_number + 1
                        if targets[x][y] == predicts[x][y]:
                            acc_data_negative = acc_data_negative + 1
                    elif targets[x][y] == torch.tensor(1):
                        positive_number = positive_number + 1
                        if targets[x][y] == predicts[x][y]:
                            acc_data_positive = acc_data_positive + 1
        fpr = (negative_number - acc_data_negative) / negative_number
        fps = fpr * negative_number / test_img_num

    print('The accuracy of positive samples:{}'.format(acc_data_positive / positive_number))
    print('The accuracy of negative samples:{}'.format(acc_data_negative / negative_number))
    print('sensitivity:{}'.format(acc_data_positive / positive_number))
    print('the average number of false positive regions of each sample:{}'.format(fps))
    print('The loss on test set:{}'.format(total_test_loss))
    print("The accuracy of test set:{}".format(total_test_correct / test_length))
    writer.add_scalar("test_loss", total_test_loss, total_valid_step)
    writer.add_scalar("test_accuracy", total_test_correct / test_length, total_valid_step)
    total_valid_step = total_valid_step + 1

    #torch.save(model, "moblienet_crf_{}.pth".format(i))
    torch.save(model.state_dict(), r"D:\CKPT\mobilev2crf\mobilev2crf_{}.pth".format(i+21))
    print("model saved")

writer.close()
