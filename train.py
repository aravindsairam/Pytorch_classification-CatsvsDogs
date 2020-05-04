from dataset_class import catsvsdogsTrain
from dispatcher import MODELS_DISPATCH
import torch.optim as optim
import torch.nn as nn
import ast

MODEL = MODELS_DISPATCH[os.environ.get("MODEL")]

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))

TRAIN_FOLDS = ast.literal_eval(os.environ.get("TRAIN_FOLDS"))
VAL_FOLDS = ast.literal_eval(os.environ.get("VAL_FOLDS"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
EPOCHS = int(os.environ.get("EPOCHS"))
LEARNING_RATE = os.environ.get("EPOCHS")


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    
    model = MODELS_DISPATCH[MODEL](pretrain = True)
    
    
    model.to(DEVICE)
    
    train_data = catsvsdogsTrain(folds = TRAIN_FOLDS,
                              img_height = IMG_HEIGHT,
                              img_width = IMG_WIDTH,
                              mean= MODEL_MEAN,
                              std = MODEL_STD)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE,
                                         shuffle  = True, num_workers = 0)
    
    val_data = catsvsdogsTrain(folds = VAL_FOLDS,
                              img_height = IMG_HEIGHT,
                              img_width = IMG_WIDTH,
                              mean= MODEL_MEAN,
                              std = MODEL_STD)
    
    valloader = torch.utils.data.DataLoader(val_data, batch_size = BATCH_SIZE,
                                         shuffle  = False, num_workers = 0)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    for epoch in range(EPOCHS):
        print(f"At epoch {epoch+1}:")
        for phase in ['train', 'val']:
            running_loss = 0.0
            auc = 0.0
            if phase == 'train': 
                model.train()
                loader = trainloader
            else:
                model.eval()
                loader = valloader
            count = 0
            for data in loader:
                count +=1
                inputs  = data["images"].to(DEVICE)
                labels = data["targets"].to(DEVICE)
                outputs = model(inputs)
                labels = labels.type_as(outputs)
                loss = criterion(outputs, labels)
                auc += clf_met(metric = "auc", 
                               y_true = labels.cpu().numpy(), 
                               y_proba = outputs.cpu().detach().numpy())
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * labels.size(0)
            if phase == 'train':
                epoch_loss = running_loss/len(train_data)
                epoch_auc = auc/count
            else:
                epoch_loss = running_loss/len(val_data)
                epoch_auc = auc/count
            print(f"{phase}:\nLoss = {epoch_loss}\tAUC={epoch_auc}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'model/{MODEL}_{VAL_FOLDS[0]}.pth')
        print("-*-"*20)
    print('Finished Training')
main()