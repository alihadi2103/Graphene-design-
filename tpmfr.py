import os
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from dataset import GraphenImageDataset
import numpy as np
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, in_dim, bias=False),
        )

    def forward(self, x):
        return x + self.main(x)

class MLP(nn.Module):
    def __init__(self, in_size):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.ReLU(),
        )
        self.output = nn.Linear(64, 1)
    
    def forward(self, x):
        out = self.model(x)
        return self.output(out), out

def save_weights(resnet, mlp, model_dir, obj):
    torch.save(mlp.state_dict(), os.path.join(model_dir, f'mlp_{obj}.ckpt'))
    torch.save(resnet.state_dict(), os.path.join(model_dir, f'resnet_{obj}.ckpt'))

def load_weights(resnet, mlp, model_dir, obj, device):
    resnet.load_state_dict(torch.load(os.path.join(model_dir, f'resnet_{obj}.ckpt'), map_location=device))
    mlp.load_state_dict(torch.load(os.path.join(model_dir, f'mlp_{obj}.ckpt'), map_location=device))

def train(num_epochs=50, resume_training=False,obj="rejection",
          batch_size = 64,device="cpu",lr_mlp = 0.001,
          lr_resnet = 0.0001,model_dir = './models'):
    

    
    
    obj = 'rejection'

    transform = transforms.Compose([
        transforms.CenterCrop((380, 380)),
        transforms.Resize((224, 224))
    ])

    train_dataset = GraphenImageDataset(
        img_dir='./data/image', 
        csv_path='./data/noisy_augmented_label.csv', 
        transform=transform, 
        mode='train',
        label_mode=obj
    )

    test_dataset = GraphenImageDataset(
        img_dir='./data/image', 
        csv_path=r'C:\Users\SAMSUNG\Downloads\Graphene-RL\data\noisy_augmented_label.csv', 
        transform=transform, 
        mode='test',
        label_mode=obj
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    resnet = torchvision.models.resnet18(pretrained=True).to(device)
    mlp = MLP(in_size=1000).to(device)

    ct = 0
    for child in resnet.children():
        ct += 1
        if ct < 6:
            for param in child.parameters():
                param.requires_grad = False

    loss_func = nn.MSELoss(reduction='mean')
    optim_mlp = Adam(mlp.parameters() )
    optim_resnet = Adam(resnet.parameters() )

   

    
    if resume_training:
        # Load weights for resume training
        load_weights(resnet, mlp, model_dir, obj, device)

    train_losses, test_losses = [], []
    for epoch in range(num_epochs):

        acc_train_loss = 0.0
        resnet.train()
        mlp.train()
        for i, (img, label) in enumerate(train_dataloader):
            img, label = img.to(device), label.to(device)
            if len(label.shape) == 1:
                label = torch.unsqueeze(label, 1)
            
            feat = resnet(img)
            pred, __ = mlp(feat)

            optim_mlp.zero_grad()
            optim_resnet.zero_grad()
            loss = loss_func(pred, label)
            loss.backward()
            acc_train_loss += loss.item()
            optim_mlp.step()
            optim_resnet.step()

            torch.cuda.empty_cache()

        train_losses.append(acc_train_loss/(i+1))

        
        

        # validation on test data
        resnet.eval()
        mlp.eval()
        predictions = np.zeros(len(test_dataset))
        labels = np.zeros(len(test_dataset))
        start_idx, end_idx = 0, 0
        acc_test_loss = 0.0
        with torch.no_grad():
            for i, (img, label) in enumerate(test_dataloader):
                img, label = img.to(device), label.to(device)
                if len(label.shape) == 1:
                    label = torch.unsqueeze(label, 1)

                feat = resnet(img)
                pred, __ = mlp(feat)

                loss = loss_func(pred, label)
                acc_test_loss += loss.item()

        test_losses.append(acc_test_loss/(i+1))
        
        print(f"epoch: {epoch}, training Loss: {train_losses[-1]}, testing loss: {test_losses[-1]}")
     # Save weights after each train sesion deppending on number of epochs
    save_weights(resnet, mlp, model_dir, obj)
    
    while save_weights(resnet, mlp, model_dir, obj) :
        print("This session has been saved safly ")
    




def validate(obj='flux',batch_size = 64,model_dir = './models' ) :
    
      transform = transforms.Compose([
        transforms.CenterCrop((380, 380)),
        transforms.Resize((224, 224))
        ])

      train_dataset = GraphenImageDataset(
        img_dir='./data/image', 
        csv_path='./data/noisy_augmented_label.csv', 
        transform=transform, 
        mode='train',
        label_mode=obj)
    

      test_dataset = GraphenImageDataset(
        img_dir='./data/image', 
        csv_path=r'C:\Users\SAMSUNG\Downloads\Graphene-RL\data\noisy_augmented_label.csv', 
        transform=transform, 
        mode='test',
        label_mode=obj)
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
      test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
      
      
      resnet = torchvision.models.resnet18(pretrained=True).to(device)
      #mlp = MLP(in_size=1000).to(device)
      load_weights(resnet, mlp, model_dir, obj, device)
      #validation on train data
      train_predictions = np.zeros(len(train_dataset))
      train_labels = np.zeros(len(train_dataset))
      start_idx, end_idx = 0, 0
      for i, (img, label) in enumerate(train_dataloader):
         img, label = img.to(device), label.to(device)
         if len(label.shape) == 1:
             label = torch.unsqueeze(label, 1)
        
         batch_size = label.shape[0]
         end_idx += batch_size
         feat = resnet(img)
         pred, __ = mlp(feat)
         pred = pred.detach().numpy()
         print(pred.shape)
         label = label.detach().numpy()
        
         train_predictions[start_idx:end_idx] = np.squeeze(pred)
         train_labels[start_idx:end_idx] = np.squeeze(label)
         start_idx = end_idx
     #validation on test data
      print("first loop has finished secessfully")

      test_predictions = np.zeros(len(test_dataset))
      test_labels = np.zeros(len(test_dataset))
      start_idx, end_idx = 0, 0
      for i, (img, label) in enumerate(test_dataloader):
         img, label = img.to(device), label.to(device)
         if len(label.shape) == 1:
            label = torch.unsqueeze(label, 1)
         batch_size = label.shape[0]
         end_idx += batch_size
        
         feat = resnet(img)
         pred, __ = mlp(feat)
       
       
         pred = pred.detach().numpy()
         print(pred.shape)
        
         label = label.detach().numpy()
        
        
         test_predictions[start_idx:end_idx] = np.squeeze(pred)
         test_labels[start_idx:end_idx] = np.squeeze(label)
         start_idx = end_idx
      print("Second loop has finished secessfully")



      

      print("MSE on training set:", np.mean(np.square(train_predictions - train_labels)))
      print("MSE on testing set:", np.mean(np.square(test_predictions - test_labels)))
      print("L1 error on training set:", np.mean(np.abs(train_predictions - train_labels)))
      print("L1 error on testing set:", np.mean(np.abs(test_predictions - test_labels)))


      plot_dir = './plot'
      if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
      x = np.linspace(np.min(test_dataset.labels), np.max(train_dataset.labels))
      y = np.linspace(np.min(test_dataset.labels), np.max(train_dataset.labels))
      plt.figure()
      plt.scatter(train_predictions, train_labels, c='blue', marker='x')
      plt.scatter(test_predictions, test_labels, c='red', marker='x')
      plt.plot(x, y, linestyle='dashed', c='black')
      plt.xlabel('prediction')
      plt.ylabel('label')
      plt.title('Flux prediction')
      plt.show()
      plt.savefig(os.path.join(plot_dir, 'pred_new_{}.png'.format(obj)))
    
if __name__ == "__main__":
    device = 'cpu'
    obj = 'rejection'
    model_dir = './models'
    batch_size = 64
    train(num_epochs=1, resume_training=False,obj=obj,)
    #validate(obj=obj ,batch_size=batch_size)
    
      

    
   