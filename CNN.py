
# coding: utf-8

# In[3]:


get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
import torch.utils.data as data_utils
import csv
from os import listdir
import numpy as np
from PIL import Image
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

#matplotlib.use('Agg')


###-----------------------DEFINITION START-----------------------###

#Ultra Parameters
total_images = 112120
num_images = total_images/2
shuffle = True
random_seed = 41
valid_size = 0.2
data_entry_path = '/datasets/ChestXray-NIHCC/Data_Entry_2017.csv'
image_dir1 = '/datasets/tmp/tum_torch/001/'
image_dir2 = '/datasets/tmp/tum_torch/002/'
training_loss = []
training_loss_buffer = []
training_acc = []
validation_loss = []
validation_loss_buffer = []
validation_acc = []



#Hyper Parameters
num_epochs = 10
batch_size = 25
learning_rate = 0.001

'''
This function takes in an image's path and returns 
a numpy array that contains all its pixel values.
The dimension of the output array is (channels, width, height).
And in our case, channels = 1.
'''
def get_image(image_path):
    image = Image.open(image_path, 'r')
#    width, height = image.size[0]/4, image.size[1]/4
#    image.thumbnail((width,height),Image.ANTIALIAS)
    width, height = image.size[0], image.size[1]
#    if image.mode != 'L':
#        image = image.convert('L')
    pixel_values = list(image.getdata())
    #pixel_values = np.array(pixel_values).reshape((width, height))
    #pixel_values = preprocessing.scale(pixel_values)
    pixel_values = np.array(pixel_values).reshape((1, width, height))
    #pixel_values = np.array(pixel_values)
    return pixel_values

'''
Just loop through all the images in the image folder
and call convert each of them to an array.
The dimension of output is (num of images, (channels, width, height))
'''
def loadImages(directory,num_images):
    imageList = listdir(directory)
    loadedImages = []
    index = 0
    for image in imageList:
        if index == num_images: break
        pixel_values = torch.load(directory + image)
        loadedImages.append(pixel_values)
        if index % 100 == 0:
            print "Loading Image: ", index
        index += 1
	#print index
    return loadedImages

'''
Function name explains itself
'''
def constructLabelByString(labelStr):
    label_vector = [0] * 14
    if labelStr == 'No Finding':
        return label_vector

    symptom_vector = ['Atelectasis', 'Cardiomegaly', 'Effusion',     'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',     'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',     'Pleural_Thickening', 'Hernia']
    
    symptoms = labelStr.split('|')
    for s in symptoms:
        label_vector[symptom_vector.index(s)] = 1
    return label_vector
   
'''
Generate labels
'''
def parseDataEntry(dataEntryPath):   
    #Each element in the map is a tuple (image filename, corresponding label)
    labels = []
    index = 0
    with open('/datasets/ChestXray-NIHCC/Data_Entry_2017.csv', 'rb') as file:
        reader = csv.reader(file)
        for row in reader:
            if index == 0: 
                index += 1
                continue
            label_vector = constructLabelByString(row[1])
            labels.append(label_vector)
    return labels

'''
Model
'''
# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(32 * 64 * 64, 64)
        self.fc2 = nn.Linear(64, 14)
        self.normal = nn.BatchNorm1d(14)
        
        
        
    def forward(self, x):
        out = self.layer1(x)
        layer1_filters = out
        out = self.layer2(out)
        layer2_filters = out
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc2(out)
        out = self.normal(out)
        out = torch.sigmoid(out)
        return out, layer1_filters, layer2_filters
        
cnn = CNN()
cnn.cuda()
for m in cnn.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight)

# Loss and Optimizer
#criterion = nn.MultiLabelMarginLoss()
criterion = nn.BCELoss()
criterion = criterion.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

###-----------------------DEFINITION END-------------------------###


filter1 = None
filter2 = None
#correct_test = 0
valid_size_num = int(np.floor(valid_size*num_images))
# Train the Model
for epoch in xrange(num_epochs):

#---------- LOADING DATASET FROM 001 -------------#    
    #Load Images
    loadedImages = torch.stack(loadImages(image_dir1,num_images))
    print "done loading images"

    #Get labels
    loadedLabels = np.array(parseDataEntry(data_entry_path)[:num_images])
    print "done loading labels"

    #Convert to Datasets
    features = loadedImages.float()
    targets = torch.from_numpy(loadedLabels).float()
    train_dataset = data_utils.TensorDataset(features, targets)


    #Using SubsetRandomSampler for 1-fold cross validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_sampler = data_utils.sampler.SubsetRandomSampler(indices)
    train_test_sampler = data_utils.sampler.SubsetRandomSampler(indices[:valid_size_num])

    train_loader = data_utils.DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler, pin_memory = True)

    train_test_loader = data_utils.DataLoader(train_dataset,batch_size = 1, sampler = train_test_sampler, pin_memory = True)
#----------- END LOADING DATASET FROM 001 -----------#

    #Begin Running Epoch for Dataset 001
    for i, (images, labels) in enumerate(train_loader):
        #images = Variable(images)
        #labels = Variable(labels)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs, filter1, filter2 = cnn(images)
        #longOutputs = Variable(outputs.data.long()).cuda()
        filter1 = filter1.data.cpu().numpy()[0]
        #print ("out = ", outputs)
        #print (type(outputs))
        #print (type(longOutputs))
        #raw_input()
        loss = criterion(outputs, labels)
        #loss = F.binary_cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
        %(epoch+1, num_epochs, i+1, np.ceil(len(train_dataset))//batch_size, loss.data[0]))
        training_loss_buffer.append(loss.data[0])
    
    correct_test = 0
    cnn.eval()   
    for images, labels in train_test_loader:
        images = Variable(images).cuda()
        outputs,filter1,filter2 = cnn(images)
        prediction = outputs[0].data.cpu().numpy()
        prediction[prediction >= 0.6] = 1
        prediction[prediction < 0.6] = 0
        labels = labels.numpy()
        #print ("labels = ", labels)
        #print ("prediction = ", prediction)
        #print np.array_equal(prediction, labels)
        #raw_input()
        
        diff = np.count_nonzero(prediction != labels)
        correct_rate_test = 1.0 - (float(diff) / 14.0)
        correct_test+= correct_rate_test
    cnn.train()
    
#Dump memory to save space
    loadedImages = 0
    loadedLabels = 0
    features = 0
    targets = 0
    train_dataset = 0
    train_sampler = 0
    valid_sampler = 0
    train_loader = 0
    valid_loader = 0
    train_test_loader = 0
#End Dump Memory

#---------- LOADING DATASET FROM 002 -------------#          
    #Load Images
    loadedImages = torch.stack(loadImages(image_dir2,num_images))
    print "done loading images"

    #Get labels
    loadedLabels = np.array(parseDataEntry(data_entry_path)[total_images/2:(total_images-(total_images/2 - num_images))])
    print "done loading labels"

    #Convert to Datasets
    features = loadedImages.float()
    targets = torch.from_numpy(loadedLabels).float()
    train_dataset = data_utils.TensorDataset(features, targets)


    #Using SubsetRandomSampler for 1-fold cross validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size*num_train))
    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = data_utils.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = data_utils.sampler.SubsetRandomSampler(valid_idx)

    train_loader = data_utils.DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler, pin_memory = True)
    valid_loader = data_utils.DataLoader(train_dataset, batch_size = 1, sampler = valid_sampler, pin_memory = True)
#----------- END LOADING DATASET FROM 002 ----------#

    #Begin Running Epoch for Dataset 002
    for i, (images, labels) in enumerate(train_loader):
        #images = Variable(images)
        #labels = Variable(labels)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs, filter1, filter2 = cnn(images)
        #longOutputs = Variable(outputs.data.long()).cuda()
        filter1 = filter1.data.cpu().numpy()[0]
        #print ("out = ", outputs)
        #print (type(outputs))
        #print (type(longOutputs))
        #raw_input()
        loss = criterion(outputs, labels)
        #loss = F.binary_cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
        %(epoch+1, num_epochs, i+1, np.ceil(len(train_dataset)*(1-valid_size))//batch_size, loss.data[0]))
        training_loss_buffer.append(loss.data[0])
    
    #Append Average Loss over Each Batch and flush Buffer
    training_loss.append(np.mean(training_loss_buffer))
    training_loss_buffer = []
    # Test the Model
    cnn.eval()  # Change model to eval mode
    correct = 0
    
    total = 0
    for images, labels in valid_loader:
        images = Variable(images).cuda()
        oglabels = Variable(labels).cuda()
        outputs,filter1,filter2 = cnn(images)
        prediction = outputs[0].data.cpu().numpy()
        prediction[prediction >= 0.6] = 1
        prediction[prediction < 0.6] = 0
        total += labels.size(0)
        labels = labels.numpy()
        #print ("labels = ", labels)
        #print ("prediction = ", prediction)
        #print np.array_equal(prediction, labels)
        #raw_input()
        
        diff = np.count_nonzero(prediction != labels)
        correct_rate = 1.0 - (float(diff) / 14.0)
        correct+= correct_rate
        
        loss = criterion(outputs,oglabels)
        loss.backward()
        validation_loss_buffer.append(loss.data[0])
        
    

    
    validation_loss.append(np.mean(validation_loss_buffer))
    validation_loss_buffer = []
    test_acc = 100 * correct / total
    train_test_acc = 100 * correct_test / valid_size_num
    print('Test Accuracy of the model after epoch %d: %d %% (Total: %d; Correct: %d)' % (epoch+1, test_acc, total, correct))
    print('Train Accuracy of the model after epoch %d: %d %% (Total: %d; Correct: %d)' % (epoch+1, train_test_acc, valid_size_num, correct_test))
    validation_acc.append(test_acc)
    training_acc.append(train_test_acc)
    #Dump memory to save space
    loadedImages = 0
    loadedLabels = 0
    features = 0
    targets = 0
    train_dataset = 0

    train_sampler = 0
    valid_sampler = 0
    train_loader = 0
    valid_loader = 0
    #End Dump Memory
    
    #change model back to train mode
    cnn.train()
    
features_layer1 = None
for m in cnn.modules():
    if isinstance(m, nn.Conv2d):
	features_layer1 = m.weight.data.cpu().numpy()
#print features_layer1.shape

plt.plot(range(1,num_epochs+1),training_loss, label = 'Training Loss')
plt.plot(range(1,num_epochs+1),validation_loss, label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('BCELoss on Training/Validation Set vs Epochs')
plt.show()
plt.figure(2)

plt.plot(range(1,num_epochs+1),validation_acc,label = 'Test Accuracy')
plt.plot(range(1,num_epochs+1),training_acc,label = 'Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train/Test Accuracy vs Epochs')
plt.show()

# '''
# fig, ax = plt.subplots(nrows = 32, ncols = 16)
# index = 0
# irow = 0
# icol = 0
# for row in ax:
#     for col in row:
#         image = features_layer1[irow][icol]
# 	#print image.shape
#         #image = image.reshape((28, 28))
#         col.imshow(image, cmap = 'gray')
#         col.axis('off')
#         index += 1
# 	icol += 1
#     icol = 0
#     irow += 1
# plt.show()
# plt.savefig('features.png')
# '''




# Save the Trained Model
#torch.save(cnn.state_dict(), 'cnn.pkl')


# In[ ]:




