# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.
## Neural Network Model

![image](https://github.com/user-attachments/assets/84093ee0-48a5-4bd2-b78d-5d8ee258d189)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: JAWAHAR RAJ N
### Register Number:212223240057
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history={'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x)) 
    x=self.relu(self.fc2(x))
    x=self.fc3(x)  
    return x


# Initialize the Model, Loss Function, and Optimizer

jawa = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(jawa.parameters(),lr=0.001)


def train_model(jawa, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(jawa(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
       jawa.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
    



```
## Dataset Information

![image](https://github.com/user-attachments/assets/2b45a519-d54f-410a-9b12-6c909e64d249)


## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/6a8454da-97d8-4522-99a1-fddc33a19d50)



### New Sample Data Prediction
<img width="1024" height="161" alt="image" src="https://github.com/user-attachments/assets/4c04598c-c17d-41f5-b4c7-a5d6667a185c" />


## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
