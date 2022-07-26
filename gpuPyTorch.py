
# How to use gpu training
#https://towardsdatascience.com/pytorch-switching-to-the-gpu-a7c0b21e8a99
#define the device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# APPROACH 1: using train_test_split
#transfer data to the device
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

#transfer model to the device
model = MyAwesomeNeuralNetwork()
model.to(device)
# start training


# APPROACH 2: using DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyAwesomeNeuralNetwork()
model.to(device)

epochs = 10
for epoch in range(epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # backpropagation code here
        # evaluation
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
        # ...


# How to use dataloaders for custom dataset

#first create the dataset, e.g.:

def read_dataset(data_path):
    samples = []
    with io.open(data_path, mode="r", encoding="utf-8") as file:
        for line in file:
            samples.append(json.loads(line.strip()))
    return samples

train_samples = read_dataset(args.train_data_l1)
train_examples = []
for s, sample in enumerate(tqdm(train_samples))
    train_examples.append(InputExample(texts=[context, sample["l1_output"][0]], label=0))
#train_examples has all the training examples in the required format

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=512)
bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)], epochs=args.num_epochs, use_amp=True,
               warmup_steps=args.warmup_steps, optimizer_params={'lr': 5e-05}, checkpoint_path=args.biencoder_save,
               checkpoint_save_steps=10000)
