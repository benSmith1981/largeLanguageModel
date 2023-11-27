import requests
import torch  # we use PyTorch: https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
# The URL from which to download the file
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ---------------

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Write the contents of the response to a file
    with open('input.txt', 'wb') as file:
        file.write(response.content)
    print('File downloaded successfully!')
else:
    print('Failed to download file. Status code:', response.status_code)# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# Output shown in the image:
# length of dataset in characters:  1115394


# print(set(text))
# print(list(set(text)))#create a list of the characters
chars = sorted(list(set(text)))
# print(chars)
vocab_size = len(chars)
print(vocab_size)
print(''.join(chars))
#tokenise the input text
# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)} #look up table char to int
print(stoi)
itos = {i:ch for i, ch in enumerate(chars)}
# print(itos)
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: "".join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode(text[:1000]))
print(decode(encode(text[:1000])))

# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])  # the 1000 characters we looked at earlier will to the GPT look like this

# Let's now split up the data into train and validation sets
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
#want to work out how much model is overfitting
val_data = data[n:]

#we train chunks at a time...
block_size = 8 # max length of chunk to train transformer on
train_data[:block_size+1]#going to train it to make preduciont at everyon one of the positions
#This has multiple examples plugged into it so it will make a prediciton at any one of the positions
print(train_data[:block_size+1])

x = train_data[:block_size]#inputs to teh trainsformer...
y = train_data[1:block_size+1]#y next block size... targets for each 

#want transform network to see characters of context of size 1 up to 
#block_size, so we make the transformer network used to seeing context
#from 1 all the way up to block_size.. later useful during inference
#as we can sample any number of characters...
for t in range(block_size):
	context = x[:t+1]
	target = y[t]
	print(f"when input is {context} the target:{target}")


# batch_size = 4  # how many independent sequences will we process in parallel?
# block_size = 8  # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    #ix is 4 random generated numbers between zero and lenght of data - blocksize
    #so basically some random samples into the data as to where to get the blocks
    #get 4 because that is batch size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #x is first blocksize characters starting at i
    x = torch.stack([data[i:i+block_size] for i in ix])
    #stack is stacking up 1d tensors in a row of 4x8
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
# Assuming train_data and val_data are predefined somewhere above this code
# For example:
# train_data = torch.randn(1000, 32)  # (sequence_length, feature_size)
# val_data = torch.randn(200, 32)     # (sequence_length, feature_size)

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-----')

for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimension
        context = xb[b, t]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target.tolist()}")



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        #Logits are the score for the next character in the 
        #sequence
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
        	loss = None
        else:
        	#batch, time, channels (the range of vocab?)
	        B, T, C = logits.shape
	        #stretches out the array so it is 2d
	        logits = logits.view(B*T, C)
	        targets = targets.view(B*T)
	        #this tells us how well we are predicting the next cahracter
	        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
	    # idx is (B, T) array of indices in the current context
	    for _ in range(max_new_tokens):
	        # get the predictions
	        logits, loss = self(idx)
	        # focus only on the last time step
	        logits = logits[:, -1, :]  # becomes (B, C)
	        # apply softmax to get probabilities
	        probs = F.softmax(logits, dim=-1)  # (B, C)
	        # sample from the distribution
	        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
	        # append sampled index to the running sequence
	        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
	    return idx


# Assuming vocab_size is defined
# vocab_size = ...
model = BigramLanguageModel(vocab_size)
m = model.to(device)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# idx = torch.zeros((1,1),dtype=torch.long)
# print(decode(m.generate(,max_new_tokens=100)[0].tolist()))
# print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


#takes gradients and update the parameters (which ones contribute to error?)
optimizer = torch.optim.AdamW(m.parameters(),lr=learning_rate) 


batch_size = 32

for steps in range(max_iters):
	# every once in a while evaluate the loss on train and val sets
	if iter % eval_interval == 0:
	    losses = estimate_loss()
	    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
	xb, yb = get_batch('train')
    
    # evaluate the loss
	logits, loss = m(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()
print(loss.item())

context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=400)[0].tolist()))
