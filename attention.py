import torch 

# We want to couple the tokens, one at 5 shoudl not comm with one in 6,7,8 as they are future
# should only talk to 4,3,2,1 so from previous to curtrent

#do an average of all the preceeding tokens
torch.manual_seed(1337)
B,T,C = 4,8,2
x = torch.randn(B,T,C)
x.shape


#X bag of words, there is word stored at each of the 8 locations
xbow = torch.zeros((B,T,C))

#iteratve over bathc
for b in range(B):
	#iterate over time
	for t in range(T):


		# 		x = [[0, 1, 2, 3],
		#      [4, 5, 6, 7],
		#      [8, 9, 10, 11]]
		# Now, if you have b = 1 and t = 2, then x[b, :t+1] would slice the tensor to get the second sequence up to the third element:

		# makefile
		# Copy code
		# xprev = x[1, :3]  
		# This would give you [4, 5, 6]

		#1 hot encoding means taking categorical data cat, dog, fish
		#and encoding it into categories like 1,2,3 or [0,1,0] == dog

		#Previous tokens at b, everytthing up to and including t token
		xprev = x[b,:t+1] #(t,C) shape is t elements in past and C 2d info
		
		#then do average of 0th dimension, average time which is 0 dimension
		xbow[b,t] = torch.mean(xprev, 0)#then we have a 1D vector store in xbow