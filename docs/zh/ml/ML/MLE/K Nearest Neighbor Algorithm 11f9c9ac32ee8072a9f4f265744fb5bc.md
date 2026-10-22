# K Nearest Neighbor Algorithm

### **What is k-NN Algorithm?**

It stores all the available data and classifies a new data point based on similarity

It is supervised

### ⚙️ **How It Works**

1. Choose the number of neighbors **k**.
2. For a new data point, compute its **distance** (e.g., Euclidean) to all training samples.
3. Select the **k nearest neighbors**.
4. **Classification:** majority vote among neighbors.
    
    **Regression:** average of neighbors’ values.
    

### **What is the role of the k value in the k-NN algorithm?**

The ‘k’ in k-NN refers to the number of nearest neighbors that are considered for making a prediction for a new data point

### **Why is the odd value of ‘k’ preferred over an even value in the k-NN algorithm?**

possibility of having an equal number of neighbors for each class

### **How does the k-NN algorithm make predictions on unseen datasets?**

1. k-closest points
2. assign class label of the majority of k neighbors