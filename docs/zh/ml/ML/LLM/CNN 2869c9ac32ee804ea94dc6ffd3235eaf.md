# CNN

### Hierarchical visual representations

![image.png](CNN/image.png)

### Same kernel applied to all sliding windows

![image.png](CNN/image%201.png)

![image.png](CNN/image%202.png)

### RCNN

run CNN on the whole image, then segment

![image.png](CNN/image%203.png)

### CAM

the last fully connected layer will lose spatial information (instead of up/down/left/right, we flatten the matrix)

CAM set weights for each channel of the final feature embedding

![image.png](CNN/image%204.png)