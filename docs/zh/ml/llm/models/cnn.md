!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# CNN

### Hierarchical visual representations

![image.png](../../../../assets/notion/025-00.png)

### Same kernel applied to all sliding windows

![image.png](../../../../assets/notion/025-01.png)

![image.png](../../../../assets/notion/025-02.png)

### RCNN

run CNN on the whole image, then segment

![image.png](../../../../assets/notion/025-03.png)

### CAM

the last fully connected layer will lose spatial information (instead of up/down/left/right, we flatten the matrix)

CAM set weights for each channel of the final feature embedding

![image.png](../../../../assets/notion/025-04.png)
