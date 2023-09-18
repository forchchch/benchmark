# Benchmark

## File Structure

1. Photos in instance
2. prompts in prompt
3. pairs and classes in benchmark.json
4. assets no use
5. metrics can now compute frame-consistency, CLIP-i, CLIP-t and number-of-objects for a certain generated video

## metrics

### Number of objects

这个metric衡量主体数量是否符合文字描述

1. 读入mp4，逐帧提取
2. 对每一帧，用yolov5检测所有的与cls1, cls2相符合的物体的个数，计算与prompt中指定的个数的差的绝对值

### CLIP-I

这个metric衡量视频中的每一个主体能否保留各自instance的特征

以两个物体为例

1. 读入mp4，逐帧提取

2. 对每一帧，用yolov5检测所有的与cls1, cls2相符合的物体的边框位置，提取这些图片，记为p1, p2。

   （应该是两张图，如果多于两张就删掉后面相似度低的，如果少于两张……不知道，可能用纯黑的照片代替吧）

3. 计算 CLIP-I < p1, {ins1} >，CLIP-I < p1, {ins2} >， CLIP-I <p2, {ins1}>, CLIP-I <p2, {ins2}> 。这里 {ins1}, {ins2}都是几张图片的集合，默认取3-5个数的平均。

4. 计算

$$ max (CLIP-I < p1, {ins1} > + CLIP-I <p2, {ins2}>,  CLIP-I < p1, {ins2} > + CLIP-I <p2, {ins1}>) $$

### DINO-I

与CLIP-I同理，区别是第3步利用DINO

### DIV

这个metric的定义和dreambooth中的同名指标相同，目的是衡量同一段text生成的多段视频的多样性，避免过拟合

### Frame Consistency

这个metric衡量视频时间连续性

与Tune-A-Video的做法相同，计算一段视频帧与帧之间的CLIP相似度，取平均，得到一段视频的时间连续性

### CLIP-T

这个metric衡量视频的整体场景是否贴合文字描述

逐帧提取，计算CLIP-T < text, {frame} >，对所有帧取平均