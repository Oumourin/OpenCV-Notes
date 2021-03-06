#  Day3

##  OpenCV中图像对象创建与赋值

###  C++实现

####   Mat下clone()方法

```c++
Mat m1 = src.clone()
```

值一样，地址不一样

####  Mat下copyTo()

```C++
Mat m2;
src.copyTo(m2);
```

与clone()方法类似

####  Mat下赋值法

```C++
Mat m3 = src;
```

地址一样，share同一个对象

![OpenCV Mat结构](https://image.nuccombat.cn/images/2019/03/20/FtHKxYuo6Io6zVXqxzwk-92FopQie1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDMZMwDTwvFyedbGQHUsG10Tc0kSQ.png)

####  OpenCV C++下创建空白图像

```C++
	Mat m4 = Mat::zeros(src.size(), src.type());
	Mat m5 = Mat::zeros(Size(512, 512), CV_8UC3);
	Mat m6 = Mat::ones(Size(512, 512), CV_8UC3);
```

m4 调用src的属性，创建同属性的零矩阵

m5 创建一个零矩阵，大小和属性为自定义

zeros、ones的API为Matlab舶来品

####  OpenCV C++下创建类似卷积核矩阵

```c++
	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0,
		-1, 5, -1,
		0, -1, 0);
```

创建一个3X3矩阵

###  Python实现

```python
m2 = src
src[100:200, 200:300, :] = 255
cv.imshow("m2", m2)
```

在宽度100-200 高度200-300位置创建一个100X100矩阵 矩阵填充255

```python
m4 = np.zeros([512, 512], np.uint8)
cv.imshow("m4", m4)
```

创建一个512X512的零矩阵，类型为uint8



```python
m5 = np.ones(shape=[512, 512, 3], dtype=np.uint8)
m5[:, :, 0] = 255
cv.imshow("m5", m5)
```

创建一个三通道512X512矩阵，每个通道类型为uint8，并且对第一个通道赋值255，其中m5[:, :, 0]代表高度、宽度、通道数

#  Day4

##  图像像素操作（遍历与访问）

###  C++实现

###  读取宽、高、通道数

```C++
int height = src.rows;
int width = src.cols;
int ch = src.channels();
```

src的rows成员对应高度、cols对应宽度、channels对应通道（一般为3通道，特殊情况可以为4通道多一个透明通道）

####  实现反色(数值方法)

```c++
	for (int c = 0; c < ch; c++) {
		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				if (ch == 3) {
					Vec3b bgr = src.at<Vec3b>(row, col);
					bgr[0] = 255 - bgr[0];
					bgr[1] = 255 - bgr[1];
					bgr[2] = 255 - bgr[2];
					src.at<Vec3b>(row, col) = bgr;
				} else if(ch == 1) {
					int gray = src.at<uchar>(row, col);
					src.at<uchar>(row, col) = 255 - gray;
				}
			}
		}
```

利用数组方式遍历整个图片，采用at方法获取像素值

####  复制图片（指针方法）

```c++
	for (int c = 0; c < ch; c++) {
		for (int row = 0; row < height; row++) {
			uchar* curr_row = src.ptr<uchar>(row);
			uchar* result_row = result.ptr<uchar>(row);
			for (int col = 0; col < width; col++) {
				if (ch == 3) {
					blue = *curr_row++;
					green = *curr_row++;
					red = *curr_row++;

					*result_row++ = blue;
					*result_row++ = green;
					*result_row++ = red;
				}
				else if (ch == 1) {
					gray = *curr_row++;
					*result_row++ = gray;
				}
			}
		}
	}
```



通过ptr方法获取图片行列头指针，在三通道情况下，应该使用自增运算符实现对三个通道的访问

###  Python实现

```python
h, w, ch = src.shape
print("h , w, ch", h, w, ch)
```

利用shape方法获取行，列数（通道数在单通道时不返回）

####  实现反色

```python
for row in range(h):
    for col in range(w):
        b, g, r = src[row, col]
        b = 255 - b
        g = 255 - g
        r = 255 - r
        src[row, col] = [b, g, r]
```

# Day5

##  OpenCV的算术操作

###  C++实现

####  加减乘除

API函数对应英文翻译，参与运算的图像的要求：类型必须一致，通道数目一致，宽高相同

#####  加法

```c++
	Mat add_result = Mat::zeros(src1.size(), src1.type());
	add(src1, src2, add_result);
	imshow("add_result", add_result);
```



* 第一个参数：第一张图像
* 第二个参数：第二张图像
* 第三个图像：输出图像



#####  减法

````c++
	Mat sub_result = Mat::zeros(src1.size(), src1.type());
	subtract(src1, src2, sub_result);
	imshow("sub_result", sub_result);
````

参数同加法



#####  乘法

````c++
	Mat mul_result = Mat::zeros(src1.size(), src1.type());
	multiply(src1, src2, mul_result);
	imshow("mul_result", mul_result);
````

参数同上



#####  除法

```c++
	Mat div_result = Mat::zeros(src1.size(), src1.type());
	divide(src1, src2, div_result);
	imshow("div_result", div_result);
```

参数同上



####  加减乘数的像素点遍历实现

#####  saturate_cast\<T>方法

实现C++的精确类型转换



#####  实现代码

```C++
	int height = src1.rows;
	int width = src1.cols;

	int b1 = 0, g1 = 0, r1 = 0;
	int b2 = 0, g2 = 0, r2 = 0;
	int b = 0, g = 0, r = 0;
	Mat result = Mat::zeros(src1.size(), src1.type());
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
				b1 = src1.at<Vec3b>(row, col)[0];
				g1 = src1.at<Vec3b>(row, col)[1];
				r1 = src1.at<Vec3b>(row, col)[2];

				b2 = src2.at<Vec3b>(row, col)[0];
				g2 = src2.at<Vec3b>(row, col)[1];
				r2 = src2.at<Vec3b>(row, col)[2];

				result.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b1 + b2);
				result.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g1 + g2);
				result.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r1 + r2);
		}
	}
```

saturate_cast避免因为两个像素相加所带来的数值溢出

###  Python实现

#####  加减乘除

```Python
add_result = np.zeros(src1.shape, src1.dtype);
cv.add(src1, src2, add_result);
cv.imshow("add_result", add_result);

sub_result = np.zeros(src1.shape, src1.dtype);
cv.subtract(src1, src2, sub_result);
cv.imshow("sub_result", sub_result);

mul_result = np.zeros(src1.shape, src1.dtype);
cv.multiply(src1, src2, mul_result);
cv.imshow("mul_result", mul_result);

div_result = np.zeros(src1.shape, src1.dtype);
cv.divide(src1, src2, div_result);
cv.imshow("div_result", div_result);
```

Python下创建空白矩阵利用numpy下的zeros方法

* 参数一：第一张图像
* 参数二：第二张图像
* 参数三：输出结果



#  Day6

##  Look Up Table(LUT)查找表

![LUT](https://image.nuccombat.cn/images/2019/03/21/Fk5_5JvkflTl70xmC3E-aJqw7fEte1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDEt-s5NPkIOv9DaPxEI9aCj1MUnQ.png)

节省算力，将数据快速转化

###  C++实现

####  利用LUT将灰度图像变为二值图像

```c++
void customColorMap(Mat &image) {
	int lut[256];
	for (int i = 0; i < 256; i++) {
		if (i < 127)
			lut[i] = 0;
		else
			lut[i] = 255;
	}

	int h = image.rows;
	int w = image.cols;
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			int pv = image.at<uchar>(row, col);
			image.at<uchar>(row, col) = lut[pv];
		}
	}
	imshow("lut demo", image);
}
```

####  利用OpenCV API实现LUT

```c++
applyColorMap(src, dst, COLORMAP_SUMMER);
```

###  Python实现

####  利用OpenCV API实现LUT

```python
cv.applyColorMap(src, cv.COLORMAP_COOL)
```



#  Day7

##  像素操作之逻辑操作

*  bitwise_and
* bitwise_xor
* bitwise_or



![](https://image.nuccombat.cn/images/2019/03/21/FskkWDxOASUd699oiRsOVrh_3vMFe1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDtvoeihWcbbs8VcdeOURur9tmL2M.png)

以上为两张图像的操作

*  bitwise_not

以上针对一张图像操作

##  C++实现

###  创建demo图片

```c++
	Mat src1 = Mat::zeros(Size(400, 400), CV_8UC3);
	Rect rect(100, 100, 100, 100);
	src1(rect) = Scalar(0, 0, 255);
```

### 两张图运算

```
	bitwise_and(src1, src2, dst1);
	bitwise_xor(src1, src2, dst2);
	bitwise_or(src1, src2, dst3);
```

### 一张图取反操作

```c++
	bitwise_and(src1, src2, dst1);
	bitwise_xor(src1, src2, dst2);
	bitwise_or(src1, src2, dst3);
```

*  bitwise_not 常用于对二值图像取反操作

* bitwise_and 常用于取特定区域，尤其对于不规则图像

## Python实现

###  创建demo图片

```python
src1 = np.zeros(shape=[400, 400, 3], dtype=np.uint8)
src1[100:200, 100:200, 1] = 255
src1[100:200, 100:200, 2] = 255
```

###  两张图运算

```python
dst1 = cv.bitwise_and(src1, src2)
dst2 = cv.bitwise_xor(src1, src2)
dst3 = cv.bitwise_or(src1, src2)
```

###  一张图运算

```python
dst = cv.bitwise_not(src)
```

#  Day8

##  通道分离与合并

OpenCV中默认imread函数加载图像文件，加载进来的是三通道彩色图像，色彩空间是RGB色彩空间、通道顺序是BGR（蓝色、绿色、红色）、对于三通道的图像OpenCV中提供了两个API函数用以实现通道分离与合并。

###  相关函数

* split	//通道分离
* merge    //通道合并

![](https://image.nuccombat.cn/images/2019/03/23/Fv8RQhGQWRpF54z4S7FLRzANYCnae1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDQmxdARaw8qTeFojLyHpdWaI6SSI.png)

###  C++实现

####  通道分离与合并

```c++
	vector<Mat> mv;
	Mat dst1, dst2, dst3;
	// 蓝色通道为零
	split(src, mv);
	mv[0] = Scalar(0);
	merge(mv, dst1);
	imshow("output1", dst1);

	// 绿色通道为零
	split(src, mv);
	mv[1] = Scalar(0);
	merge(mv, dst2);
	imshow("output2", dst2);

	// 红色通道为零
	split(src, mv);
	mv[2] = Scalar(0);
	merge(mv, dst3);
	imshow("output3", dst3);
```

C++中利用vector来存储分离出的通道，之后利用Scalar设置单独的通道值

###  Python实现

通道分离与合并

```python
# 蓝色通道为零
mv = cv.split(src)
mv[0][:, :] = 0
dst1 = cv.merge(mv)
cv.imshow("output1", dst1)

# 绿色通道为零
mv = cv.split(src)
mv[1][:, :] = 0
dst2 = cv.merge(mv)
cv.imshow("output2", dst2)

# 红色通道为零
mv = cv.split(src)
mv[2][:, :] = 0
dst3 = cv.merge(mv)
cv.imshow("output3", dst3)

dst = np.zeros(src.shape, dtype=np.uint8)
print(src.shape)
print(src.shape)
cv.mixChannels([src], [dst], fromTo=[2, 0, 1, 1, 0, 2])
cv.imshow("Dst", dst)
```

注：*split*函数分离出来的图像是单通道图像，对于多余通道采取填充该通道相同数值方式，故直接输出mv[0]等实质上会得到灰度图，*mixchannels*函数为混合通道函数，fromTo参数在这里表示，将src的2通道赋给dst的0通道，将1通道给dst的1通道，将0通道给dst的2通道

#  Day9

##  色彩空间与色彩空间转换

知识点： 色彩空间与色彩空间转换
- RGB色彩空间
- HSV色彩空间
- YUV色彩空间
- YCrCb色彩空间

API知识点
- 色彩空间转换cvtColor
- 提取指定色彩范围区域inRange

RGB为最常用的色彩空间，HSV在常常应用于直方图处理中，YUV色彩空间为欧洲电视标准，也用于Android的摄像头系统，YCrCb色彩空间用于皮肤检测等，有色人种皮肤检测会有很好效果

![](https://image.nuccombat.cn/images/2019/03/23/Fo15__R6kQoB2YOyg0mw17Lp8CQUe1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDskYQ1F4Qb-mX_5kDBWxO5hTQ0F4.png)

HSV在理论上是0-360，在OpenCV中为了面向工程，采取了0-180的范围，在损失可以接受情况下，避免空间的浪费。通过HSV色彩空间，可以很方便提取出特定颜色

![](https://image.nuccombat.cn/images/2019/03/23/FqAwsdZTYFXlWwVybyH3s_iCiQzbe1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDKsPEeYW1z5BTxI-PqBwyNlSAC0s.png)

###  C++实现

```c++
	Mat hsv;
	cvtColor(src, hsv, COLOR_RGB2HSV);
	imshow("HSV", hsv);

	Mat yuv;
	cvtColor(src, yuv, COLOR_BGR2YUV);
	imshow("YUV", yuv);

	Mat YCrCb;
	cvtColor(src, YCrCb, COLOR_BGR2YCrCb);
	imshow("YCrCb", YCrCb);

	Mat mask;
	//cvtColor(src, mask, COLOR_BGR2HSV);
	inRange(hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), mask);
	imshow("mask", mask);
	Mat dst;
	bitwise_not(src,src, dst, mask);
	imshow("and", dst);
```



###  Python实现

```python
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
cv.imshow("HSV", hsv)

yuv = cv.cvtColor(src, cv.COLOR_BGR2YUV)
cv.imshow("YUV", yuv)

yCrCb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
cv.imshow("YCrCb", yCrCb)

src2 = cv.imread("F:/timg.jpg")
cv.imshow("Demo", src2)
hsv2 = cv.cvtColor(src2, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv2, (35, 43, 46), (77, 255, 255))
cv.imshow("Mask", mask)
dst = cv.bitwise_and(hsv2, hsv2, mask=mask)
dst2 = cv.bitwise_not(hsv2, mask=mask)
cv.imshow("Dst", dst)
cv.imshow("Dst2", dst2)
```

#  Day10

##  像素统计

####  知识点： 像素值统计

- 最小(min) 
- 最大(max)
  - (单通道归一化将结果作为网络输入)
- 均值(mean)  
  - (作为分割依据 二值分割)
- 标准方差(standard deviation)
  - (扫描图像异常使用)

#### API知识点

- 最大最小值minMaxLoc
- 计算均值与标准方差meanStdDev

###  Python实现

```python
min, max, minLoc, maxLoc = cv.minMaxLoc(src)
print("min: %.2f, max: %.2f"% (min, max))
print("min loc: ", minLoc)
print("max loc: ", maxLoc)

means, stddev = cv.meanStdDev(src)
print("mean: %.2f, stddev: %.2f"% (means, stddev))
src[np.where(src < means)] = 0
src[np.where(src > means)] = 255
cv.imshow("binary", src)
```

输入为灰度图像

np.where 对图像进行分割，即生成一个二值图像

###  C++实现

```c++
	double minVal; 
	double maxVal; 
	Point minLoc; 
	Point maxLoc;
	minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	printf("min: %.2f, max: %.2f \n", minVal, maxVal);
	printf("min loc: (%d, %d) \n", minLoc.x, minLoc.y);
	printf("max loc: (%d, %d)\n", maxLoc.x, maxLoc.y);


	src = imread("D:/vcprojects/images/test.png");
	Mat means, stddev;
	meanStdDev(src, means, stddev);
	printf("blue channel->> mean: %.2f, stddev: %.2f\n", means.at<double>(0, 0), stddev.at<double>(0, 0));
	printf("green channel->> mean: %.2f, stddev: %.2f\n", means.at<double>(1, 0), stddev.at<double>(1, 0));
	printf("red channel->> mean: %.2f, stddev: %.2f\n", means.at<double>(2, 0), stddev.at<double>(2, 0));
```

C++调用稍微麻烦，需要定义Point类

minMaxLoc传递定义变量的引用

meanStdDev读取一张三通道彩色图像，输出存入Mat中



meanStdDev输入图像为多通道时，以每个通道为基准计算均值和标准差

meanStdDev返回Scalar矩阵，大小为1X1

练习：

C++实现二值化

实现代码：

```c++
	int heigh = src.rows;
	int width = src.cols;

	for(int row = 0; row<heigh; row++)
		for (int col = 0; col < width; col++)
		{
			Vec3b bgr = src.at<Vec3b>(row, col);
			bgr[0] = bgr[0] < means.at<double>(0, 0) ? 0 : 255;
			bgr[1] = bgr[1] < means.at<double>(1, 0) ? 0 : 255;
			bgr[2] = bgr[2] < means.at<double>(2, 0) ? 0 : 255;
			src.at<Vec3b>(row, col) = bgr;
		}
	imshow("Binary", src);
```

#  Day11

##  图像归一化

#### 知识点： 像素归一化

OpenCV中提供了四种归一化的方法

- NORM_MINMAX(最常用)
- NORM_INF
- NORM_L1
- NORM_L2
  最常用的就是NORM_MINMAX归一化方法。

####  相关API函数：

```C++
normalize(
InputArray 	src, // 输入图像
InputOutputArray 	dst, // 输出图像
double 	alpha = 1, // NORM_MINMAX时候低值
double 	beta = 0, // NORM_MINMAX时候高值
int 	norm_type = NORM_L2, // 只有alpha
int 	dtype = -1, // 默认类型与src一致
InputArray 	mask = noArray() // mask默认值为空
)	
```



图像归一化前，需要进行类型转换，归一化后会产生浮点数，而默认的int8无法存储，会造成数据错误

![归一化](https://image.nuccombat.cn/images/2019/03/31/FqoQhob7D9hwpLWT4Vap0Rlh65HQe1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zD07nkqFzGmc2USSHA8GWo2c9iq5w.png)

###  Python实现

```python
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# 转换为浮点数类型数组
gray = np.float32(gray)
print(gray)

# scale and shift by NORM_MINMAX
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst=dst, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX)
print(dst)
cv.imshow("NORM_MINMAX", np.uint8(dst*255))

# scale and shift by NORM_INF
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst=dst, alpha=1.0, beta=0, norm_type=cv.NORM_INF)
print(dst)
cv.imshow("NORM_INF", np.uint8(dst*255))

# scale and shift by NORM_L1
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst=dst, alpha=1.0, beta=0, norm_type=cv.NORM_L1)
print(dst)
cv.imshow("NORM_L1", np.uint8(dst*10000000))

# scale and shift by NORM_L2
dst = np.zeros(gray.shape, dtype=np.float32)
cv.normalize(gray, dst=dst, alpha=1.0, beta=0, norm_type=cv.NORM_L2)
print(dst)
cv.imshow("NORM_L2", np.uint8(dst*10000))
```

* NORM_MINMAX	
  * 必须做float32的转换
  * alpha beta对应最小值最大值
  * float32矩阵输出使用*255进行输出
* NORM_INF
  * 最大值归一化时只有alpha有意义

###  C++实现

```c++
	Mat gray, gray_f;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	// 转换为浮点数类型数组
	gray.convertTo(gray, CV_32F);

	// scale and shift by NORM_MINMAX
	Mat dst = Mat::zeros(gray.size(), CV_32FC1);
	normalize(gray, dst, 1.0, 0, NORM_MINMAX);
	Mat result = dst * 255;
	result.convertTo(dst, CV_8UC1);
	imshow("NORM_MINMAX", dst);

	// scale and shift by NORM_INF
	normalize(gray, dst, 1.0, 0, NORM_INF);
	result = dst * 255;
	result.convertTo(dst, CV_8UC1);
	imshow("NORM_INF", dst);

	// scale and shift by NORM_L1
	normalize(gray, dst, 1.0, 0, NORM_L1);
	result = dst * 10000000;
	result.convertTo(dst, CV_8UC1);
	imshow("NORM_L1", dst);

	// scale and shift by NORM_L2
	normalize(gray, dst, 1.0, 0, NORM_L2);
	result = dst * 10000;
	result.convertTo(dst, CV_8UC1);
	imshow("NORM_L2", dst);
```

*  C++下使用converTo进行数据类型转换
* 如果不进行浮点数转换，系统会采取截断措施，造成无法显示

#  Day12

##  视频读写

#### 知识点 和API

VideoCapture 视频文件读取、摄像头读取、视频流读写

VideoWriter 视频写出、文件保存

- CAP_PROP_FRAME_HEIGHT

- CAP_PROP_FRAME_WIDTH

- CAP_PROP_FRAME_COUNT(帧数)

- CAP_PROP_FPS（每秒帧数）

  不支持音频编码与解码保存，不是一个音视频处理的库！主要是分析与解析视频内容。保存文件最大支持单个文件为2G

###  Python实现

```python
capture = cv.VideoCapture("D:/vcprojects/images/768x576.avi")
# capture = cv.VideoCapture(0) 打开摄像头
height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
count = capture.get(cv.CAP_PROP_FRAME_COUNT)
fps = capture.get(cv.CAP_PROP_FPS)
print(height, width, count, fps)
out = cv.VideoWriter("D:/test.mp4", cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), 15,
                     (np.int(width), np.int(height)), True)
//VideoWriter writer("F:/test.mp4", VideoWriter::fourcc('D', 'I', 'V', 'X'), fps, s, true);
//OpenCV4接口
while True:
    ret, frame = capture.read()
    if ret is True:
        cv.imshow("video-input", frame)
        out.write(frame)
        c = cv.waitKey(50)
        if c == 27: # ESC
            break
    else:
        break

capture.release()
out.release()
```

VideoWriter()：

*  第一个参数输出路径
* 第二个参数视频编码格式
* 第三个参数帧数
* 第四个参数视频尺寸
* 是否边处理边解码

capture.read()方法

* 第一个返回值：ret布尔类型，标记是否结尾， False代表视频结束，True表示未结束
* 第二个返回值：获取的视频帧

waitKey（）方法

在本代码中暂停50ms，如果接受到Esc结束视频读取

*需要手动释放视频流！*

### C++实现

```C++
	// 打开摄像头
	// VideoCapture capture(0); 

	// 打开文件
	VideoCapture capture;
	capture.open("D:/vcprojects/images/768x576.avi");
	if (!capture.isOpened()) {
		printf("could not read this video file...\n");
		return -1;
	}
	Size S = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	int fps = capture.get(CV_CAP_PROP_FPS);
	printf("current fps : %d \n", fps);
	VideoWriter writer("D:/test.mp4", CV_FOURCC('D', 'I', 'V', 'X'), fps, S, true);

	Mat frame;
	namedWindow("camera-demo", CV_WINDOW_AUTOSIZE);
	while (capture.read(frame)) {
		imshow("camera-demo", frame);
		writer.write(frame);
		char c = waitKey(50);
		if (c == 27) {
			break;
		}
	}
	capture.release();
	writer.release();
```

C++基本与Python一致

注意:

*OpenCV4下VideoWriter第二个参数fourcc改为了VideoWriter::fourcc*



#  Day13

##  图像翻转

####   图像翻转(Image Flip)
图像翻转的本质像素映射，OpenCV支持三种图像翻转方式

- X轴翻转，flipcode = 0
- Y轴翻转, flipcode = 1
- XY轴翻转, flipcode = -1

####  相关的API
flip

- src输入参数
- dst 翻转后图像（Python使用接受返回值，C++需要预先定义）
- flipcode

本质是将像素点进行映射

![图像翻转 左上为原图 右上为沿X轴翻转 左下沿Y轴翻转 右下沿XY轴翻转](https://image.nuccombat.cn/images/2019/04/02/75e1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDDuLi-0sG0UiJc8Q0Hfl7ULQeeFQ.jpg)

###  C++实现

```c++
	Mat dst;
	// X Flip 倒影
	flip(src, dst, 0);
	imshow("x-flip", dst);

	// Y Flip 镜像
	flip(src, dst, 1);
	imshow("y-flip", dst);

	// XY Flip 对角
	flip(src, dst, -1);
	imshow("xy-flip", dst);
```

### Python实现

```python
# X Flip 倒影
dst1 = cv.flip(src, 0);
cv.imshow("x-flip", dst1);

# Y Flip 镜像
dst2 = cv.flip(src, 1);
cv.imshow("y-flip", dst2);

# XY Flip 对角
dst3 = cv.flip(src, -1);
cv.imshow("xy-flip", dst3);

# custom y-flip
h, w, ch = src.shape
dst = np.zeros(src.shape, src.dtype)
for row in range(h):
    for col in range(w):
        b, g, r = src[row, col]
        dst[row, w - col - 1] = [b, g, r]
cv.imshow("custom-y-flip", dst)
```

custom为自己实现



###  应用场景

Android前置摄像头调整

某些摄像头调整

#  Day14

##  图像插值

####  最常见四种插值算法

图像放大缩小时处理小数

e.g 图像变为原来的0.75倍

* INTER_NEAREST = 0(临近点插值)

  * 图像映射新值时，用于计算新像素值在原坐标对应的值

    10的图像缩放为0.75倍时 新图应取原图13位置的像素（13.3)

    从整体图像上考虑

* INTER_LINEAR = 1（双线性插值）

  考虑X和Y轴，距离越近权重越大 

  e.g 新像素点距离13 为0.3 14位0.7 根据距离计算权重 先考虑X轴再考虑Y轴，之后合并

  实质为图像的一阶求导差异值

* INTER_CUBIC = 2(双立方插值)

  多项式拟合，三阶导数，求图像周围16个点坐标，计算量较大

* INTER_LANCZOS4 = 4(卢卡斯插值)

  能量场插值，求周围点权重，与原像素值求均值

####  应用场景

* 几何变换
  * 透视 畸变 缩放等会产生新像素的情况
  * 双立方和卢卡斯插值图像质量较好，临近点速度快
* 透视变换
* 插值计算新像素

####  相关API

*resize*

实现图像放缩

参数：

原图像、输出图像、Size、x轴缩放大小、y轴缩放大小、插值方法

Size优先级比缩放因子高

###  C++实现

```c++
	int h = src.rows;
	int w = src.cols;
	float fx = 0.0, fy = 0.0;
	Mat dst = Mat::zeros(src.size(), src.type());
	resize(src, dst, Size(w * 2, h * 2), fx = 0, fy = 0, INTER_NEAREST);
	imshow("INTER_NEAREST", dst);

	resize(src, dst, Size(w * 2, h * 2), fx = 0, fy = 0, INTER_LINEAR);
	imshow("INTER_LINEAR", dst);

	resize(src, dst, Size(w * 2, h * 2), fx = 0, fy = 0, INTER_CUBIC); //双立方插值具有反锯齿功能 PS实现
	imshow("INTER_CUBIC", dst);

	resize(src, dst, Size(w * 2, h * 2), fx = 0, fy = 0, INTER_LANCZOS4);
	imshow("INTER_LANCZOS4", dst);
```

###  Python实现

```python
h, w = src.shape[:2]
print(h, w)
dst = cv.resize(src, (w*2, h*2), fx=0.75, fy=0.75, interpolation=cv.INTER_NEAREST)
cv.imshow("INTER_NEAREST", dst)

dst = cv.resize(src, (w*2, h*2), interpolation=cv.INTER_LINEAR)
cv.imshow("INTER_LINEAR", dst)

dst = cv.resize(src, (w*2, h*2), interpolation=cv.INTER_CUBIC)
cv.imshow("INTER_CUBIC", dst)

dst = cv.resize(src, (w*2, h*2), interpolation=cv.INTER_LANCZOS4)
cv.imshow("INTER_LANCZOS4", dst)

```

Size不为0时 fx fy不起作用 Size为0时 fx fy起作用

###  参考资料

 [图像处理之三种常见双立方插值算法]("https://blog.csdn.net/jia20003/article/details/40020775 ")

[图像放缩之双立方插值]("https://blog.csdn.net/jia20003/article/details/6919845 ")

[图像放缩之双线性内插值]("https://blog.csdn.net/jia20003/article/details/6915185 ")

[图像处理之Lanczos采样放缩算法]("https://blog.csdn.net/jia20003/article/details/17856859 ")



#  Day15

##  几何形状的绘制

####  绘制几何形状

- 绘制直线

- 绘制圆

- 绘制矩形

- 绘制椭圆

  

- 填充几何形状

  - OpenCV没有专门的填充方法，只是把绘制几何形状时候的线宽 - thickness参数值设置为负数即表示填充该几何形状或者使用参数CV_FILLED

####  随机数方法

RNG 表示OpenCV C++版本中的随机数对象，rng.uniform(a, b)生成[a, b)之间的随机数，包含a，但是不包含b。
np.random.rand() 表示numpy中随机数生成，生成浮点数0～1的随机数, 包含0，不包含1。



####  应用

主要用于标注

###  Python实现

```Python
cv.rectangle(image, (100, 100), (300, 300), (255, 0, 0), 2, cv.LINE_8, 0)
# 矩形
# 第一个参数 输入图像 第二个参数 矩形绘制相对于image的坐标 左上角点 第三个参数 右下角点 第四个参数 绘制颜色 第五个参数 线宽 两个像素 第五个参数 采取的绘制方法 LINE_8 八领域方法 LINE_AA 反锯齿方法 第六个参数 相对位移
cv.circle(image, (256, 256), 50, (0, 0, 255), 2, cv.LINE_8, 0)
# 圆心
# 第二个参数指定圆心位置 第三个参数 半径 其他同矩形
cv.ellipse(image, (256, 256), (150, 50), 360, 0, 360, (0, 255, 0), 2, cv.LINE_8, 0)
# 椭圆
# 第三个参数 长轴和短轴参数 第四个参数 角度 第五个参数 起始角度 第六个参数结束角度 其他同上
cv.imshow("image", image)
cv.waitKey(0)

for i in range(100000):
    image[:,:,:]= 0
    # 擦除上次绘制结果
    x1 = np.random.rand() * 512
    y1 = np.random.rand() * 512
    x2 = np.random.rand() * 512
    y2 = np.random.rand() * 512
    # rand返回0-1的随机数

    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    # cv.line(image, (np.int(x1), np.int(y1)), (np.int(x2), np.int(y2)), (b, g, r), 4, cv.LINE_8, 0)
    cv.rectangle(image, (np.int(x1), np.int(y1)), (np.int(x2), np.int(y2)), (b, g, r), 1, cv.LINE_8, 0)
    cv.imshow("image", image)
    c = cv.waitKey(20)
    if c == 27:
        break  # ESC不
```

### C++实现

```c++
Mat image = Mat::zeros(Size(512, 512), CV_8UC3);
	Rect rect(100, 100, 200, 200);
	rectangle(image, rect, Scalar(255, 0, 0), 2, LINE_8, 0);
	circle(image, Point(256, 256), 50, Scalar(0, 0, 255), 2, LINE_8, 0);
	ellipse(image, Point(256, 256), Size(150, 50), 360, 0, 360, Scalar(0, 255, 0), 2, LINE_8, 0);
	imshow("image", image);
	waitKey(0);

	RNG rng(0xFFFFFF);
	// 随机数种子
	image.setTo(Scalar(0, 0, 0));
	// 生成黑色背景
	for (int i = 0; i < 100000; i++) {
		// image.setTo(Scalar(0, 0, 0));
		int x1 = rng.uniform(0, 512);
		int y1 = rng.uniform(0, 512);
		int x2 = rng.uniform(0, 512);
		int y2 = rng.uniform(0, 512);

		int b = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int r = rng.uniform(0, 256);
		line(image, Point(x1, y1), Point(x2, y2), Scalar(b, g, r), 1, LINE_AA, 0);
		rect.x = x1;
		rect.y = y1;
		rect.width = x2 - x1;
		rect.height = y2 - y1;
		// rectangle(image, rect, Scalar(b, g, r), 1, LINE_AA, 0);
		imshow("image", image);
		char c = waitKey(20);
		if (c == 27)
			break;

		imshow("image", image);
	}
```



#  Day16

##  图像ROI与ROI操作

#### 图像ROI
图像的ROI(region of interest)是指图像中感兴趣区域、在OpenCV中图像设置图像ROI区域，实现只对ROI区域操作。

1. 矩形ROI区域提取
2. 矩形ROI区域copy

3. 不规则ROI区域
- ROI区域mask生成
- 像素位 and操作
- 提取到ROI区域
- 加背景or操作
- add 背景与ROI区域

###  Python实现

```python
h, w = src.shape[:2]

# 获取ROI
cy = h//2
cx = w//2
roi = src[cy-100:cy+100,cx-100:cx+100,:]
# roi内存区域依然是src
cv.imshow("roi", roi)

# copy ROI
image = np.copy(roi)

# modify ROI
roi[:, :, 0] = 0
cv.imshow("result", src)

# modify copy roi
image[:, :, 2] = 0
cv.imshow("result", src)
cv.imshow("copy roi", image)

# 提取不规则ROI
# example with ROI - generate mask
src2 = cv.imread("D:/javaopencv/tinygreen.png");
cv.imshow("src2", src2)
hsv = cv.cvtColor(src2, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, (35, 43, 46), (99, 255, 255))

# extract person ROI
mask = cv.bitwise_not(mask)
person = cv.bitwise_and(src2, src2, mask=mask);

# generate background
result = np.zeros(src2.shape, src2.dtype)
result[:,:,0] = 255

# 绿底换蓝底
# combine background + person
mask = cv.bitwise_not(mask)
dst = cv.bitwise_or(person, result, mask=mask)
dst = cv.add(dst, person)
```

不规则ROI区域提取：

* 生成ROI区域mask(重点)
* 利用原图与Mask与操作提取
  * mask白色区域为保留下来的区域
  * 方便均值 直方图计算等
  * 变换背景图像

###  C++实现

```c++
	// 获取ROI
	int cy = h / 2;
	int cx = w / 2;
	Rect rect(cx - 100, cy - 100, 200, 200);
	Mat roi = src(rect);
	imshow("roi", roi);

	Mat image = roi.clone();
	// modify ROI
	roi.setTo(Scalar(255, 0, 0));
	imshow("result", src);

	// modify copy roi
	image.setTo(Scalar(0, 0, 255));
	imshow("result", src);
	imshow("copy roi", image);

	// example with ROI - generate mask
	Mat src2 = imread("D:/javaopencv/tinygreen.png");
	imshow("src2", src2);
	Mat hsv, mask;
	cvtColor(src2, hsv, COLOR_BGR2HSV);
	inRange(hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), mask);
	imshow("mask", mask);

	// extract person ROI
	Mat person;
	bitwise_not(mask, mask);
	bitwise_and(src2, src2, person, mask);
	imshow("person", person);

	// generate background
	Mat result = Mat::zeros(src2.size(), src2.type());
	result.setTo(Scalar(255, 0, 0));

	// combine background + person
	Mat dst;
	bitwise_not(mask, mask);
	bitwise_or(person, result, dst, mask);
	add(dst, person, dst);
```



#  Day17

##  图像直方图

####  图像直方图的解释

![](https://image.nuccombat.cn/images/2019/04/07/75e1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDpRcznf180gkcls2bDAxmxhGR6P4.jpg)

图像直方图是图像像素值的统计学特征、计算代价较小，具有图像平移、旋转、缩放不变性等众多优点，广泛地应用于图像处理的各个领域，特别是灰度图像的阈值分割、基于颜色的图像检索以及图像分类、反向投影跟踪。常见的分为

- 灰度直方图
- 颜色直方图

Bins是指直方图的大小范围， 对于像素值取值在0～255之间的，最少有256个bin，此外还可以有16、32、48、128等，256除以bin的大小应该是整数倍。

一个Bins即为图像总的像素点个数

图像直方图特征具有平移缩放旋转不变形

利用归一化可以很好得出统计特征

利用直方图可以进行灰度图像的分割，转变为二值图像

利用直方图进行颜色检索 相似图像分类 图像的反向投影 图像均衡化等等

####  OpenCV中相关API
* calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges)
* cv.calcHist([image], [i], None, [256], [0, 256])

###  Python实现

```python
def custom_hist(gray):
    h, w = gray.shape
    hist = np.zeros([256], dtype=np.int32)
    for row in range(h):
        for col in range(w):
            pv = gray[row, col]
            hist[pv] += 1

    y_pos = np.arange(0, 256, 1, dtype=np.int32)
    plt.bar(y_pos, hist, align='center', color='r', alpha=0.5)
    plt.xticks(y_pos, y_pos)
    plt.ylabel('Frequency')
    plt.title('Histogram')

    # plt.plot(hist, color='r')
    # plt.xlim([0, 256])
    plt.show()


def image_hist(image):
    cv.imshow("input", image)
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


src = cv.imread("D:/vcprojects/images/flower.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imshow("input", gray)
# custom_hist(gray)
image_hist(src)
cv.waitKey(0)
cv.destroyAllWindows()
```



###  C++实现

```c++
void showHistogram() {
	// 三通道分离
	vector<Mat> bgr_plane;
	split(src, bgr_plane);
	// 定义参数变量
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };
	Mat b_hist;
	Mat g_hist;
	Mat r_hist;
	// 计算Blue, Green, Red通道的直方图
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
	// 显示直方图
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	// 归一化直方图数据
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	// 绘制直方图曲线
	for (int i = 1; i < bins[0]; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}
	// 显示直方图
	namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
	imshow("Histogram Demo", histImage);
```

- cvRound()：返回跟参数最接近的整数值，即四舍五入；
- cvFloor()：返回不大于参数的最大整数值，即向下取整；
- cvCeil()：返回不小于参数的最小整数值，即向上取整；



#  Day18

##  图像直方图均衡化

图像直方图均衡化可以用于图像增强、对输入图像进行直方图均衡化处理，提升后续对象检测的准确率在OpenCV人脸检测的代码演示中已经很常见。此外对医学影像图像与卫星遥感图像也经常通过直方图均衡化来提升图像质量。

OpenCV中，人脸检测Demo中，利用直方图均衡化对级联检测器的输入进行优化的效果

####  相关API

equalizeHist(src, dst)

src: 灰度图像

dst:输入图像

####  直方图均衡化理解

![](https://image.nuccombat.cn/images/2019/04/07/75e1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDtQCFqrBh3mKrJkF_fcifgdZScVI.jpg)

K：八个级别

Rk：级别值

Nk：某一级别中含有的像素点数量

Nk/N：占比

第五列：计算新像素应匹配的值

第六列：求新分配后像素比重

###  Python实现

````python
src = cv.imread("D:/vcprojects/images/flower.png")
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", gray)
dst = cv.equalizeHist(gray)
cv.imshow("eh", dst)
````

###  C++实现

```c++
	Mat gray, dst;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	imshow("input", gray);
	equalizeHist(gray, dst);
	imshow("eq", dst);
```

# Day19

## 图像直方图比较

 图像直方图比较，就是计算两幅图像的直方图数据，比较两组数据的相似性，从而得到两幅图像之间的相似程度，直方图比较在早期的CBIR（以图搜图）中是应用很常见的技术手段，通常会结合边缘处理、词袋等技术一起使用。

###  相关API

OpenCV中直方图比较的API很简单

compareHist(hist1, hist2, method)
-常见比较方法有

* 相关性 
  * 常用，相关程度越高值越接近于1，程度越低越接近-1，未归一化
* 卡方
  * 欧氏距离，未归一化
* 交叉
  * 取最小值，准确度较低
* 巴氏
  * 常用，OpenCV中越接近于0越相似，越接近于1越不相似，实际中想反（OpenCV对公式进行了修正）

相关性计算

![](https://image.nuccombat.cn/images/2019/04/26/FnTXyRpjchuCq_uip5Pi4cOI0PADe1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDGS92peJEe1Q6vDXPWDp1HfMX13w.png)

###  Python实现

```python
hsv1 = cv.cvtColor(src1, cv.COLOR_BGR2HSV)
hsv2 = cv.cvtColor(src2, cv.COLOR_BGR2HSV)
hsv3 = cv.cvtColor(src3, cv.COLOR_BGR2HSV)
hsv4 = cv.cvtColor(src4, cv.COLOR_BGR2HSV)

hist1 = cv.calcHist([hsv1], [0, 1], None, [60, 64], [0, 180, 0, 256])
# [0, 1] 选择 0 1通道 None mask参数 [60, 64] bins 最后一个参数取值范围
hist2 = cv.calcHist([hsv2], [0, 1], None, [60, 64], [0, 180, 0, 256])
hist3 = cv.calcHist([hsv3], [0, 1], None, [60, 64], [0, 180, 0, 256])
hist4 = cv.calcHist([hsv4], [0, 1], None, [60, 64], [0, 180, 0, 256])

cv.normalize(hist1, hist1, 0, 1.0, cv.NORM_MINMAX, dtype=np.float32)
# 浮点数归一化需要明示dtype
cv.normalize(hist2, hist2, 0, 1.0, cv.NORM_MINMAX)
cv.normalize(hist3, hist3, 0, 1.0, cv.NORM_MINMAX)
cv.normalize(hist4, hist4, 0, 1.0, cv.NORM_MINMAX)

methods = [cv.HISTCMP_CORREL, cv.HISTCMP_CHISQR,
           cv.HISTCMP_INTERSECT, cv.HISTCMP_BHATTACHARYYA]
str_method = ""
for method in methods:
    src1_src2 = cv.compareHist(hist1, hist2, method)
    src3_src4 = cv.compareHist(hist3, hist4, method)
    if method == cv.HISTCMP_CORREL:
        str_method = "Correlation"
    if method == cv.HISTCMP_CHISQR:
        str_method = "Chi-square"
    if method == cv.HISTCMP_INTERSECT:
        str_method = "Intersection"
    if method == cv.HISTCMP_BHATTACHARYYA:
        str_method = "Bhattacharyya"
```

直方图比较 直方图反向投影 经验性处理需要将色彩空间转化为HSV 最有用的通道为H S通道

###  C++实现

```C++
	Mat hsv1, hsv2, hsv3, hsv4;
	cvtColor(src1, hsv1, COLOR_BGR2HSV);
	cvtColor(src2, hsv2, COLOR_BGR2HSV);
	cvtColor(src3, hsv3, COLOR_BGR2HSV);
	cvtColor(src4, hsv4, COLOR_BGR2HSV);

	int h_bins = 60; int s_bins = 64;
	int histSize[] = { h_bins, s_bins };
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0, 1 };
	Mat hist1, hist2, hist3, hist4;
	calcHist(&hsv1, 1, channels, Mat(), hist1, 2, histSize, ranges, true, false);
	calcHist(&hsv2, 1, channels, Mat(), hist2, 2, histSize, ranges, true, false);
	calcHist(&hsv3, 1, channels, Mat(), hist3, 2, histSize, ranges, true, false);
	calcHist(&hsv4, 1, channels, Mat(), hist4, 2, histSize, ranges, true, false);

	normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
	normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());
	normalize(hist3, hist3, 0, 1, NORM_MINMAX, -1, Mat());
	normalize(hist4, hist4, 0, 1, NORM_MINMAX, -1, Mat());

	for (int i = 0; i < 4; i++)
	{
		int compare_method = i;
		double src1_src2 = compareHist(hist1, hist2, compare_method);
		double src3_src4 = compareHist(hist3, hist4, compare_method);
		printf(" Method [%d]  : src1_src2 : %f, src3_src4: %f,  \n", i, src1_src2, src3_src4);
	}
```



#  Day20

## 图像直方图反向投影

文字解释：
图像直方图反向投影是通过构建指定模板图像的二维直方图空间与目标的二维直方图空间，进行直方图数据归一化之后， 进行比率操作，对所有得到非零数值，生成查找表对原图像进行像素映射之后，再进行图像模糊输出的结果。

直方图反向投影流程

- 计算直方图
- 计算比率R
- LUT查找表
- 卷积模糊
- 归一化输出

###  相关API

- calcBackProject

![](https://image.nuccombat.cn/images/2019/04/26/FspoceGUtiQDQ9kkZ58a5AeQ4pjde1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDzwrhN8raUyj6VjrwGlBwa4-umbo.png)

参数1：原图像

参数2： 要处理的图像数量

参数3： 要处理的通道

参数4： 输入图像

参数5： 输出图像

参数6：范围

参数7：缩放比例

直方图反向投影效果

![](https://image.nuccombat.cn/images/2019/04/26/75e1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDG_RqZTWcREhkkjpaaOupFWCJh7I.jpg)

###  Python实现

```Python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def back_projection_demo():
    sample = cv.imread("D:/javaopencv/sample.png")
    # hist2d_demo(sample)
    target = cv.imread("D:/javaopencv/target.png")
    # hist2d_demo(target)
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    # show images
    cv.imshow("sample", sample)
    cv.imshow("target", target)

    roiHist = cv.calcHist([roi_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([target_hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    cv.imshow("backProjectionDemo", dst)


def hist2d_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    dst = cv.resize(hist, (400, 400))
    cv.imshow("image", image)
    cv.imshow("hist", dst)
    plt.imshow(hist, interpolation='nearest')
    plt.title("2D Histogram")
    plt.show()

```

###  C++实现

```c++
const int bins = 256;
Mat src;
const char* winTitle = "input image";
void backProjection_demo(Mat &image, Mat &model);
int main(int argc, char** argv) {
	Mat src = imread("D:/javaopencv/target.png");
	Mat model = imread("D:/javaopencv/sample.png");
	if (src.empty() || model.empty()) {
		printf("could not load image...\n");
		return 0;
	}
	namedWindow(winTitle, WINDOW_AUTOSIZE);
	imshow(winTitle, src);
	imshow("model", model);

	backProjection_demo(src, model);
	waitKey(0);
	return 0;
}

void backProjection_demo(Mat &image, Mat &model) {
	Mat model_hsv, image_hsv;
	cvtColor(model, model_hsv, COLOR_BGR2HSV);
	cvtColor(image, image_hsv, COLOR_BGR2HSV);

	// 定义直方图参数与属性
	int h_bins = 32; int s_bins = 32;
	int histSize[] = { h_bins, s_bins };
	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0, 1 };
	Mat roiHist;
	calcHist(&model_hsv, 1, channels, Mat(), roiHist, 2, histSize, ranges, true, false);
	normalize(roiHist, roiHist, 0, 255, NORM_MINMAX, -1, Mat());
	MatND backproj;
	calcBackProject(&image_hsv, 1, channels, roiHist, backproj, ranges, 1.0);
	imshow("BackProj", backproj);
}
```





#  Day21

##  图像卷积操作

​       图像卷积可以看成是一个窗口区域在另外一个大的图像上移动，对每个窗口覆盖的区域都进行点乘得到的值作为中心像素点的输出值。窗口的移动是从左到右，从上到下。窗口可以理解成一个指定大小的二维矩阵，里面有预先指定的值。

### 相关API

* C++版本
  * blur(
    InputArray 	src, // 输入
    OutputArray 	dst, 输出
    Size 	ksize, // 窗口大小
    Point 	anchor = Point(-1,-1), // 默认值
    int 	borderType = BORDER_DEFAULT // 默认值
    )

* Python版本
  * dst	= cv.blur(	src, ksize[, dst[, anchor[, borderType]]]	)

![](https://image.nuccombat.cn/images/2019/04/28/FgNmE4eNH09Q2sysTYU0pkKpDq7We1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDlyClBaKe6LNEwF2mWU-uN7bhNWA.png)

卷积核需要和窗口完全重合才能运算，卷积中间值为卷积核和窗口的点积之和求平均得到，TF中，为了处理边缘窗口，采用补零操作

### C++实现

```C++
	int h = src.rows;
	int w = src.cols;

	// 3x3 均值模糊，自定义版本实现
	Mat dst = src.clone();
	for (int row = 1; row < h-1; row++) {
		for (int col = 1; col < w-1; col++) {
			Vec3b p1 = src.at<Vec3b>(row-1, col-1);
			Vec3b p2 = src.at<Vec3b>(row-1, col);
			Vec3b p3 = src.at<Vec3b>(row-1, col+1);
			Vec3b p4 = src.at<Vec3b>(row, col-1);
			Vec3b p5 = src.at<Vec3b>(row, col);
			Vec3b p6 = src.at<Vec3b>(row, col+1);
			Vec3b p7 = src.at<Vec3b>(row+1, col-1);
			Vec3b p8 = src.at<Vec3b>(row+1, col);
			Vec3b p9 = src.at<Vec3b>(row+1, col+1);

			int b = p1[0] + p2[0] + p3[0] + p4[0] + p5[0] + p6[0] + p7[0] + p8[0] + p9[0];
			int g = p1[1] + p2[1] + p3[1] + p4[1] + p5[1] + p6[1] + p7[1] + p8[1] + p9[1];
			int r = p1[2] + p2[2] + p3[2] + p4[2] + p5[2] + p6[2] + p7[2] + p8[2] + p9[2];

			dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b / 9);
			dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g / 9);
			dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r / 9);
		}
	}
	imshow("blur", dst);
	imwrite("D:/result.png", dst);

	// OpenCV 均值模糊
	Mat result;
	blur(src, result, Size(15, 15), Point(-1, -1), 4);
	// Size为卷积核大小 OpenCV中采用积分图来进行优化计算 Point为中心像素点位置 4位边缘处理方法
	imshow("result", result);
```

###  Python

```python
def custom_blur(src):
    h, w, ch = src.shape
    print("h , w, ch", h, w, ch)
    result = np.copy(src)
    for row in range(1, h-1, 1):
        for col in range(1, w-1, 1):
            v1 = np.int32(src[row-1, col-1])
            v2 = np.int32(src[row-1, col])
            v3 = np.int32(src[row-1, col+1])
            v4 = np.int32(src[row, col-1])
            v5 = np.int32(src[row, col])
            v6 = np.int32(src[row, col+1])
            v7 = np.int32(src[row+1, col-1])
            v8 = np.int32(src[row+1, col])
            v9 = np.int32(src[row+1, col+1])

            b = v1[0] + v2[0] + v3[0] + v4[0] + v5[0] + v6[0] + v7[0] + v8[0] + v9[0];
            g = v1[1] + v2[1] + v3[1] + v4[1] + v5[1] + v6[1] + v7[1] + v8[1] + v9[1];
            r = v1[2] + v2[2] + v3[2] + v4[2] + v5[2] + v6[2] + v7[2] + v8[2] + v9[2];
            result[row, col] = [b//9, g//9, r//9]
    cv.imshow("result", result)


src = cv.imread("D:/vcprojects/images/lena.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
dst = cv.blur(src, (15, 15))
cv.imshow("blur", dst)
custom_blur(src)
```



* 卷积核一般为奇数，偶数为少部分情况，中心需要自定义

#  Day22

##  图像均值与高斯模糊

​       均值模糊 是卷积核的系数完全一致，高斯模糊考虑了中心像素距离的影响，对距离中心像素使用高斯分布公式生成不同的权重系数给卷积核，然后用此卷积核完成图像卷积得到输出结果就是图像高斯模糊之后的输出。

###  相关API

```c++
void GaussianBlur(
InputArray src, 
OutputArray dst, 
Size ksize, // Ksize为高斯滤波器窗口大小 Size为（0,0）将会从SimgaX开始计算核函数，不为零时，会从Size开始计算核函数
double sigmaX, // X方向滤波系数
double sigmaY=0, // Y方向滤波系数 sigmaY默认时，自动等于sigmaX
    // sigma越大模糊越厉害 窗口越大模糊越厉害
int borderType=BORDER_DEFAULT // 默认边缘插值方法
)
```

当Size(0, 0)就会从sigmax开始计算生成高斯卷积核系数，当时size不为零是优先从size开始计算高斯卷积核系数

![](https://image.nuccombat.cn/images/2019/04/28/Fu9US8uN_k6wN1sWjCfG1f-jmbcae1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDF5ScKMtVUM9WrOluiie8prE9x_I.png)

高斯模糊卷积核的权重根据到中心距离生成（依赖高斯分布生成）

![](https://image.nuccombat.cn/images/2019/04/28/Fvp2G4cWEafYn55baTU8i80vqsjUe1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDvuqDa1dBhIc3QZddy910hrmhg0A.png)

极值出现在（x, y）=(0, 0)

高斯核为3X3时，取值为X-1到X+1 Y-1到Y+1，上图表格左图为高斯核函数原始值，右图为归一化结果，高斯模糊和均值模糊相比更清晰（中心权重更高）

###  C++实现

```c++
	Mat dst1, dst2;
	blur(src, dst1, Size(5, 5), Point(-1, -1), 4);
	GaussianBlur(src, dst2, Size(5, 5), 15, 0, 4);


	imshow("blur", dst1);
	imshow("gaussian blur", dst2);
```

###  Python实现

```Python
dst1 = cv.blur(src, (5, 5))
dst2 = cv.GaussianBlur(src, (5, 5), sigmaX=15)
dst3 = cv.GaussianBlur(src, (0, 0), sigmaX=15)
```





# Day23

## 中值滤波

​        中值滤波本质上是统计排序滤波器的一种，中值滤波对图像特定噪声类型（椒盐噪声）会取得比较好的去噪效果，也是常见的图像去噪声与增强的方法之一。中值滤波也是窗口在图像上移动，其覆盖的对应ROI区域下，所有像素值排序，取中值作为中心像素点的输出值

其余的统计学滤波器：极大值 极小值滤波器

中值滤波的原理：

![](https://image.nuccombat.cn/images/2019/04/30/FiCSQI5eLGFordElFpylXiCSqJ0Ie1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDLVLljjEsXLc3Xcb1qDaAksSYfkQ.png)

![](https://image.nuccombat.cn/images/2019/04/30/FiH3JLrkVwpjENpk43O3X-74M49Qe1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zDh6dmIxLX4KgMgcEXuAxPEXLe5Sk.png)

## 相关API

OpenCV中值滤波API函数如下：

C++:

```c++
medianBlur	(	
InputArray 	src,
OutputArray 	dst,
int 	ksize // 必须是奇数，而且必须大于1
)

```

Python:

```python
dst = cv.medianBlur(	src, ksize[, dst]	)
```

###  Python实现

````python
dst = cv.medianBlur(src, 5)
cv.imshow("blur ksize=5", dst)
````

###  C++实现

```c++
	Mat dst;
	medianBlur(src, dst, 5);
	imshow("medianblur ksize=5", dst);
```



#  Day24

##  图像噪声

图像噪声产生的原因很复杂，有的可能是数字信号在传输过程中发生了丢失或者受到干扰，有的是成像设备或者环境本身导致成像质量不稳定，反应到图像上就是图像的亮度与颜色呈现某种程度的不一致性。从噪声的类型上，常见的图像噪声可以分为如下几种：

- 椒盐噪声， 
是一种随机在图像中出现的稀疏分布的黑白像素点， 对椒盐噪声一种有效的去噪手段就是图像中值滤波

- 高斯噪声/符合高斯分布
一般会在数码相机的图像采集(acquisition)阶段发生,这个时候它的物理/电/光等各种信号都可能导致产生高斯分布噪声。

- 均匀分布噪声
均匀/规则噪声一般都是因为某些规律性的错误导致的

##  代码演示

- 图像椒盐噪声生成
- 图像高斯噪声生成

### C++实现噪声

```c++
void add_salt_pepper_noise(Mat &image) {
	RNG rng(12345);
	int h = image.rows;
	int w = image.cols;
	int nums = 10000;
	for (int i = 0; i < nums; i++) {
		int x = rng.uniform(0, w);
		int y = rng.uniform(0, h);
		if (i % 2 == 1) {
			image.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
		}
		else {
			image.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
		}
	}
	imshow("salt pepper", image);
}

void gaussian_noise(Mat &image) {
	Mat noise = Mat::zeros(image.size(), image.type());
	randn(noise, (15, 15, 15), (30, 30, 30));
    // 15是产生的噪声均值，30是产生的噪声的方差
	Mat dst;
	add(image, noise, dst);
	imshow("gaussian noise", dst);
}
```



###  Python实现噪声

```python
def add_salt_pepper_noise(image):
    h, w = image.shape[:2]
    nums = 10000
    rows = np.random.randint(0, h, nums, dtype=np.int)
    cols = np.random.randint(0, w, nums, dtype=np.int)
    for i in range(nums):
        if i % 2 == 1:
            image[rows[i], cols[i]] = (255, 255, 255)
        else:
            image[rows[i], cols[i]] = (0, 0, 0)
    return image


def gaussian_noise(image):
    noise = np.zeros(image.shape, image.dtype)
    m = (15, 15, 15)
    s = (30, 30, 30)
    cv.randn(noise, m, s)
    dst = cv.add(image, noise)
    cv.imshow("gaussian noise", dst)
    return dst
```



#  Day25

##  图像去噪声

图像去噪声在OCR、机器人视觉与机器视觉领域应用开发中是重要的图像预处理手段之一，对图像二值化与二值分析很有帮助，OpenCV中常见的图像去噪声的方法有
- 均值去噪声
- 高斯模糊去噪声
- 非局部均值去噪声
- 双边滤波去噪声
- 形态学去噪声

![](https://image.nuccombat.cn/images/2019/04/30/FtXdxPyYuMD69s1nTDMnDZkEwoDme1906272000tokenkIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zD8xIuJARZsf4fJ3sl-NhC37ZmX1E.png)

均值 高斯滤波器一般选择3X3的卷积核，最多使用5X5的卷积核，过大的Kernel会导致图像细节丢失

###  Python实现

```python
src = cv.imread("D:/vcprojects/images/example.png")
cv.imshow("input", src)
h, w = src.shape[:2]
src = gaussian_noise(src)

result1 = cv.blur(src, (5, 5))
cv.imshow("result-1", result1)

result2 = cv.GaussianBlur(src, (5, 5), 0)
cv.imshow("result-2", result2)

result3 = cv.medianBlur(src, 5)
cv.imshow("result-3", result3)

result4 = cv.fastNlMeansDenoisingColored(src, None, 15, 15, 10, 30)
# 两个版本对应灰度图和多通道图， 15 15分别是颜色和亮度分量的阈值，一般设置为10，最多为15, 10 30搜索窗口和模板窗口大小 两者最常见比例为1比3（模板比搜索）模板窗口不能大过搜索窗口
cv.imshow("result-4", result4)
```

###  C++实现

```c++
	Mat result1, result2, result3, result4;
	blur(src, result1, Size(5, 5));
	imshow("result-1", result1);

	GaussianBlur(src, result2, Size(5, 5), 0);
	imshow("result-2", result2);

	medianBlur(src, result3, 5);
	// keysize为奇数并且一定大于零
	imshow("result-3", result3);

	fastNlMeansDenoisingColored(src, result4, 15, 15, 10, 30);
	imshow("result-4", result4);
```

