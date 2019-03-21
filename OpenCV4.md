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