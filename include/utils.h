// utils.h - 工具函数接口：数据集操作、随机数、统计函数
// CSV 格式：第一行表头，后续每行逗号分隔，最后一列为标签
#ifndef UTILS_H
#define UTILS_H

// 数据集结构体
typedef struct {
    double **X;        // 二维特征数组，每一行是一条样本
    int *y;            // 标签数组，对于无标签预测数据可置为 0
    int n_samples;     // 样本数量
    int n_features;    // 每条样本的特征数量
} Dataset;

// 数据集操作：从 CSV 文件加载数据
Dataset *dataset_load(const char *filename);
// 数据集操作：释放 Dataset 占用的内存
void dataset_free(Dataset *ds);
// 数据集操作：Bootstrap 有放回采样，dst 需要提前分配内存
void dataset_bootstrap(Dataset *src, Dataset *dst);

// 统计函数：返回数组中最大值的索引
int argmax(int *vals, int n);
// 统计函数：计算 Gini 不纯度，是 CART 决策树的分裂准则
double gini_impurity(int *y, int n_samples, int n_classes);

// 随机数函数：返回 [0, 1) 范围内的小数，首次调用自动播种
double rand_double(void);
// 随机数函数：返回 [0, max) 范围内的整数
int rand_int(int max);
// 洗牌函数：打乱样本索引
void shuffle_indices(int *indices, int n);
// 洗牌函数：打乱特征索引，用于随机森林特征采样
void shuffle_features(int *features, int n);

#endif
