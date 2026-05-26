/*
 * utils.h - 工具函数：数据集操作、随机数、统计
 *
 * Dataset: X[n_samples][n_features] 特征矩阵, y[n_samples] 标签数组
 * CSV 格式: 第一行表头，后续每行逗号分隔 (特征..., 标签)
 */
#ifndef UTILS_H
#define UTILS_H

typedef struct {
    double **X;        /* 二维特征数组，每一行是一条样本 */
    int *y;            /* 对于无标签的预测数据置为全 0 */
    int n_samples;     /* 样本数量 */
    int n_features;    /* 每条样本的特征数量 */
} Dataset;

/* 数据集操作 */
Dataset *dataset_load(const char *filename);
void dataset_free(Dataset *ds);
void dataset_bootstrap(Dataset *src, Dataset *dst);  /* 有放回采样，dst 需预分配内存 */

/* 统计函数 */
int argmax(int *vals, int n);                                   /* 返回最大值的索引 */
double gini_impurity(int *y, int n_samples, int n_classes);     /* CART 分裂准则 */

/* 随机数（首次调用自动播种） */
double rand_double(void);            /* [0, 1) */
int rand_int(int max);               /* [0, max) */
void shuffle_indices(int *indices, int n);    /* Fisher-Yates 洗牌 */
void shuffle_features(int *features, int n);  /* 随机森林特征采样前打乱特征索引 */

#endif
